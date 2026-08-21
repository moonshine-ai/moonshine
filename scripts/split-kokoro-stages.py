#!/usr/bin/env python3
"""Cut the Kokoro graph into a prosody stage and a decoder stage.

Kokoro ships as one graph: phoneme ids in, a whole waveform out. The graph
divides cleanly, with exactly three tensors crossing between the two halves:

    prosody: input_ids, style, speed  ->  asr [1,512,T], f0 [1,2T], n [1,2T]
    decoder: asr, f0, n, style        ->  waveform

Both stages keep ``style`` as an input, and the weights are partitioned rather
than duplicated, so the pair is about the size of the original.

In a quantized export the crossing tensors are uint8, which would make the
stage interface depend on which build you split. The three are given a float32
interface either by cutting one node earlier (where a ``QuantizeLinear``
produces the tensor) or by adding a dequantize/requantize pair at the seam,
reusing the scales already in the graph so the round trip is exact.

The reason for wanting the split was to run the decoder once per sub-sentence
chunk and start speaking sooner. That did not work — the decoder's output
depends on how much input it is handed, and no amount of context fixes it, so
the pieces do not reassemble into the render they came from. See the numbers in
scripts/kokoro-stream-prototype.py. Nothing shipped uses these stages; the
script is kept so the measurement can be repeated.

Usage:
    python scripts/split-kokoro-stages.py model.onnx --out-dir kokoro-stages
    python scripts/split-kokoro-stages.py model.onnx --out-dir out --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper

# The three tensors that cross the encoder/decoder boundary in the upstream
# Kokoro export. Named by what the PyTorch model calls them.
BOUNDARY = {
    "asr": "/encoder/MatMul_1_output_0",
    "f0": "/encoder/Reshape_output_0",
    "n": "/encoder/Reshape_1_output_0",
}

PROSODY_INPUTS = ["input_ids", "style", "speed"]
DECODER_INPUTS = ["asr", "f0", "n", "style"]


def find_boundary_tensors(model: onnx.ModelProto) -> dict[str, str]:
    """Confirm the three boundary tensors exist, by name or by graph shape.

    The literal names come from the upstream export. If a re-export renames
    them, fall back to the structure: the decoder's first node reads exactly
    three encoder outputs.
    """
    produced = {
        output for node in model.graph.node for output in node.output
    }
    if all(name in produced for name in BOUNDARY.values()):
        return dict(BOUNDARY)

    # Structural fallback: tensors produced under /encoder/ and consumed under
    # /decoder/ are, by definition, the cut set.
    producer = {
        output: node.name
        for node in model.graph.node
        for output in node.output
    }
    crossing = []
    for node in model.graph.node:
        if not node.name.startswith("/decoder/"):
            continue
        for name in node.input:
            owner = producer.get(name, "")
            if owner.startswith("/encoder/") and name not in crossing:
                crossing.append(name)
    if len(crossing) != 3:
        raise SystemExit(
            f"Expected 3 tensors crossing encoder->decoder, found {len(crossing)}: "
            f"{crossing}. The graph has changed shape; update BOUNDARY."
        )
    # Order matches the PyTorch call: decoder(asr, F0, N, style).
    return dict(zip(("asr", "f0", "n"), crossing))


def rename_tensor(model: onnx.ModelProto, old: str, new: str) -> None:
    """Rename a tensor everywhere it appears, so the stages join on clear names."""
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            if name == old:
                node.input[i] = new
        for i, name in enumerate(node.output):
            if name == old:
                node.output[i] = new
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            if value.name == old:
                value.name = new


# Ops that move quantized values around without changing their scale, so a
# consumer downstream of one still tells us how the tensor is quantized.
SHAPE_ONLY_OPS = ("Unsqueeze", "Squeeze", "Reshape", "Transpose", "Identity")


def quantization_params(model: onnx.ModelProto, tensor: str) -> tuple[str, str]:
    """The scale and zero-point a consumer already uses to dequantize ``tensor``.

    Reusing them means the dequantize we add on one side and the quantize on
    the other cancel exactly, so the split stays numerically identical. The
    nearest consumer is often a reshape rather than an arithmetic op, so follow
    those through: they carry the quantization unchanged.
    """
    consumers: dict[str, list] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    seen = set()
    frontier = [tensor]
    while frontier:
        current = frontier.pop(0)
        if current in seen:
            continue
        seen.add(current)
        for node in consumers.get(current, []):
            if node.op_type in ("DequantizeLinear", "QLinearConv", "QLinearMatMul"):
                if node.input[0] == current and len(node.input) >= 3:
                    return node.input[1], node.input[2]
            elif node.op_type in SHAPE_ONLY_OPS and node.input[0] == current:
                frontier.extend(node.output)
    raise SystemExit(
        f"Could not find the scale/zero-point for quantized boundary tensor {tensor}."
    )


def copy_initializers(
    source: onnx.ModelProto, target: onnx.ModelProto, names: list[str]
) -> None:
    existing = {init.name for init in target.graph.initializer}
    by_name = {init.name: init for init in source.graph.initializer}
    for name in names:
        if name in existing:
            continue
        if name not in by_name:
            raise SystemExit(f"Missing initializer {name} in the source model.")
        target.graph.initializer.append(by_name[name])


def plan_cut(model: onnx.ModelProto, boundary: dict[str, str]) -> dict[str, dict]:
    """Decide, per boundary tensor, where to cut and whether a seam is needed.

    A tensor produced by ``QuantizeLinear`` is cut one node earlier so the
    float value crosses and the quantize moves to the decoder. Anything else
    that is still integer gets an explicit dequantize/requantize seam.
    """
    producer = {output: node for node in model.graph.node for output in node.output}
    plan: dict[str, dict] = {}
    for friendly, tensor in boundary.items():
        node = producer.get(tensor)
        if node is not None and node.op_type == "QuantizeLinear":
            plan[friendly] = {"cut": node.input[0], "seam": None}
            continue
        if is_integer_tensor(model, tensor):
            scale, zero_point = quantization_params(model, tensor)
            plan[friendly] = {"cut": tensor, "seam": (scale, zero_point)}
            continue
        plan[friendly] = {"cut": tensor, "seam": None}
    return plan


def is_integer_tensor(model: onnx.ModelProto, tensor: str) -> bool:
    """True when ``tensor`` is uint8/int8 according to the graph's value info."""
    for collection in (model.graph.value_info, model.graph.output):
        for value in collection:
            if value.name == tensor:
                return value.type.tensor_type.elem_type in (
                    TensorProto.UINT8,
                    TensorProto.INT8,
                )
    # No declared type. Integer boundary tensors in these exports are always
    # produced by a quantizing op, so use the producer as the tell.
    for node in model.graph.node:
        if tensor in node.output:
            return node.op_type in ("QuantizeLinear", "Reshape", "Unsqueeze", "Squeeze") and any(
                is_integer_tensor(model, name) for name in node.input[:1]
            )
    return False


def extract(
    source: Path, destination: Path, inputs: list[str], outputs: list[str]
) -> None:
    onnx.utils.extract_model(
        str(source), str(destination), inputs, outputs, check_model=False
    )


def add_prosody_seams(
    stage: onnx.ModelProto, source: onnx.ModelProto, plan: dict[str, dict]
) -> None:
    """Dequantize integer outputs so prosody hands back float32."""
    for friendly, entry in plan.items():
        seam = entry["seam"]
        if seam is None:
            continue
        scale, zero_point = seam
        quantized = entry["cut"]
        copy_initializers(source, stage, [scale, zero_point])
        stage.graph.node.append(
            helper.make_node(
                "DequantizeLinear",
                [quantized, scale, zero_point],
                [f"{friendly}_dequantized"],
                name=f"moonshine_split_dequantize_{friendly}",
            )
        )
        # Swap the graph output over to the float tensor.
        for output in stage.graph.output:
            if output.name == quantized:
                output.name = f"{friendly}_dequantized"
                output.type.tensor_type.elem_type = TensorProto.FLOAT
        rename_tensor(stage, f"{friendly}_dequantized", friendly)


def add_decoder_seams(
    stage: onnx.ModelProto, source: onnx.ModelProto, plan: dict[str, dict]
) -> None:
    """Requantize float32 inputs so the decoder body sees what it expects."""
    inserted = []
    for friendly, entry in plan.items():
        seam = entry["seam"]
        if seam is None:
            continue
        scale, zero_point = seam
        quantized = entry["cut"]
        copy_initializers(source, stage, [scale, zero_point])
        for value in stage.graph.input:
            if value.name == quantized:
                value.name = friendly
                value.type.tensor_type.elem_type = TensorProto.FLOAT
        inserted.append(
            helper.make_node(
                "QuantizeLinear",
                [friendly, scale, zero_point],
                [quantized],
                name=f"moonshine_split_quantize_{friendly}",
            )
        )
    # Quantize nodes have to run before anything that reads their output.
    for node in reversed(inserted):
        stage.graph.node.insert(0, node)


def split(model_path: Path, out_dir: Path) -> tuple[Path, Path]:
    model = onnx.load(str(model_path), load_external_data=False)
    boundary = find_boundary_tensors(model)
    plan = plan_cut(model, boundary)
    for friendly, entry in plan.items():
        seam = "dequantize/requantize seam" if entry["seam"] else "direct"
        print(f"{friendly}: cut at {entry['cut']} ({seam})")

    out_dir.mkdir(parents=True, exist_ok=True)
    prosody_path = out_dir / "prosody.onnx"
    decoder_path = out_dir / "decoder.onnx"

    cuts = [plan[name]["cut"] for name in ("asr", "f0", "n")]
    extract(model_path, prosody_path, PROSODY_INPUTS, cuts)
    extract(model_path, decoder_path, cuts + ["style"], ["waveform"])

    prosody = onnx.load(str(prosody_path))
    add_prosody_seams(prosody, model, plan)
    decoder = onnx.load(str(decoder_path))
    add_decoder_seams(decoder, model, plan)

    # The extracted graphs still call the boundary tensors by their internal
    # names, which say nothing to a caller. Rename to asr / f0 / n on both sides.
    for stage in (prosody, decoder):
        for friendly, entry in plan.items():
            if entry["seam"] is None:
                rename_tensor(stage, entry["cut"], friendly)
    onnx.save(prosody, str(prosody_path))
    onnx.save(decoder, str(decoder_path))

    for path in (prosody_path, decoder_path):
        stage = onnx.load(str(path), load_external_data=False)
        print(
            f"{path.name}: {len(stage.graph.node)} nodes, "
            f"in={[i.name for i in stage.graph.input]}, "
            f"out={[o.name for o in stage.graph.output]}, "
            f"{path.stat().st_size / 1e6:.1f} MB"
        )
    return prosody_path, decoder_path


def verify(model_path: Path, prosody_path: Path, decoder_path: Path) -> None:
    """Check the two stages together reproduce the single graph's waveform."""
    import numpy as np
    import onnxruntime as ort

    def session(path: Path) -> ort.InferenceSession:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])

    whole = session(model_path)
    prosody = session(prosody_path)
    decoder = session(decoder_path)

    rng = np.random.default_rng(0)
    ids = np.zeros((1, 24), dtype=np.int64)
    ids[0, 1:-1] = rng.integers(1, 100, size=22)
    style = rng.standard_normal((1, 256)).astype(np.float32) * 0.1
    speed = np.array([1.0], dtype=np.float32)

    reference = whole.run(["waveform"], {"input_ids": ids, "style": style, "speed": speed})[0]
    stage_one = prosody.run(
        ["asr", "f0", "n"], {"input_ids": ids, "style": style, "speed": speed}
    )
    asr, f0, n = stage_one
    print(f"asr {asr.shape}, f0 {f0.shape}, n {n.shape}")
    if f0.shape[-1] != 2 * asr.shape[-1]:
        print(
            f"WARNING: f0 is {f0.shape[-1]} wide for asr length {asr.shape[-1]}; "
            "the 2T assumption the chunk slicer relies on does not hold."
        )
    combined = decoder.run(
        ["waveform"], {"asr": asr, "f0": f0, "n": n, "style": style}
    )[0]

    if reference.shape != combined.shape:
        raise SystemExit(
            f"Shape mismatch: whole graph {reference.shape}, stages {combined.shape}"
        )
    error = float(np.max(np.abs(reference - combined)))
    print(f"Max absolute difference over {reference.size} samples: {error:.3e}")
    if error > 1e-4:
        raise SystemExit("Stages do not reproduce the single graph.")

    # A partial decode is the whole point: check the decoder accepts a slice.
    half = asr.shape[-1] // 2
    piece = decoder.run(
        ["waveform"],
        {
            "asr": asr[:, :, :half],
            "f0": f0[:, : 2 * half],
            "n": n[:, : 2 * half],
            "style": style,
        },
    )[0]
    print(f"Half-length slice decoded to {piece.shape[-1]} samples (whole: {reference.shape[-1]})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Kokoro model.onnx to split")
    parser.add_argument(
        "--out-dir", type=Path, required=True, help="Where to write the two stages"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run both stages against the single graph and compare waveforms",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"No such model: {args.model}")

    prosody_path, decoder_path = split(args.model, args.out_dir)
    if args.verify:
        verify(args.model, prosody_path, decoder_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
