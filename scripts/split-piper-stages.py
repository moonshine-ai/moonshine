#!/usr/bin/env python3
"""Cuts a Piper voice into the two stages sub-sentence streaming needs.

A Piper voice is a VITS model: a body that turns phonemes into a frame-rate
latent, and a HiFi-GAN generator that turns that latent into audio. Streaming
below a sentence means running the body once and then asking the generator for
a range of frames at a time, so the first audio arrives before the rest of the
sentence has been decoded. That needs the two halves as separate models.

    upstream:  input, input_lengths, scales[, sid]  ->  latent [1, 192, T]
    generator: latent [1, 192, T][, speaker embedding]  ->  output [1, 1, S]

Run this on the **quantized** ``.onnx`` we publish, not on the float32 original
from ``rhasspy/piper-voices``. Splitting after quantization partitions the
weights that already ship, so the stages render what the monolithic voice
renders today and no voice changes as a side effect of gaining streaming. Get
one with, for example::

    curl -O https://download.moonshine.ai/tts/en_us/piper-voices/en_US-amy-medium.onnx

The generator is a stack of transposed convolutions with weight normalization,
so a slice of the latent decodes to the matching slice of audio once the
padding either side covers the receptive field. Sixteen frames is enough to
put the seam 1e-6 below the signal, which is why streamed Piper needs neither
the crossfade nor the offline gain table that Kokoro's decoder does. See
``scripts/piper-stream-prototype.py`` for the measurements.

Usage:
    python scripts/split-piper-stages.py en_US-amy-medium.onnx --verify
    python scripts/split-piper-stages.py voices/*.onnx --out-dir stages
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx

# Noise off, so a whole render and a staged one are the same utterance. Piper
# draws both noise terms per session, so leaving them on would compare two
# different draws and say the split was broken when it is not.
VERIFY_SCALES = [0.0, 1.0, 0.0]

# Frames decoded either side of a chunk and discarded. Measured, not guessed:
# 8 leaves 4e-4 of relative error at the seam and 16 leaves none.
PAD_FRAMES = 16


def depends_on_inputs(model: onnx.ModelProto) -> set[str]:
    """Tensors whose value depends on what the caller passed in.

    Everything else folds from initializers. The distinction matters on a
    quantized voice, where the generator's weights are dequantized by chains
    that sit outside it in the graph: those belong inside the generator stage,
    not crossing the seam on every utterance.
    """
    live = {value.name for value in model.graph.input}
    # ONNX requires nodes in topological order, so one pass reaches everything.
    for node in model.graph.node:
        if any(name in live for name in node.input):
            live.update(node.output)
    return live


def generator_nodes(model: onnx.ModelProto) -> set[int]:
    """Indices of the nodes making up the HiFi-GAN generator.

    Voices exported by newer PyTorch carry hierarchical node names and the
    generator's are the ones under ``/dec``, but older exports name every node
    flatly (``Conv_5947``) and that test finds nothing. What both keep is the
    parameter names from the checkpoint, where the generator's weights are the
    ones under ``dec.``. Seeding on those and following the data forward finds
    the same nodes either way.
    """
    seeded = set()
    produced_by_generator = set()
    for index, node in enumerate(model.graph.node):
        touches_generator_weight = any(
            name.startswith("dec.") for name in list(node.input) + list(node.output)
        )
        consumes_generator_value = any(
            name in produced_by_generator for name in node.input
        )
        if (touches_generator_weight or consumes_generator_value
                or node.name.startswith("/dec")):
            seeded.add(index)
            produced_by_generator.update(node.output)
    return seeded


def find_boundary(model: onnx.ModelProto) -> list[str]:
    """The tensors carrying the utterance from the VITS body into the generator.

    One is the latent, at one column per frame. A multi-speaker voice adds the
    speaker embedding, which is one column for the whole utterance and so is
    handed to every chunk unchanged.
    """
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    inside = generator_nodes(model)
    live = depends_on_inputs(model)
    crossing = []
    for index, node in enumerate(model.graph.node):
        if index not in inside:
            continue
        for name in node.input:
            source = producer.get(name)
            if source is None or source in inside:
                continue
            if name in live and name not in crossing:
                crossing.append(name)
    if not crossing:
        raise SystemExit(
            "Found no tensor entering the generator. This does not look like a "
            "Piper VITS export: its generator weights are named dec.*."
        )
    return crossing


def referenced_tensors(node: onnx.NodeProto) -> set[str]:
    """Every tensor a node reads, including from inside its subgraphs.

    A node holding an ``If`` reads tensors its branches name but its own inputs
    do not, and missing those is what makes a partitioned graph fail to load.
    """
    names = set(node.input)
    for attribute in node.attribute:
        graphs = []
        if attribute.HasField("g"):
            graphs.append(attribute.g)
        graphs.extend(attribute.graphs)
        for graph in graphs:
            for inner in graph.node:
                names |= referenced_tensors(inner)
    return names


def backward_closure(model: onnx.ModelProto, wanted: list[str],
                     stop: set[str]) -> set[int]:
    """Indices of the nodes needed to produce `wanted`, stopping at `stop`."""
    producer = {
        output: index
        for index, node in enumerate(model.graph.node)
        for output in node.output
    }
    needed: set[int] = set()
    seen: set[str] = set()
    pending = list(wanted)
    while pending:
        name = pending.pop()
        if name in seen or name in stop:
            continue
        seen.add(name)
        index = producer.get(name)
        if index is None or index in needed:
            continue
        needed.add(index)
        pending.extend(referenced_tensors(model.graph.node[index]))
    return needed


def stage_graph(model: onnx.ModelProto, keep: list[onnx.NodeProto],
                inputs: list[onnx.ValueInfoProto],
                outputs: list[onnx.ValueInfoProto],
                name: str) -> onnx.ModelProto:
    """Builds one half of a split model out of the nodes it keeps.

    Partitioning by hand rather than with ``onnx.utils.extract_model`` keeps
    control-flow nodes intact: some voices were exported with an ``If`` whose
    branches reference outer tensors, and extraction drops those and produces a
    graph that will not load.
    """
    wanted = set()
    for node in keep:
        wanted |= referenced_tensors(node)
    graph = onnx.helper.make_graph(
        keep, name, inputs, outputs,
        [tensor for tensor in model.graph.initializer if tensor.name in wanted],
    )
    graph.value_info.extend(
        value for value in model.graph.value_info if value.name in wanted
    )
    stage = onnx.helper.make_model(
        graph, opset_imports=list(model.opset_import),
        producer_name="split-piper-stages",
    )
    stage.ir_version = model.ir_version
    return stage


def feed_for(model_path: Path, ids: np.ndarray, multi_speaker: bool) -> dict:
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array(VERIFY_SCALES, dtype=np.float32),
    }
    if multi_speaker:
        feed["sid"] = np.array([0], dtype=np.int64)
    return feed


def measured_seam_shapes(model: onnx.ModelProto, upstream: Path,
                         model_path: Path,
                         crossing: list[str]) -> dict[str, list]:
    """The shape of each crossing tensor, found by running the body.

    Shape inference gives up part way through some exports and leaves the seam
    with a type but no shape, which ONNX Runtime will not serialize as a graph
    boundary. Running the stage says what the shapes really are. Two runs of
    different lengths, so the axis that follows the utterance can be told from
    the ones that do not and left free.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    body = ort.InferenceSession(str(upstream), options,
                                providers=["CPUExecutionProvider"])
    multi_speaker = any(value.name == "sid" for value in model.graph.input)
    ids = phoneme_ids(model_path)
    short = ids[:, :max(3, ids.shape[1] // 2)]

    long_run = body.run(crossing, feed_for(model_path, ids, multi_speaker))
    short_run = body.run(crossing, feed_for(model_path, short, multi_speaker))
    shapes = {}
    for name, wide, narrow in zip(crossing, long_run, short_run):
        shapes[name] = [
            "frames" if a != b else a
            for a, b in zip(wide.shape, narrow.shape)
        ]
    return shapes


def split_voice(model_path: Path, out_dir: Path) -> tuple[Path, Path, list[str]]:
    model = onnx.load(str(model_path), load_external_data=False)
    crossing = find_boundary(model)

    # Types for the tensors the stages join on. They are activations, so they
    # only appear in the graph once shapes have been inferred.
    inferred = onnx.shape_inference.infer_shapes(model)
    known = {value.name: value for value in inferred.graph.value_info}
    missing = [name for name in crossing if name not in known]
    if missing:
        raise SystemExit(f"No inferred type for {missing}; cannot join stages.")
    seam = [known[name] for name in crossing]
    unshaped = [value.name for value in seam
                if not value.type.tensor_type.HasField("shape")]

    # Take each stage as everything its own outputs need, stopping at its
    # inputs. Anything folded from constants rather than from the text, such as
    # a dequantized weight or a shape, is reached by whichever stage uses it and
    # so lands inside that stage instead of crossing the seam.
    graph_inputs = {value.name for value in model.graph.input}
    initializers = {tensor.name for tensor in model.graph.initializer}
    outputs = [value.name for value in model.graph.output]
    generator_set = backward_closure(model, outputs,
                                     set(crossing) | initializers)
    upstream_set = backward_closure(model, crossing,
                                    graph_inputs | initializers)
    overlap = generator_set & upstream_set
    if any(model.graph.node[index].op_type not in ("Constant", "Identity")
           for index in overlap):
        raise SystemExit(
            "The stages want to share work beyond constants, so the cut is in "
            "the wrong place."
        )

    upstream_model = stage_graph(
        inferred, [node for index, node in enumerate(model.graph.node)
                   if index in upstream_set],
        list(model.graph.input), seam, "upstream")
    generator_model = stage_graph(
        inferred, [node for index, node in enumerate(model.graph.node)
                   if index in generator_set],
        seam, list(model.graph.output), "generator")

    out_dir.mkdir(parents=True, exist_ok=True)
    upstream = out_dir / f"{model_path.stem}.upstream.onnx"
    generator = out_dir / f"{model_path.stem}.generator.onnx"
    onnx.save(upstream_model, str(upstream))
    if unshaped:
        measured = measured_seam_shapes(model, upstream, model_path, crossing)
        for value in list(upstream_model.graph.output) + \
                list(generator_model.graph.input):
            if value.name not in measured:
                continue
            shape = value.type.tensor_type.shape
            del shape.dim[:]
            for extent in measured[value.name]:
                dimension = shape.dim.add()
                if isinstance(extent, str):
                    dimension.dim_param = extent
                else:
                    dimension.dim_value = int(extent)
        onnx.save(upstream_model, str(upstream))
    onnx.save(generator_model, str(generator))
    return upstream, generator, crossing


def phoneme_ids(model_path: Path) -> np.ndarray:
    """A phoneme sequence the voice accepts, for verification only.

    Uses the voice's own table rather than a phonemizer, so this stays a check
    on the graph rather than on G2P, and runs for any language.
    """
    import json

    config = json.loads(model_path.with_suffix(".onnx.json").read_text())
    table = config["phoneme_id_map"]
    # Whatever this voice's alphabet is, its own table names it. Take a spread
    # of it so the latent is long enough to slice.
    letters = [value for key, value in sorted(table.items())
               if key not in ("^", "$", "_", " ")]
    ids = list(table.get("^", [1])[:1])
    for value in letters[:120]:
        ids.extend(value)
        ids.extend(table.get("_", [0]))
    ids.extend(table.get("$", [2])[:1])
    return np.array([ids], dtype=np.int64)


def verify(model_path: Path, upstream: Path, generator: Path,
           crossing: list[str]) -> tuple[float, float, int]:
    """Renders the voice whole and in stages, and reports how far apart they are.

    Returns the whole-graph difference, the error left at a padded seam, and
    the number of latent frames used.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.log_severity_level = 3
    def load(path):
        return ort.InferenceSession(str(path), options,
                                    providers=["CPUExecutionProvider"])

    whole, body, gen = load(model_path), load(upstream), load(generator)
    ids = phoneme_ids(model_path)
    feed = {
        "input": ids,
        "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
        "scales": np.array(VERIFY_SCALES, dtype=np.float32),
    }
    if any(value.name == "sid" for value in whole.get_inputs()):
        feed["sid"] = np.array([0], dtype=np.int64)

    reference = whole.run(["output"], feed)[0].reshape(-1)
    crossed = dict(zip(crossing, body.run(crossing, feed)))
    joined = gen.run(["output"], dict(crossed))[0].reshape(-1)
    length = min(joined.size, reference.size)
    whole_error = float(np.max(np.abs(joined[:length] - reference[:length])))

    latent_name = max(crossed, key=lambda name: crossed[name].shape[-1])
    latent = crossed[latent_name]
    conditioning = {n: v for n, v in crossed.items() if n != latent_name}
    frames = latent.shape[-1]
    per_frame = reference.size // max(frames, 1)

    # Decode a window out of the middle with padding, the way streaming will,
    # and compare it against the same window of the whole render.
    start = frames // 3
    end = min(frames, start + max(1, frames // 6))
    seam_error = 0.0
    if end > start and per_frame:
        target = reference[start * per_frame:end * per_frame]
        low = max(0, start - PAD_FRAMES)
        high = min(frames, end + PAD_FRAMES)
        sliced = dict(conditioning)
        sliced[latent_name] = latent[:, :, low:high]
        piece = gen.run(["output"], sliced)[0].reshape(-1)
        offset = (start - low) * per_frame
        window = piece[offset:offset + target.size]
        count = min(window.size, target.size)
        scale = float(np.sqrt(np.mean(target[:count] ** 2))) or 1e-9
        seam_error = float(
            np.sqrt(np.mean((window[:count] - target[:count]) ** 2))) / scale
    return whole_error, seam_error, frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", type=Path, nargs="+",
                        help="Quantized Piper voice .onnx files")
    parser.add_argument("--out-dir", type=Path,
                        help="Defaults to beside each model")
    parser.add_argument("--verify", action="store_true",
                        help="Render each voice whole and in stages and compare")
    args = parser.parse_args()

    failures = 0
    for model_path in args.models:
        if not model_path.is_file():
            print(f"{model_path}: missing", file=sys.stderr)
            failures += 1
            continue
        out_dir = args.out_dir or model_path.parent
        try:
            upstream, generator, crossing = split_voice(model_path, out_dir)
        except (SystemExit, Exception) as error:  # noqa: BLE001
            print(f"{model_path.stem}: FAILED to split: {error}",
                  file=sys.stderr)
            failures += 1
            continue
        whole_mb = model_path.stat().st_size / 1e6
        parts_mb = (upstream.stat().st_size + generator.stat().st_size) / 1e6
        report = (f"{model_path.stem}: {len(crossing)} crossing, "
                  f"{parts_mb:.1f} MB in two stages vs {whole_mb:.1f} MB whole")
        if args.verify:
            try:
                whole_error, seam_error, frames = verify(
                    model_path, upstream, generator, crossing)
            except Exception as error:  # noqa: BLE001
                print(f"{report}\n{model_path.stem}: FAILED to verify: {error}",
                      file=sys.stderr)
                failures += 1
                continue
            report += (f", {frames} frames, whole {whole_error:.1e}, "
                       f"seam {seam_error:.1e}")
            # The stages are a partition of the same graph, so anything above
            # float rounding means the cut landed in the wrong place.
            if whole_error > 1e-4 or seam_error > 1e-2:
                report += "  FAILED"
                failures += 1
        print(report)

    if failures:
        print(f"\n{failures} voice(s) failed", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
