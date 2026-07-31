#!/usr/bin/env python3
"""Split a shrink-ray'd ONNX model into an ORT graph plus an ORT weights blob.

An ORT-format model has its graph optimizations baked in at conversion time and
ORT never re-applies them at load (``inference_session.cc`` only registers graph
transformers when not loading ORT format). For models whose weights are stored
as int8 and dequantized by a ``Cast -> Mul -> Add`` chain, that leaves a bad
choice: fold the chain at conversion and store float32 weights (~4x the size),
or keep the chain and pay for it on every inference (~3x slower).

Splitting the model avoids both. The optimized graph keeps its fusions but
declares the weights as graph *inputs*, so it carries no weight data at all,
and a second model holds the int8 weights plus the dequantize chains and
produces the float32 weights as its outputs. The runtime executes the weights
model once at startup and feeds its outputs to the graph model on every
inference, so the dequantize happens exactly once.

Emits, for ``<voice>.onnx``:
    <voice>.model.ort     fused compute graph, weights as inputs
    <voice>.weights.ort   int8 weights + dequantize chains

Usage:
    python scripts/split-model-weights.py <model.onnx> [...]
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

SHRINK_RAY_SRC = Path.home() / "projects" / "onnx_shrink_ray" / "src"
QUANT_SUFFIX = "_quantized"
DEQUANT_OUTPUT_SUFFIX = "_add_tensor"


def _load_shrink_ray():
    try:
        import onnx_shrink_ray.shrink as shrink  # noqa: PLC0415
        return shrink
    except ImportError:
        pass
    if SHRINK_RAY_SRC.is_dir():
        sys.path.insert(0, str(SHRINK_RAY_SRC))
        import onnx_shrink_ray.shrink as shrink  # noqa: PLC0415
        return shrink
    raise SystemExit(
        "onnx_shrink_ray not found; pip install onnx_shrink_ray or clone it to "
        f"{SHRINK_RAY_SRC}"
    )


def _save_via_ort(src: str, dst: str, optimize: bool, as_ort: bool) -> str:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    opts.optimized_model_filepath = dst
    opts.add_session_config_entry(
        "session.save_model_format", "ORT" if as_ort else "ONNX"
    )
    opts.log_severity_level = 3
    ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])
    return dst


def find_dequantize_chains(graph) -> list[dict]:
    """Locate Cast -> Mul -> Add chains that expand int8 weights to float32.

    Chains are identified by the tensor names shrink-ray generates rather than by
    node object identity, which protobuf does not keep stable across iterations.
    """
    produced = {out: node for node in graph.node for out in node.output}
    inits = {i.name: i for i in graph.initializer}
    chains = []
    for name, init in inits.items():
        if init.data_type != onnx.TensorProto.INT8 or not name.endswith(QUANT_SUFFIX):
            continue
        base = name[: -len(QUANT_SUFFIX)]
        chain_outputs = {
            base + "_cast_tensor",
            base + "_mul_tensor",
            base + DEQUANT_OUTPUT_SUFFIX,
        }
        if not chain_outputs.issubset(produced.keys()):
            continue
        chains.append({
            "output": base + DEQUANT_OUTPUT_SUFFIX,
            "node_outputs": chain_outputs,
            "init_names": {name, base + "_scale", base + "_zero_point"},
            "shape": list(numpy_helper.to_array(init).shape),
        })
    return chains


def _chain_node_outputs(chains) -> set:
    return {out for c in chains for out in c["node_outputs"]}


def _is_chain_node(node, chain_outputs) -> bool:
    return any(out in chain_outputs for out in node.output)


def _build_weights_model(src_onnx: str, dst_onnx: str) -> int:
    model = onnx.load(src_onnx)
    graph = model.graph
    chains = find_dequantize_chains(graph)
    if not chains:
        raise SystemExit(f"{src_onnx}: no int8 dequantize chains found")

    chain_outputs = _chain_node_outputs(chains)
    keep_init_names = {n for c in chains for n in c["init_names"]}

    nodes = [n for n in graph.node if _is_chain_node(n, chain_outputs)]
    inits = [i for i in graph.initializer if i.name in keep_init_names]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    del graph.input[:]
    del graph.output[:]
    del graph.value_info[:]
    for chain in chains:
        graph.output.append(
            helper.make_tensor_value_info(
                chain["output"], onnx.TensorProto.FLOAT, chain["shape"]
            )
        )
    onnx.save(model, dst_onnx)
    return len(chains)


def _build_graph_model(src_onnx: str, dst_onnx: str) -> None:
    model = onnx.load(src_onnx)
    graph = model.graph
    chains = find_dequantize_chains(graph)

    chain_outputs = _chain_node_outputs(chains)
    drop_init_names = {n for c in chains for n in c["init_names"]}

    nodes = [n for n in graph.node if not _is_chain_node(n, chain_outputs)]
    inits = [i for i in graph.initializer if i.name not in drop_init_names]
    del graph.node[:]
    graph.node.extend(nodes)
    del graph.initializer[:]
    graph.initializer.extend(inits)
    for chain in chains:
        graph.input.append(
            helper.make_tensor_value_info(
                chain["output"], onnx.TensorProto.FLOAT, chain["shape"]
            )
        )
    onnx.save(model, dst_onnx)


def split_model(src: Path, force: bool) -> tuple[int, int, int]:
    model_out = src.with_suffix(".model.ort")
    weights_out = src.with_suffix(".weights.ort")
    if model_out.exists() and weights_out.exists() and not force:
        return (0, model_out.stat().st_size, weights_out.stat().st_size)

    shrink = _load_shrink_ray()
    with tempfile.TemporaryDirectory() as tmp:
        # Optimize first so the fusions are baked in, then re-quantize the
        # optimized graph. shrink-ray's per-tensor quantization is idempotent,
        # so re-quantizing already-quantized weights is lossless.
        optimized = _save_via_ort(str(src), f"{tmp}/optimized.onnx", True, False)
        requantized = f"{tmp}/requantized.onnx"
        onnx.save(shrink.quantize_weights(onnx.load(optimized)), requantized)

        weights_onnx = f"{tmp}/weights.onnx"
        graph_onnx = f"{tmp}/graph.onnx"
        count = _build_weights_model(requantized, weights_onnx)
        _build_graph_model(requantized, graph_onnx)

        _save_via_ort(weights_onnx, str(weights_out), False, True)
        _save_via_ort(graph_onnx, str(model_out), False, True)

    return (count, model_out.stat().st_size, weights_out.stat().st_size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for src in args.models:
        if not src.is_file():
            print(f"skip {src}: not a file")
            continue
        count, model_size, weights_size = split_model(src, args.force)
        original = src.stat().st_size
        total = model_size + weights_size
        print(f"{src.name}")
        print(f"  {count} weight tensors moved out of the graph"
              if count else "  already split")
        print(f"  model.ort   {model_size/1e6:7.1f} MB")
        print(f"  weights.ort {weights_size/1e6:7.1f} MB")
        print(f"  total       {total/1e6:7.1f} MB vs {original/1e6:.1f} MB original "
              f"({100.0 * (total - original) / original:+.1f}%)")
    _ = os
    return 0


if __name__ == "__main__":
    sys.exit(main())
