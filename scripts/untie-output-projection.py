#!/usr/bin/env python3
"""Give a decoder's output projection its own weight so it quantizes properly.

Moonshine decoders tie the output projection to the token embedding:

    logits = MatMul(hidden, Transpose(embedding.weight))

The quantizer (``onnx_shrink_ray --method integer_activations``) only turns a
``MatMul`` into ``DynamicQuantizeMatMul`` when its weight operand is an
initializer. Here it is the output of a ``Transpose`` node, so the quantizer
falls back to storing the transposed table as int8 behind a
``DequantizeLinear``, and the ``MatMul`` stays float. At inference that means,
on every decoder step, dequantizing the whole 640x32768 table to an 84 MB
float tensor and then running a float GEMM whose weight has to be re-packed
because it is not a constant. Measured on the medium streaming decoder this
is 40-45% of the per-step time on every platform, and it is the single
largest allocation the model makes.

This script replaces the ``Transpose`` with a pre-transposed initializer so the
projection quantizes like every other linear layer. The embedding ``Gather``
keeps using the original table; the quantized file already stored both copies,
so the model does not get bigger.

``quantize-streaming-model.sh`` runs it on the float ``decoder_kv.onnx``
before quantizing. By hand:

    python scripts/untie-output-projection.py decoder_kv.onnx -o decoder_kv_untied.onnx

Without ``-o`` the file is rewritten in place and the original kept as
``*.tied.onnx``. A model without the pattern is written out unchanged.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def untie(model: onnx.ModelProto) -> int:
    graph = model.graph
    initializers = {t.name: t for t in graph.initializer}
    producers = {o: n for n in graph.node for o in n.output}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)

    rewritten = 0
    for node in list(graph.node):
        if node.op_type != "MatMul" or len(node.input) != 2:
            continue
        transpose = producers.get(node.input[1])
        if transpose is None or transpose.op_type != "Transpose":
            continue
        source = initializers.get(transpose.input[0])
        if source is None:
            continue
        perm = None
        for attr in transpose.attribute:
            if attr.name == "perm":
                perm = list(attr.ints)
        weight = numpy_helper.to_array(source)
        if weight.ndim != 2 or (perm is not None and perm != [1, 0]):
            continue

        new_name = f"{source.name}_T"
        if new_name in initializers:
            raise RuntimeError(f"initializer {new_name} already exists")
        transposed = numpy_helper.from_array(
            np.ascontiguousarray(weight.T), name=new_name
        )
        graph.initializer.append(transposed)
        initializers[new_name] = transposed
        node.input[1] = new_name
        print(
            f"  {node.name}: weight {source.name}{list(weight.shape)} -> "
            f"{new_name}{list(weight.T.shape)}"
        )
        rewritten += 1

        # Drop the Transpose if nothing else reads it.
        other_readers = [
            n for n in consumers.get(transpose.output[0], []) if n is not node
        ]
        if not other_readers:
            graph.node.remove(transpose)
    return rewritten


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model", type=Path)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Where to write the result (default: in place, keeping a "
        "*.tied.onnx copy of the original)",
    )
    args = parser.parse_args()

    print(args.model)
    model = onnx.load(str(args.model))
    count = untie(model)
    if count == 0:
        print("  no tied output projection found")
    if args.output is not None:
        onnx.save(model, str(args.output))
        print(f"  wrote {args.output}")
        return 0
    if count == 0:
        return 0
    backup = args.model.with_suffix(".tied.onnx")
    if not backup.exists():
        shutil.copy2(args.model, backup)
    onnx.save(model, str(args.model))
    print(f"  rewrote {count} projection(s); original kept at {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
