#!/usr/bin/env python3
"""Fail if an ORT model stores dequantized float weights that should be int8.

ORT conversion at full optimization constant-folds Shrink Ray's
``Cast -> Mul -> Add`` dequant chains back into float32 initializers, ~4x the
file, with no runtime benefit beyond skipping a dequant the split-weights
path already runs once at load. This check walks shipped ``.ort`` trees and
catches that fold, plus the same failure on TTS ``*.weights.ort`` files.

Usage:
    python scripts/check-ort-weight-storage.py
    python scripts/check-ort-weight-storage.py path/to/model/dir
"""

from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Files that are supposed to keep integer weight storage. A leftover
# ``frontend.ort`` next to a split pair is ignored (the pair is what loads).
SPLIT_GRAPH_SUFFIX = ".model.ort"
SPLIT_WEIGHTS_SUFFIX = ".weights.ort"

# Initializer bytes below this are biases / scalars, not the fold we care about.
LARGE_TENSOR_BYTES = 64 * 1024

# Graph halves (*.model.ort) keep some float (embeddings, LayerNorm). A folded
# frontend dumps ~8 MB of weights back into the graph; Piper leftovers are <2 MB.
MAX_GRAPH_HALF_FLOAT_BYTES = 4 * 1024 * 1024

# Integer share of large-tensor bytes below which a quantized graph has folded.
MIN_INTEGER_SHARE = 0.70


def _dtype_names():
    from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs import (
        TensorDataType,
    )

    return {
        v: k
        for k, v in TensorDataType.TensorDataType.__dict__.items()
        if not k.startswith("_")
    }


def initializer_bytes_by_dtype(path: Path) -> dict[str, int]:
    """Return raw initializer payload bytes grouped by ORT tensor dtype name."""
    from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.InferenceSession import (
        InferenceSession,
    )

    names = _dtype_names()
    buf = bytearray(path.read_bytes())
    graph = InferenceSession.GetRootAsInferenceSession(buf, 0).Model().Graph()
    totals: dict[str, int] = collections.Counter()
    for i in range(graph.InitializersLength()):
        tensor = graph.Initializers(i)
        dtype = names.get(tensor.DataType(), str(tensor.DataType()))
        totals[dtype] += tensor.RawDataLength()
    return dict(totals)


def _integer_bytes(by_dtype: dict[str, int]) -> int:
    return by_dtype.get("INT8", 0) + by_dtype.get("UINT8", 0)


def _float_bytes(by_dtype: dict[str, int]) -> int:
    return by_dtype.get("FLOAT", 0) + by_dtype.get("FLOAT16", 0)


def _large_weight_bytes(by_dtype: dict[str, int]) -> tuple[int, int]:
    """Integer vs float bytes, treating only large payloads as weights."""
    # Per-dtype totals already drop tiny tensors if the file is mostly weights;
    # a FLOAT bias of a few KB cannot hide an 8 MB fold. Use the raw totals
    # but ignore files whose combined large dtypes are under the threshold.
    integer = _integer_bytes(by_dtype)
    floating = _float_bytes(by_dtype)
    return integer, floating


def _has_split_frontend(directory: Path) -> bool:
    return (directory / "frontend.model.ort").is_file() and (
        directory / "frontend.weights.ort"
    ).is_file()


def check_ort_file(path: Path) -> list[str]:
    """Return human-readable failures for one ``.ort`` file, or empty."""
    name = path.name
    parent = path.parent
    errors: list[str] = []

    # Skip a leftover single-file frontend when the split pair is present.
    if name == "frontend.ort" and _has_split_frontend(parent):
        return []

    try:
        by_dtype = initializer_bytes_by_dtype(path)
    except Exception as exc:  # noqa: BLE001 - report the file, keep walking
        return [f"{path}: failed to parse ORT initializers ({exc})"]

    integer, floating = _large_weight_bytes(by_dtype)
    weight_bytes = integer + floating

    if name == "frontend.ort":
        if floating >= LARGE_TENSOR_BYTES:
            errors.append(
                f"{path}: frontend.ort stores {floating:,} bytes of float "
                f"weights; convert with scripts/quantize-streaming-model.sh "
                f"so the file splits into frontend.model.ort + "
                f"frontend.weights.ort (int8 on disk, dequant once at load)"
            )
        return errors

    if name.endswith(SPLIT_WEIGHTS_SUFFIX):
        if weight_bytes < LARGE_TENSOR_BYTES:
            return errors
        share = integer / weight_bytes if weight_bytes else 1.0
        if share < MIN_INTEGER_SHARE:
            errors.append(
                f"{path}: {share:.0%} of large tensors are integer "
                f"(int8/uint8 {integer:,} B, float {floating:,} B); "
                f"ORT conversion folded the dequant chain. Split with "
                f"scripts/split-model-weights.py"
            )
        return errors

    if name.endswith(SPLIT_GRAPH_SUFFIX):
        if floating >= MAX_GRAPH_HALF_FLOAT_BYTES:
            errors.append(
                f"{path}: graph half still holds {floating:,} bytes of float "
                f"weights; they belong in the sibling .weights.ort"
            )
        return errors

    # Quantized-activation STT graphs. Non-streaming tiny/base keep mixed
    # float + uint8 (a different recipe) and are not in this set.
    quantized_stt = name in {
        "encoder.ort",
        "adapter.ort",
        "cross_kv.ort",
        "decoder_kv.ort",
        "decoder_kv_with_attention.ort",
    }
    if quantized_stt and integer >= LARGE_TENSOR_BYTES:
        share = integer / weight_bytes if weight_bytes else 1.0
        if share < MIN_INTEGER_SHARE:
            errors.append(
                f"{path}: {share:.0%} of large tensors are integer "
                f"(int8/uint8 {integer:,} B, float {floating:,} B); "
                f"activation-quantized graphs should keep uint8 weights"
            )
    return errors


def find_ort_files(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file() and root.suffix == ".ort":
            found.append(root)
            continue
        if not root.is_dir():
            continue
        found.extend(sorted(p for p in root.rglob("*.ort") if p.is_file()))
    return found


def default_roots() -> list[Path]:
    roots = [
        REPO_ROOT / "test-assets",
        REPO_ROOT / "core" / "moonshine-tts" / "data",
    ]
    return [r for r in roots if r.exists()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        help="Files or directories to scan (default: test-assets and TTS data)",
    )
    args = parser.parse_args()
    try:
        from onnxruntime.tools.ort_format_model.ort_flatbuffers_py.fbs.InferenceSession import (  # noqa: F401
            InferenceSession,
        )
    except ImportError:
        print(
            "warning: onnxruntime Python package not installed; "
            "skipping .ort weight-storage check "
            "(pip install onnxruntime to enable it)",
            file=sys.stderr,
        )
        return 0
    roots = args.roots or default_roots()
    if not roots:
        print("no model roots found", file=sys.stderr)
        return 1

    files = find_ort_files(roots)
    if not files:
        print(f"no .ort files under {', '.join(str(r) for r in roots)}", file=sys.stderr)
        return 1

    failures: list[str] = []
    for path in files:
        failures.extend(check_ort_file(path))

    print(f"checked {len(files)} .ort files under {len(roots)} root(s)")
    if failures:
        for line in failures:
            print(f"FAIL {line}", file=sys.stderr)
        print(f"{len(failures)} weight-storage failure(s)", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
