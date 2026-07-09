#!/usr/bin/env python3
"""Regenerate core/silero-vad-model-data.h from the Silero VAD model.

The Silero VAD ships from upstream as an ONNX protobuf. This script converts it
to the ONNX Runtime `.ort` flatbuffer format and bakes the bytes into the C
header that gets embedded in libmoonshine. The `.ort` format is what every other
model in this repo uses: it is smaller, loads faster, works with minimal/mobile
ORT builds, and (because ORT can parse it in place) avoids the protobuf
alignment reads that fault on 32-bit ARM.

The generated array keeps the historical symbol names (`silero_vad_onnx` /
`silero_vad_onnx_len`) so no C++ consumer needs to change; ORT's
CreateSessionFromArray auto-detects the format from the bytes.

By default the source .onnx is downloaded from a commit-pinned upstream URL into
a temporary directory and checked against a known SHA-256, so nothing is left
behind in the tree. Pass --onnx to convert a local model instead.

Note: the .onnx -> .ort conversion is NOT byte-reproducible (ORT embeds
non-deterministic data), so regenerating produces a different byte array even
from the identical source model. The output is functionally identical; the
integrity guarantee is on the source .onnx (SHA-256), not the converted bytes.

Optimization level (`--optimization`) is a space/latency tradeoff, and note
that for .ort models ORT applies whatever optimizations are baked in here -- the
runtime graph_optimization_level does NOT re-optimize a .ort graph:
  * disable  -- raw graph, smallest (~onnx size). Constant folding is skipped,
                so the STFT basis tables are recomputed each inference (slightly
                slower). Best when binary footprint matters (the default here).
  * all      -- fully pre-optimized, ~1MB larger (folded constants baked in),
                fastest inference.
  * basic/extended sit in between but already include the constant folding, so
                they are the same size as `all`.

Usage:
    # Convert with an ORT version matching the bundled runtime (recommended):
    uv run --python 3.13 --with onnxruntime==1.23.2 \
        scripts/generate-silero-vad-data.py

    # Or, if onnxruntime is already importable:
    python3 scripts/generate-silero-vad-data.py

    # Bake in full optimizations (larger, faster inference):
    python3 scripts/generate-silero-vad-data.py --optimization all

    # Convert a local model instead of downloading:
    python3 scripts/generate-silero-vad-data.py --onnx /path/to/silero_vad.onnx
"""

import argparse
import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEADER = REPO_ROOT / "core" / "silero-vad-model-data.h"

# Commit-pinned upstream Silero VAD model (snakers4/silero-vad) so the download
# is immutable, plus its SHA-256 for integrity verification.
SILERO_VAD_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/"
    "b163605b3f44c3aadf28f97b125a2f7c461e9a7f/src/silero_vad/data/silero_vad.onnx"
)
SILERO_VAD_SHA256 = (
    "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
)

GUARD = "SILERO_VAD_MODEL_DATA_H"
SYMBOL = "silero_vad_onnx"
BYTES_PER_LINE = 12
# .ort is a FlatBuffer; the buffer start must be aligned to the largest scalar
# it contains (up to FLATBUFFERS_MAX_ALIGNMENT). Baked char[] arrays are only
# 1-byte aligned, which faults (SIGBUS) on strict-alignment CPUs such as 32-bit
# ARM. 64 dominates every FlatBuffer requirement, covers SIMD tensor reads, and
# lands on a cache line, at a cost of <=63 bytes of padding.
ALIGNMENT = 64

OPT_LEVELS = {
    "disable": "ORT_DISABLE_ALL",
    "basic": "ORT_ENABLE_BASIC",
    "extended": "ORT_ENABLE_EXTENDED",
    "all": "ORT_ENABLE_ALL",
}


def fetch_source_onnx(dest_dir: Path) -> Path:
    """Download the pinned upstream Silero VAD .onnx and verify its SHA-256."""
    dest = dest_dir / "silero_vad.onnx"
    print(f"Downloading {SILERO_VAD_URL}")
    with urllib.request.urlopen(SILERO_VAD_URL) as response:
        data = response.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != SILERO_VAD_SHA256:
        raise RuntimeError(
            "Downloaded Silero VAD model failed integrity check:\n"
            f"  expected {SILERO_VAD_SHA256}\n"
            f"  got      {digest}"
        )
    dest.write_bytes(data)
    return dest


def convert_onnx_to_ort(onnx_path: Path, out_dir: Path, optimization: str) -> Path:
    """Serialize an .onnx model to a .ort flatbuffer at the given opt level.

    Uses ONNX Runtime's optimized_model_filepath save path rather than the
    convert_onnx_models_to_ort CLI, because the CLI does not expose the
    optimization level (it always bakes in full optimization).
    """
    import onnxruntime as ort

    out_path = out_dir / (onnx_path.stem + ".ort")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = getattr(
        ort.GraphOptimizationLevel, OPT_LEVELS[optimization]
    )
    session_options.optimized_model_filepath = str(out_path)
    # Force ORT flatbuffer output regardless of the filename extension.
    session_options.add_session_config_entry("session.save_model_format", "ORT")
    ort.InferenceSession(
        str(onnx_path), session_options, providers=["CPUExecutionProvider"]
    )
    if not out_path.exists():
        raise RuntimeError(f"Conversion produced no .ort file at {out_path}")
    return out_path


def render_header(data: bytes, source_name: str, optimization: str) -> str:
    lines = [
        f"#ifndef {GUARD}",
        f"#define {GUARD}",
        "",
        "// Generated by scripts/generate-silero-vad-data.py -- do not edit by hand.",
        f"// Contents: {source_name} converted to the ONNX Runtime .ort flatbuffer",
        f"// format (optimization level: {optimization}). The symbol name is kept for",
        "// backwards compatibility; the bytes are .ort, which CreateSessionFromArray",
        f"// auto-detects. alignas({ALIGNMENT}) keeps the FlatBuffer buffer start aligned so",
        "// in-place parsing does not fault on strict-alignment CPUs (e.g. 32-bit ARM).",
        "",
        "#include <cstddef>",
        "",
        f"alignas({ALIGNMENT}) unsigned char {SYMBOL}[] = {{",
    ]
    for i in range(0, len(data), BYTES_PER_LINE):
        chunk = data[i : i + BYTES_PER_LINE]
        body = ", ".join(f"0x{b:02x}" for b in chunk)
        terminator = "," if i + BYTES_PER_LINE < len(data) else ""
        lines.append(f"    {body}{terminator}")
    lines[-1] += "};"
    lines.append(f"unsigned int {SYMBOL}_len = {len(data)};")
    lines.append("")
    lines.append(f"#endif  // {GUARD}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="Convert a local Silero VAD .onnx instead of downloading the "
        "pinned upstream model.",
    )
    parser.add_argument(
        "--header", type=Path, default=DEFAULT_HEADER, help="Output C header path."
    )
    parser.add_argument(
        "--optimization",
        choices=sorted(OPT_LEVELS),
        default="disable",
        help="Graph optimization level baked into the .ort (default: disable, "
        "smallest footprint).",
    )
    args = parser.parse_args()

    if args.onnx is not None and not args.onnx.exists():
        parser.error(f"Source model not found: {args.onnx}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        if args.onnx is not None:
            onnx_path = tmp_dir / args.onnx.name
            shutil.copyfile(args.onnx, onnx_path)
        else:
            onnx_path = fetch_source_onnx(tmp_dir)
        ort_path = convert_onnx_to_ort(onnx_path, tmp_dir, args.optimization)
        ort_bytes = ort_path.read_bytes()
        source_name = onnx_path.name

    header_text = render_header(ort_bytes, source_name, args.optimization)
    args.header.write_text(header_text)
    print(
        f"Wrote {args.header} "
        f"({len(ort_bytes):,} bytes .ort, opt={args.optimization}, "
        f"from {source_name})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
