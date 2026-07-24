#!/usr/bin/env python3
"""Convert the intent-recognition embedding model to all-in-one ``.ort`` files.

The EmbeddingGemma models are published as an ONNX graph plus a large external
weights sidecar (e.g. ``model_q4.onnx`` + ``model_q4.onnx_data``). External data
is awkward for in-memory loading — the caller has to keep the sidecar around and
ONNX Runtime expects to resolve it by filename. This script converts each
``.onnx`` into a single self-contained ONNX Runtime ``.ort`` file (weights
embedded inline in the flatbuffer), matching how the speech-to-text models ship.
The resulting files load from a single buffer with
``moonshine_create_intent_recognizer_from_memory`` (no sidecar).

Usage:

  # Convert every model_*.onnx in a directory in place:
  python3 scripts/export-embedding-model-ort.py path/to/embeddinggemma-300m

  # Or download the published variant(s) from the CDN, convert, and write to out/:
  python3 scripts/export-embedding-model-ort.py --download --variant q4 --output-dir out

The `.ort` files should then be uploaded next to the existing model files on the
CDN (they are additive; the `.onnx`/`.onnx_data` files can stay for older
clients). ``core/moonshine-model-catalog.cpp`` and the Python
``download.py`` catalog list the ``.ort`` files as the canonical download set.
"""

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

CDN_BASE = "https://download.moonshine.ai/model/embeddinggemma-300m"

# Variant -> ONNX filename stem (mirrors GemmaEmbeddingModel::load and the model
# catalog). Note "q8" maps to model_quantized, not model_q8.
VARIANT_TO_STEM = {
    "fp32": "model",
    "fp16": "model_fp16",
    "q8": "model_quantized",
    "q4": "model_q4",
    "q4f16": "model_q4f16",
}


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"[export-ort] already have {dest.name}")
        return
    print(f"[export-ort] downloading {url}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
        shutil.copyfileobj(response, out)


def convert_one(onnx_path: Path, output_dir: Path) -> Path:
    """Convert a single ``.onnx`` (with optional external-data sidecar beside it)
    into a self-contained ``.ort`` in ``output_dir``. Returns the ``.ort`` path."""
    from onnxruntime.tools.convert_onnx_models_to_ort import (
        OptimizationStyle,
        convert_onnx_models_to_ort,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    # Fixed optimization bakes the graph optimizations into the .ort so no
    # runtime optimization config is required. convert_onnx_models_to_ort loads
    # the full model (resolving the external-data sidecar from disk) and writes
    # the initializers inline into the .ort flatbuffer, so the result is a single
    # self-contained file.
    convert_onnx_models_to_ort(
        onnx_path,
        output_dir=output_dir,
        optimization_styles=[OptimizationStyle.Fixed],
        # Keep every initializer inline in the flatbuffer (no external sidecar),
        # which is the whole point of this conversion.
        save_optimized_onnx_model=False,
        allow_conversion_failures=False,
    )
    # The tool emits "<stem>.ort" (Fixed style has no suffix decoration in recent
    # ORT; older versions used ".with_runtime_opt.ort" only for Runtime style).
    stem = onnx_path.stem
    candidates = sorted(output_dir.glob(f"{stem}*.ort"))
    if not candidates:
        raise SystemExit(f"[export-ort] conversion produced no .ort for {onnx_path}")
    produced = candidates[0]
    final = output_dir / f"{stem}.ort"
    if produced != final:
        produced.replace(final)
    return final


def verify_self_contained(ort_path: Path) -> None:
    """Load the .ort purely from bytes (no cwd sidecar) to prove it is
    self-contained, then run a tiny inference to confirm the weights are inline."""
    import numpy as np
    import onnxruntime as ort

    data = ort_path.read_bytes()
    sess = ort.InferenceSession(data, providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in sess.get_inputs():
        # input_ids / attention_mask are [batch, seq]; use a 1x4 dummy.
        shape = [d if isinstance(d, int) and d > 0 else 4 for d in inp.shape]
        if len(shape) == 0:
            shape = [1, 4]
        feeds[inp.name] = np.ones(shape, dtype=np.int64)
    sess.run(None, feeds)
    print(f"[export-ort] verified {ort_path.name} loads from memory "
          f"({len(data) / 1e6:.1f} MB, no sidecar)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=Path,
        help="Directory containing model_*.onnx (+ .onnx_data) to convert in place.",
    )
    parser.add_argument("--download", action="store_true",
                        help="Download the model(s) from the CDN first.")
    parser.add_argument("--variant", action="append", choices=sorted(VARIANT_TO_STEM),
                        help="Variant(s) to convert (default: all present / q4 when downloading).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Where to write .ort files (default: alongside the .onnx).")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip the load-from-memory verification step.")
    args = parser.parse_args()

    work_dir = args.model_dir or Path("embeddinggemma-300m")
    output_dir = args.output_dir or work_dir

    variants = args.variant
    if args.download:
        variants = variants or ["q4"]
        download(f"{CDN_BASE}/tokenizer.bin", work_dir / "tokenizer.bin")
        for variant in variants:
            stem = VARIANT_TO_STEM[variant]
            download(f"{CDN_BASE}/{stem}.onnx", work_dir / f"{stem}.onnx")
            download(f"{CDN_BASE}/{stem}.onnx_data", work_dir / f"{stem}.onnx_data")

    if not work_dir.is_dir():
        raise SystemExit(f"[export-ort] not a directory: {work_dir}")

    if variants:
        onnx_paths = [work_dir / f"{VARIANT_TO_STEM[v]}.onnx" for v in variants]
    else:
        onnx_paths = sorted(work_dir.glob("model*.onnx"))
    onnx_paths = [p for p in onnx_paths if p.exists()]
    if not onnx_paths:
        raise SystemExit(f"[export-ort] no model*.onnx found in {work_dir}")

    for onnx_path in onnx_paths:
        ort_path = convert_one(onnx_path, output_dir)
        if not args.no_verify:
            verify_self_contained(ort_path)
        print(f"[export-ort] wrote {ort_path}")

    # tokenizer.bin travels with the model unchanged.
    tok = work_dir / "tokenizer.bin"
    if tok.exists() and output_dir != work_dir:
        shutil.copy2(tok, output_dir / "tokenizer.bin")
    print("[export-ort] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
