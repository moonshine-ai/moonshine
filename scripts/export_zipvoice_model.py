#!/usr/bin/env python3
# Copyright 2026 Useful Sensors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Produce the deployable ZipVoice model bundle for Moonshine under core/moonshine-tts/data/zipvoice.

This orchestrates the ZipVoice repo's own tooling (it must be checked out separately, with its
dependencies installed in a virtualenv):

  1. ``scripts/export_onnx.py`` exports ``text_encoder.onnx`` / ``fm_decoder.onnx`` plus the mixed
     int8 variants (per-channel weights, activation-limited layers kept in fp32), the Vocos
     ``vocoder.onnx``, and copies ``tokens.txt`` / ``model.json``.
  2. (optional ``--swoosh``) ``scripts/rewrite_swoosh.py`` rewrites the fp32 fm_decoder to reference
     the ``ai.zipvoice`` custom ops that are compiled into libmoonshine.
  3. (optional) ``scripts/check_onnx_parity.py`` validates numerical parity vs PyTorch.
  4. Converts the chosen ONNX graphs to ORT format
     (``python -m onnxruntime.tools.convert_onnx_models_to_ort``). The swoosh graph is converted with
     ``--custom_op_library`` pointing at the prebuilt ZipVoice custom-op library.

By default it ships the *smallest* set that passes parity: mixed int8 text encoder + fm decoder + the
fp32 vocoder, converted to ``.ort``. Pass ``--swoosh`` to instead ship the fp32 custom-op decoder
(faster on some CPUs, larger). Either way the runtime works, since the ``ai.zipvoice`` ops are always
compiled into libmoonshine.

Usage:

  python3 scripts/export_zipvoice_model.py \
      --zipvoice-repo ~/projects/ZipVoice \
      --model-name zipvoice_distill
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run(cmd, cwd=None):
    print("+ " + " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], cwd=cwd, check=True)


def convert_to_ort(python: str, onnx_path: Path, custom_op_lib: str = None):
    cmd = [
        python,
        "-m",
        "onnxruntime.tools.convert_onnx_models_to_ort",
        str(onnx_path),
        "--optimization_style",
        "Fixed",
    ]
    if custom_op_lib is not None:
        cmd += ["--custom_op_library", custom_op_lib]
    run(cmd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zipvoice-repo", required=True, help="Path to the ZipVoice checkout")
    parser.add_argument(
        "--python",
        default=None,
        help="Python interpreter with ZipVoice deps (default: <repo>/.venv/bin/python if present).",
    )
    parser.add_argument("--model-name", default="zipvoice_distill",
                        choices=["zipvoice", "zipvoice_distill"])
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "core/moonshine-tts/data/zipvoice"))
    parser.add_argument("--int8", action="store_true", default=True,
                        help="Ship the mixed int8 acoustic models (default, smallest).")
    parser.add_argument("--swoosh", action="store_true", default=False,
                        help="Ship the fp32 ai.zipvoice custom-op fm_decoder instead of int8.")
    parser.add_argument("--custom-op-lib", default=None,
                        help="Prebuilt libzipvoice_swoosh.{dylib,so,dll} for .ort conversion of the "
                        "swoosh graph (required with --swoosh).")
    parser.add_argument("--keep-onnx", action="store_true", default=False,
                        help="Also copy the .onnx graphs (default keeps only the .ort files).")
    args = parser.parse_args()

    repo = Path(args.zipvoice_repo).expanduser()
    if not repo.is_dir():
        raise SystemExit(f"ZipVoice repo not found: {repo}")
    python = args.python or (
        str(repo / ".venv/bin/python") if (repo / ".venv/bin/python").exists() else sys.executable
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        onnx_dir = Path(tmp) / "onnx"
        run([
            python, "scripts/export_onnx.py",
            "--model-name", args.model_name,
            "--onnx-model-dir", str(onnx_dir),
            "--quantize-int8", "true",
            "--quantize-per-channel", "true",
            "--quantize-exclude-act-limited", "true",
            "--export-vocoder", "true",
        ], cwd=repo)

        if args.swoosh:
            if args.custom_op_lib is None:
                raise SystemExit("--swoosh requires --custom-op-lib (build custom_ops/ in the ZipVoice repo)")
            fm_src = onnx_dir / "fm_decoder.onnx"
            fm_swoosh = onnx_dir / "fm_decoder_swoosh.onnx"
            run([python, "scripts/rewrite_swoosh.py", "--input", str(fm_src),
                 "--output", str(fm_swoosh)], cwd=repo)
            fm_final = fm_swoosh
            te_final = onnx_dir / "text_encoder.onnx"
        else:
            fm_final = onnx_dir / "fm_decoder_int8.onnx"
            te_final = onnx_dir / "text_encoder_int8.onnx"
        vocoder = onnx_dir / "vocoder.onnx"

        # Convert each chosen graph to ORT format.
        convert_to_ort(python, te_final)
        convert_to_ort(python, fm_final, custom_op_lib=args.custom_op_lib if args.swoosh else None)
        convert_to_ort(python, vocoder)

        def deploy(src_onnx: Path, canonical_stem: str):
            ort_src = src_onnx.with_suffix(".ort")
            if ort_src.exists():
                shutil.copy(ort_src, out_dir / f"{canonical_stem}.ort")
            if args.keep_onnx:
                shutil.copy(src_onnx, out_dir / f"{canonical_stem}.onnx")

        deploy(te_final, "text_encoder")
        deploy(fm_final, "fm_decoder")
        deploy(vocoder, "vocoder")
        shutil.copy(onnx_dir / "tokens.txt", out_dir / "tokens.txt")
        shutil.copy(onnx_dir / "model.json", out_dir / "model.json")

    print(f"Wrote ZipVoice bundle to {out_dir}")
    for p in sorted(out_dir.iterdir()):
        print(f"  {p.name:<24} {p.stat().st_size / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
