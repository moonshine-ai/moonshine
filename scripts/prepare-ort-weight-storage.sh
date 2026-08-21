#!/bin/bash -e

# Split any folded streaming frontend.ort files under test-assets, then fail
# if an .ort we ship stores dequantized float weights that should be int8.
# Called from scripts/test-core.sh and scripts/test-core.bat.

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"

# The published streaming frontend.ort is the pre-split pin (int8 folded back
# to float32 at ORT conversion). Split it locally so tests load the pair and
# the weight-storage check can see int8 on disk. Idempotent if the pair exists.
if python3 -c "import onnx, onnxruntime, onnx_shrink_ray" >/dev/null 2>&1; then
  shopt -s nullglob
  for frontend in "${REPO_ROOT_DIR}"/test-assets/*/frontend.ort; do
    dir="$(dirname "${frontend}")"
    if [[ -f "${dir}/frontend.model.ort" && -f "${dir}/frontend.weights.ort" ]]; then
      continue
    fi
    echo "Splitting folded frontend ${frontend}"
    python3 "${SCRIPTS_DIR}/split-model-weights.py" --per-channel "${frontend}"
  done
  shopt -u nullglob
else
  echo "warning: onnx / onnxruntime / onnx_shrink_ray not installed; skipping frontend split" >&2
fi

echo "Checking .ort weight storage (int8 must not fold back to float32)..."
python3 "${SCRIPTS_DIR}/check-ort-weight-storage.py"
