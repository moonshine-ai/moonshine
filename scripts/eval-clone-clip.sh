#!/usr/bin/env bash
# Structural before/after eval for clone-clip word-boundary refinement.
# Target: under two minutes on a Mac with test-assets/tiny-en present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT}/core/build"
MODEL_PATH="${1:-tiny-en}"

mkdir -p "${BUILD_DIR}"
cmake -S "${ROOT}/core" -B "${BUILD_DIR}" >/dev/null
cmake --build "${BUILD_DIR}" --target clone-clip-eval clone-clip-test -j "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo "== clone-clip-test =="
"${BUILD_DIR}/clone-clip-test"

echo "== clone-clip-eval =="
cd "${ROOT}/test-assets"
exec "${BUILD_DIR}/clone-clip-eval" "${MODEL_PATH}"
