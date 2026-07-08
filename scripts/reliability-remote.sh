#!/usr/bin/env bash
#
# Runs the heavy reliability checks on a Linux x86 host with clang + libFuzzer.
# Normally invoked by scripts/reliability.sh over SSH, but can be run directly
# on the box from the repo root.
#
# Steps (each contributes to the final pass/fail):
#   1. Configure + build core with -DMOONSHINE_RELIABILITY=ON (ASan/UBSan) using
#      clang, exporting compile_commands.json.
#   2. Run the core test suite under the sanitizers.
#   3. Run clang-tidy on first-party sources (advisory unless TIDY_STRICT=1).
#   4. Fuzz each per-module target for FUZZ_SECONDS, saving any crash reproducers.
#
# Environment:
#   FUZZ_SECONDS   seconds per fuzz target (default 900)
#   TIDY_STRICT    1 => clang-tidy findings fail the run (default 0, advisory)
#   CC / CXX       compilers (default clang / clang++, required for libFuzzer)
#   JOBS           parallel build jobs (default: nproc)
set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
CORE_DIR="${REPO_ROOT_DIR}/core"
BUILD_DIR="${CORE_DIR}/build"
FUZZ_BUILD_DIR="${CORE_DIR}/reliability/build"
ARTIFACTS_DIR="${CORE_DIR}/reliability/artifacts"
CORPUS_ROOT="${CORE_DIR}/reliability/corpus"
ORT_LIB_DIR="${CORE_DIR}/third-party/onnxruntime/lib/linux/x86_64"

FUZZ_SECONDS="${FUZZ_SECONDS:-900}"
TIDY_STRICT="${TIDY_STRICT:-0}"
CC="${CC:-clang}"
CXX="${CXX:-clang++}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

overall_status=0
declare -a FAILURES=()
record_failure() {
  FAILURES+=("$1")
  overall_status=1
  echo "FAIL: $1" >&2
}

echo "=============================================================="
echo " Moonshine reliability run"
echo "   host        : $(hostname)"
echo "   repo        : ${REPO_ROOT_DIR}"
echo "   compiler    : ${CXX}"
echo "   fuzz budget : ${FUZZ_SECONDS}s per target"
echo "=============================================================="

# ---------------------------------------------------------------------------
# Preflight: clang with libFuzzer, cmake.
# ---------------------------------------------------------------------------
if ! command -v "${CXX}" >/dev/null 2>&1; then
  echo "error: '${CXX}' not found; clang is required for libFuzzer." >&2
  exit 1
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "error: cmake not found." >&2
  exit 1
fi
echo "${CXX} version: $(${CXX} --version | head -n1)"

mkdir -p "${ARTIFACTS_DIR}"

# ---------------------------------------------------------------------------
# 1. Configure + build with sanitizers.
# ---------------------------------------------------------------------------
echo ""
echo "--- Configuring sanitizer build ---"
rm -rf "${BUILD_DIR}" "${FUZZ_BUILD_DIR}"
if ! cmake -S "${CORE_DIR}" -B "${BUILD_DIR}" \
      -DMOONSHINE_RELIABILITY=ON \
      -DCMAKE_C_COMPILER="${CC}" \
      -DCMAKE_CXX_COMPILER="${CXX}" \
      2>&1 | tee "${ARTIFACTS_DIR}/cmake-configure.log"; then
  echo "error: cmake configure failed." >&2
  exit 1
fi

echo ""
echo "--- Building (sanitizers) ---"
if ! cmake --build "${BUILD_DIR}" -j "${JOBS}" \
      2>&1 | tee "${ARTIFACTS_DIR}/build.log"; then
  echo "error: build failed." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2. Sanitized test suite.
# ---------------------------------------------------------------------------
# Halt on the first real error. Leak detection is disabled for the integration
# binaries because they link the prebuilt onnxruntime, which has its own
# process-lifetime allocations; per-module leak checking happens in the fuzzers.
export ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=0:print_stacktrace=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
export LD_LIBRARY_PATH="${ORT_LIB_DIR}:${LD_LIBRARY_PATH:-}"

echo ""
echo "--- Running sanitized test suite ---"
run_test() {
  local name="$1"
  local bin="$2"
  shift 2
  if [[ ! -x "${bin}" ]]; then
    echo "  skip ${name} (missing ${bin#"${REPO_ROOT_DIR}"/})"
    return
  fi
  echo "  run  ${name}"
  if ! ( cd "${REPO_ROOT_DIR}/test-assets" 2>/dev/null && "${bin}" "$@" ) \
        >"${ARTIFACTS_DIR}/test-${name}.log" 2>&1; then
    record_failure "test:${name} (see artifacts/test-${name}.log)"
  fi
}

run_test bin-tokenizer   "${CORE_DIR}/bin-tokenizer/build/bin-tokenizer-test"
run_test string-utils    "${CORE_DIR}/moonshine-utils/build/string-utils-test"
run_test debug-utils     "${CORE_DIR}/moonshine-utils/build/debug-utils-test"
run_test resampler       "${BUILD_DIR}/resampler-test"
run_test cosine-distance "${BUILD_DIR}/cosine-distance-test"
run_test word-alignment  "${BUILD_DIR}/word-alignment-test"
run_test spelling-fusion "${BUILD_DIR}/spelling-fusion-test"
run_test voice-activity  "${BUILD_DIR}/voice-activity-detector-test"
run_test transcriber     "${BUILD_DIR}/transcriber-test"
run_test moonshine-c-api "${BUILD_DIR}/moonshine-c-api-test"
run_test moonshine-cpp   "${BUILD_DIR}/moonshine-cpp-test"

# ---------------------------------------------------------------------------
# 3. clang-tidy (advisory unless TIDY_STRICT=1).
# ---------------------------------------------------------------------------
echo ""
echo "--- clang-tidy ---"
if command -v clang-tidy >/dev/null 2>&1 && \
   command -v run-clang-tidy >/dev/null 2>&1; then
  # Limit to first-party sources; the .clang-tidy HeaderFilterRegex keeps
  # third-party headers quiet.
  TIDY_REGEX='core/(moonshine-utils|ort-utils|bin-tokenizer)/|core/[^/]*\.(cpp|cc)$|core/moonshine-tts/src/'
  if run-clang-tidy -p "${BUILD_DIR}" -quiet "${TIDY_REGEX}" \
        >"${ARTIFACTS_DIR}/clang-tidy.log" 2>&1; then
    echo "  clang-tidy: clean"
  else
    if [[ "${TIDY_STRICT}" == "1" ]]; then
      record_failure "clang-tidy (see artifacts/clang-tidy.log)"
    else
      echo "  clang-tidy: findings recorded (advisory) in artifacts/clang-tidy.log"
    fi
  fi
else
  echo "  skip: clang-tidy / run-clang-tidy not installed"
fi

# ---------------------------------------------------------------------------
# 4. Fuzzing (time-boxed, per module).
# ---------------------------------------------------------------------------
# Leak detection is on here: the fuzz targets link only the module under test.
export ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=1:print_stacktrace=1"

echo ""
echo "--- Fuzzing (${FUZZ_SECONDS}s per target) ---"
run_fuzzer() {
  local name="$1"
  local bin="${FUZZ_BUILD_DIR}/${name}"
  shift
  if [[ ! -x "${bin}" ]]; then
    echo "  skip ${name} (missing binary)"
    return
  fi
  local corpus="${CORPUS_ROOT}/${name}"
  mkdir -p "${corpus}"
  # Seed opportunistically (extra args are seed files/dirs to copy in).
  for seed in "$@"; do
    [[ -e "${seed}" ]] && cp "${seed}" "${corpus}/" 2>/dev/null || true
  done
  echo "  fuzz ${name}"
  if ! "${bin}" "${corpus}" \
        -max_total_time="${FUZZ_SECONDS}" \
        -artifact_prefix="${ARTIFACTS_DIR}/${name}-" \
        -print_final_stats=1 \
        >"${ARTIFACTS_DIR}/fuzz-${name}.log" 2>&1; then
    record_failure "fuzz:${name} crashed (see artifacts/fuzz-${name}.log and ${name}-* reproducer)"
  fi
}

# Seed corpora from existing small assets where relevant.
TOKENIZER_SEEDS=()
while IFS= read -r f; do TOKENIZER_SEEDS+=("$f"); done < <(
  find "${REPO_ROOT_DIR}/test-assets" -name 'tokenizer.bin' 2>/dev/null | head -n 4
)
WAV_SEEDS=()
while IFS= read -r f; do WAV_SEEDS+=("$f"); done < <(
  find "${REPO_ROOT_DIR}/test-assets" -name '*.wav' 2>/dev/null | head -n 8
)

run_fuzzer fuzz_bin_tokenizer "${TOKENIZER_SEEDS[@]:-}"
run_fuzzer fuzz_wav_pcm "${WAV_SEEDS[@]:-}"
run_fuzzer fuzz_resampler
run_fuzzer fuzz_string_utils

# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------
echo ""
echo "=============================================================="
if [[ "${overall_status}" -eq 0 ]]; then
  echo " Reliability run PASSED"
else
  echo " Reliability run FAILED (${#FAILURES[@]} issue(s)):"
  for f in "${FAILURES[@]}"; do
    echo "   - ${f}"
  done
fi
echo " Logs and reproducers: ${ARTIFACTS_DIR#"${REPO_ROOT_DIR}"/}"
echo "=============================================================="
exit "${overall_status}"
