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
#   3. ThreadSanitizer: build the threaded tests into a separate build dir with
#      -DMOONSHINE_SANITIZER=thread and run them (advisory unless TSAN_STRICT=1).
#   4. Run clang-tidy on first-party sources (advisory unless TIDY_STRICT=1).
#   5. Fuzz each per-module target for FUZZ_SECONDS, saving any crash reproducers.
#
# Environment:
#   FUZZ_SECONDS   seconds per fuzz target (default 900)
#   TIDY_STRICT    1 => clang-tidy findings fail the run (default 0, advisory)
#   TSAN           1 => run the ThreadSanitizer stage (default 1); 0 to skip
#   TSAN_STRICT    1 => TSan findings fail the run (default 0, advisory)
#   TSAN_TEST_TIMEOUT  per-TSan-test wall-clock limit in seconds (default 600);
#                      a timeout is always a hard failure so a hang can't stall
#                      the pipeline for hours
#   CC / CXX       compilers (default clang / clang++, required for libFuzzer)
#   JOBS           parallel build jobs (default: nproc)
set -uo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
CORE_DIR="${REPO_ROOT_DIR}/core"
BUILD_DIR="${CORE_DIR}/build"
TSAN_BUILD_DIR="${CORE_DIR}/build-tsan"
FUZZ_BUILD_DIR="${CORE_DIR}/reliability/build"
ARTIFACTS_DIR="${CORE_DIR}/reliability/artifacts"
CORPUS_ROOT="${CORE_DIR}/reliability/corpus"
ORT_LIB_DIR="${CORE_DIR}/third-party/onnxruntime/lib/linux/x86_64"
TSAN_SUPPRESSIONS="${CORE_DIR}/reliability/tsan-suppressions.txt"

FUZZ_SECONDS="${FUZZ_SECONDS:-900}"
TIDY_STRICT="${TIDY_STRICT:-0}"
TSAN="${TSAN:-1}"
TSAN_STRICT="${TSAN_STRICT:-0}"
TSAN_TEST_TIMEOUT="${TSAN_TEST_TIMEOUT:-600}"
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
# Preflight: clang with libFuzzer, cmake. Collect every missing tool first so a
# fresh box gets one actionable install hint rather than failing one at a time.
# ---------------------------------------------------------------------------
declare -a MISSING_TOOLS=()
declare -a MISSING_PKGS=()
require_tool() {
  local tool="$1"
  local pkg="$2"
  local why="$3"
  if ! command -v "${tool}" >/dev/null 2>&1; then
    MISSING_TOOLS+=("${tool} (${why})")
    MISSING_PKGS+=("${pkg}")
  fi
}

require_tool "${CXX}" clang "compiler + libFuzzer"
require_tool cmake cmake "build system"

if [[ "${#MISSING_TOOLS[@]}" -gt 0 ]]; then
  echo "error: required tool(s) not found on $(hostname):" >&2
  for t in "${MISSING_TOOLS[@]}"; do
    echo "  - ${t}" >&2
  done
  # De-duplicate the package list while preserving order.
  declare -a UNIQUE_PKGS=()
  for pkg in "${MISSING_PKGS[@]}"; do
    if [[ ! " ${UNIQUE_PKGS[*]} " == *" ${pkg} "* ]]; then
      UNIQUE_PKGS+=("${pkg}")
    fi
  done
  echo "" >&2
  if command -v apt-get >/dev/null 2>&1; then
    # clang-tidy (provides clang-tidy + run-clang-tidy) is optional/advisory, so
    # it is suggested here but not treated as a hard requirement above. Note it
    # is a SEPARATE package from clang-tools on Debian/Ubuntu.
    echo "Install them on this box (Debian/Ubuntu), then re-run:" >&2
    echo "  sudo apt-get update" >&2
    echo "  sudo apt-get install -y ${UNIQUE_PKGS[*]} clang-tidy" >&2
  else
    echo "Install these packages with your system package manager, then re-run:" >&2
    echo "  ${UNIQUE_PKGS[*]}" >&2
  fi
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
#
# detect_odr_violation=1 downgrades the ODR global check from its default
# (level 2). The test binaries link the core object files *and* libmoonshine.so,
# so single-definition globals like VoiceActivityDetector::silero_vad legitimately
# appear in both modules; level 2 flags that benign same-size duplicate, while
# level 1 still catches genuine ODR mismatches (differing sizes/types).
export ASAN_OPTIONS="abort_on_error=1:halt_on_error=1:detect_leaks=0:print_stacktrace=1:detect_odr_violation=1"
export UBSAN_OPTIONS="print_stacktrace=1:halt_on_error=1"
export LD_LIBRARY_PATH="${ORT_LIB_DIR}:${LD_LIBRARY_PATH:-}"

echo ""
echo "--- Running sanitized test suite ---"
# Tests read their fixtures relative to test-assets/. Resolve the working
# directory once: if the assets are missing (e.g. not synced), fall back to the
# repo root and warn loudly, rather than silently 'cd'-failing and reporting an
# empty-log failure for every test (including the asset-free ones).
TEST_WORKDIR="${REPO_ROOT_DIR}/test-assets"
if [[ ! -d "${TEST_WORKDIR}" ]]; then
  echo "  WARNING: ${TEST_WORKDIR#"${REPO_ROOT_DIR}"/} is missing; tests that load" >&2
  echo "           fixtures will fail. Check the rsync excludes in reliability.sh." >&2
  TEST_WORKDIR="${REPO_ROOT_DIR}"
fi
run_test() {
  local name="$1"
  local bin="$2"
  shift 2
  if [[ ! -x "${bin}" ]]; then
    echo "  skip ${name} (missing ${bin#"${REPO_ROOT_DIR}"/})"
    return
  fi
  echo "  run  ${name}"
  if ! ( cd "${TEST_WORKDIR}" && "${bin}" "$@" ) \
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
# Ubuntu's clang-tidy package installs version-suffixed binaries (clang-tidy-21,
# run-clang-tidy-21) and does not always create the plain symlinks, so resolve
# both, preferring the unversioned names and falling back to recent versions.
echo ""
echo "--- clang-tidy ---"
CLANG_TIDY_BIN="$(command -v clang-tidy 2>/dev/null || true)"
RUN_CLANG_TIDY_BIN="$(command -v run-clang-tidy 2>/dev/null || true)"
for v in 21 20 19 18 17 16 15; do
  [[ -n "${CLANG_TIDY_BIN}" ]] || CLANG_TIDY_BIN="$(command -v "clang-tidy-${v}" 2>/dev/null || true)"
  [[ -n "${RUN_CLANG_TIDY_BIN}" ]] || RUN_CLANG_TIDY_BIN="$(command -v "run-clang-tidy-${v}" 2>/dev/null || true)"
done
if [[ -n "${CLANG_TIDY_BIN}" && -n "${RUN_CLANG_TIDY_BIN}" ]]; then
  echo "  using $(basename "${RUN_CLANG_TIDY_BIN}") + $(basename "${CLANG_TIDY_BIN}")"
  # Limit to first-party sources; the .clang-tidy HeaderFilterRegex keeps
  # third-party headers quiet. -clang-tidy-binary pins the matching (possibly
  # version-suffixed) clang-tidy so run-clang-tidy does not look for a plain one.
  TIDY_REGEX='core/(moonshine-utils|ort-utils|bin-tokenizer)/|core/[^/]*\.(cpp|cc)$|core/moonshine-tts/src/'
  if "${RUN_CLANG_TIDY_BIN}" -clang-tidy-binary "${CLANG_TIDY_BIN}" \
        -p "${BUILD_DIR}" -quiet "${TIDY_REGEX}" \
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
  echo "  skip: clang-tidy / run-clang-tidy not found (install the 'clang-tidy' apt package)"
fi

# ---------------------------------------------------------------------------
# 4. ThreadSanitizer (data races), advisory unless TSAN_STRICT=1.
# ---------------------------------------------------------------------------
# TSan cannot be combined with ASan, so this is a full, clean rebuild with
# -DMOONSHINE_SANITIZER=thread. We remove the ASan build dirs first: they are no
# longer needed (tests ran in step 2, clang-tidy in step 3), and the per-module
# build directories are hard-coded absolute paths, so only one sanitizer build
# system can own them at a time. The libFuzzer harnesses in reliability/build are
# left in place for the fuzzing stage that follows.
echo ""
echo "--- ThreadSanitizer ---"
if [[ "${TSAN}" != "1" ]]; then
  echo "  skip: TSAN=0"
elif [[ "${TEST_WORKDIR}" != "${REPO_ROOT_DIR}/test-assets" ]]; then
  echo "  skip: test-assets missing (threaded tests need fixtures)"
else
  echo "  full clean rebuild with -fsanitize=thread"
  rm -rf "${BUILD_DIR}" "${TSAN_BUILD_DIR}" \
    "${CORE_DIR}/moonshine-utils/build" \
    "${CORE_DIR}/bin-tokenizer/build" \
    "${CORE_DIR}/ort-utils/build" \
    "${CORE_DIR}/moonshine-tts/build"
  tsan_built=1
  if ! cmake -S "${CORE_DIR}" -B "${TSAN_BUILD_DIR}" \
        -DMOONSHINE_RELIABILITY=ON \
        -DMOONSHINE_SANITIZER=thread \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        >"${ARTIFACTS_DIR}/tsan-configure.log" 2>&1; then
    record_failure "tsan-configure (see artifacts/tsan-configure.log)"
    tsan_built=0
  fi
  # Build only the binaries the TSan stage actually runs. The concurrency test
  # is the one that gives TSan real cross-thread contention; voice-activity is a
  # cheap threaded sanity check. Building the whole tree would just add minutes.
  if [[ "${tsan_built}" == "1" ]] && \
     ! cmake --build "${TSAN_BUILD_DIR}" -j "${JOBS}" \
        --target transcriber-concurrency-test voice-activity-detector-test \
        >"${ARTIFACTS_DIR}/tsan-build.log" 2>&1; then
    record_failure "tsan-build (see artifacts/tsan-build.log)"
    tsan_built=0
  fi
  if [[ "${tsan_built}" == "1" ]]; then
    # Halt on the first race. The suppressions file silences races inside the
    # uninstrumented onnxruntime and vendored code (see the file for rationale).
    export TSAN_OPTIONS="halt_on_error=1:abort_on_error=1:second_deadlock_stack=1:suppressions=${TSAN_SUPPRESSIONS}"
    # Force onnxruntime onto the calling thread. TSan's pthread interceptors
    # deadlock inside onnxruntime's uninstrumented thread pool (a full-inference
    # test hangs indefinitely), so the model-load paths honour this flag to run
    # sequentially with no worker threads. First-party locking is unaffected, so
    # the races we care about are still exercised. See ort_maybe_force_single_thread.
    export MOONSHINE_ORT_SINGLE_THREAD=1
    # libmoonshine.so is built into the TSan tree; make sure it is found first.
    export LD_LIBRARY_PATH="${TSAN_BUILD_DIR}:${ORT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    # TSan can trip over high-entropy ASLR ("unexpected memory mapping"); running
    # the child with randomization disabled avoids it and needs no root.
    tsan_prefix=()
    if command -v setarch >/dev/null 2>&1; then
      tsan_prefix=(setarch "$(uname -m)" -R)
    fi
    # Bound each test so a hang (e.g. an unexpected onnxruntime thread pool)
    # can never stall the pipeline: SIGTERM at the limit, SIGKILL 30s later.
    tsan_runner=()
    if command -v timeout >/dev/null 2>&1; then
      tsan_runner=(timeout -k 30 "${TSAN_TEST_TIMEOUT}")
    fi
    # transcriber-concurrency drives many streams through the shared transcriber
    # in parallel: this is what actually exercises the first-party mutexes under
    # TSan. voice-activity is a cheap threaded sanity check. The other integration
    # binaries are single-threaded, so they add build/run time with no race
    # coverage and are intentionally omitted.
    tsan_clean=1
    for entry in \
        "concurrency:${TSAN_BUILD_DIR}/transcriber-concurrency-test" \
        "voice-activity:${TSAN_BUILD_DIR}/voice-activity-detector-test"; do
      name="${entry%%:*}"
      bin="${entry#*:}"
      if [[ ! -x "${bin}" ]]; then
        echo "  skip tsan:${name} (missing binary)"
        continue
      fi
      echo "  run  tsan:${name}"
      rc=0
      ( cd "${TEST_WORKDIR}" && "${tsan_runner[@]}" "${tsan_prefix[@]}" "${bin}" ) \
            >"${ARTIFACTS_DIR}/tsan-${name}.log" 2>&1 || rc=$?
      if [[ "${rc}" == "124" || "${rc}" == "137" ]]; then
        tsan_clean=0
        echo "  tsan:${name}: TIMED OUT after ${TSAN_TEST_TIMEOUT}s (see artifacts/tsan-${name}.log)"
        record_failure "tsan:${name} timed out (see artifacts/tsan-${name}.log)"
      elif [[ "${rc}" != "0" ]]; then
        tsan_clean=0
        if [[ "${TSAN_STRICT}" == "1" ]]; then
          record_failure "tsan:${name} (see artifacts/tsan-${name}.log)"
        else
          echo "  tsan:${name}: findings recorded (advisory) in artifacts/tsan-${name}.log"
        fi
      fi
    done
    unset MOONSHINE_ORT_SINGLE_THREAD
    if [[ "${tsan_clean}" == "1" ]]; then
      echo "  ThreadSanitizer: clean"
    fi
    # Restore ASan options/library path for the fuzzing stage below.
    export LD_LIBRARY_PATH="${ORT_LIB_DIR}:${LD_LIBRARY_PATH:-}"
    unset TSAN_OPTIONS
  fi
fi

# ---------------------------------------------------------------------------
# 5. Fuzzing (time-boxed, per module).
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
