#!/usr/bin/env bash
#
# Periodic reliability driver. Run this from the Mac (or any dev machine). It:
#
#   1. Runs the fast, local static checks (formatting + banned constructs).
#   2. Syncs the working tree to a native Linux x86 box over SSH (so even
#      uncommitted changes are exercised), preserving the box's large model /
#      test assets.
#   3. Runs scripts/reliability-remote.sh there (ASan/UBSan build + test suite,
#      clang-tidy, a ThreadSanitizer rebuild + threaded tests, a long-stream
#      memory regression test, and time-boxed per-module fuzzing).
#   4. Copies logs and any crash reproducers back to core/reliability/artifacts.
#
# The whole run is designed to finish within a few hours (tune FUZZ_SECONDS).
#
# Usage:
#   scripts/reliability.sh
#
# Environment:
#   RELIABILITY_HOST   SSH host of the Linux x86 box (default: petes-alienware-pc)
#   REMOTE_DIR         repo path on that box, relative to $HOME (default: moonshine)
#   FUZZ_SECONDS       seconds per fuzz target (default 900 => ~1h of fuzzing)
#   TIDY_UPDATE_BASELINE 1 => regenerate core/.clang-tidy-baseline from this run
#                        (copied back locally to commit) instead of gating on it
#   TSAN               1 => run the ThreadSanitizer stage (default 1); 0 to skip
#   TSAN_STRICT        1 => TSan findings fail the run (default 0, advisory)
#   FORMAT_STRICT      1 => formatting drift fails the run (default 0, advisory).
#                      Run scripts/format-core.sh to fix drift wholesale.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
CORE_DIR="${REPO_ROOT_DIR}/core"

RELIABILITY_HOST="${RELIABILITY_HOST:-petes-alienware-pc}"
REMOTE_DIR="${REMOTE_DIR:-moonshine}"
FUZZ_SECONDS="${FUZZ_SECONDS:-900}"
TIDY_UPDATE_BASELINE="${TIDY_UPDATE_BASELINE:-0}"
TSAN="${TSAN:-1}"
TSAN_STRICT="${TSAN_STRICT:-0}"
FORMAT_STRICT="${FORMAT_STRICT:-0}"

echo "=== Local static checks ==="
echo "--- banned constructs (hard gate) ---"
"${SCRIPTS_DIR}/check-banned-constructs.sh"

echo "--- formatting ---"
if "${SCRIPTS_DIR}/format-core.sh" --check; then
  echo "formatting: clean"
elif [[ "${FORMAT_STRICT}" == "1" ]]; then
  echo "error: formatting drift (FORMAT_STRICT=1)." >&2
  exit 1
else
  echo "warning: formatting drift above is advisory; run scripts/format-core.sh to fix." >&2
fi

echo ""
echo "=== Syncing working tree to ${RELIABILITY_HOST}:${REMOTE_DIR} ==="
# Derive the exclude list from git's own ignore rules (root + nested .gitignore,
# .git/info/exclude, global excludes) so we never copy build output, virtualenvs,
# caches, or other temporary files. --directory collapses fully-ignored dirs to a
# single entry; a leading '/' anchors each pattern to the transfer root.
IGNORE_LIST="$(mktemp)"
trap 'rm -f "${IGNORE_LIST}"' EXIT
( cd "${REPO_ROOT_DIR}" \
  && git ls-files --others --ignored --exclude-standard --directory ) \
  | sed 's#^#/#' >"${IGNORE_LIST}"

# Overlay our source onto the box without --delete. The box is a plain rsync
# target (not a git checkout), so we must ship everything the build, tests, and
# fuzzers need: that includes test-assets/ (~190MB of models/fixtures the tests
# and fuzz seeds load) and the Linux onnxruntime libs. We still skip .git, the
# large TTS data (no TTS test runs here), and the non-Linux onnxruntime prebuilt
# libraries (~950MB) that this Linux-x86 run never links against.
ssh "${RELIABILITY_HOST}" "mkdir -p '${REMOTE_DIR}'"
rsync -az --stats \
  --exclude-from="${IGNORE_LIST}" \
  --exclude '/.git/' \
  --exclude '/core/moonshine-tts/data/' \
  --exclude '/core/third-party/onnxruntime/lib/android/' \
  --exclude '/core/third-party/onnxruntime/lib/ios/' \
  --exclude '/core/third-party/onnxruntime/lib/macos/' \
  --exclude '/core/third-party/onnxruntime/lib/windows/' \
  "${REPO_ROOT_DIR}/" \
  "${RELIABILITY_HOST}:${REMOTE_DIR}/"

echo ""
echo "=== Running remote reliability checks ==="
remote_status=0
ssh "${RELIABILITY_HOST}" \
  "cd '${REMOTE_DIR}' && FUZZ_SECONDS='${FUZZ_SECONDS}' TIDY_UPDATE_BASELINE='${TIDY_UPDATE_BASELINE}' TSAN='${TSAN}' TSAN_STRICT='${TSAN_STRICT}' bash scripts/reliability-remote.sh" \
  || remote_status=$?

echo ""
echo "=== Copying artifacts back ==="
mkdir -p "${CORE_DIR}/reliability/artifacts"
rsync -az \
  "${RELIABILITY_HOST}:${REMOTE_DIR}/core/reliability/artifacts/" \
  "${CORE_DIR}/reliability/artifacts/" \
  || echo "warning: no artifacts to copy back" >&2

# When regenerating the baseline, pull the freshly written file back so it can be
# committed. On normal (gating) runs the baseline flows box-ward only.
if [[ "${TIDY_UPDATE_BASELINE}" == "1" ]]; then
  echo "--- copying regenerated clang-tidy baseline back ---"
  rsync -az \
    "${RELIABILITY_HOST}:${REMOTE_DIR}/core/.clang-tidy-baseline" \
    "${CORE_DIR}/.clang-tidy-baseline" \
    || echo "warning: could not copy clang-tidy baseline back" >&2
fi

if [[ "${remote_status}" -eq 0 ]]; then
  echo ""
  echo "Reliability run PASSED. Artifacts in core/reliability/artifacts/."
else
  echo ""
  echo "Reliability run FAILED (exit ${remote_status}). See core/reliability/artifacts/." >&2
fi
exit "${remote_status}"
