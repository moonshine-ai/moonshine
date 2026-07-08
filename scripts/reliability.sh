#!/usr/bin/env bash
#
# Periodic reliability driver. Run this from the Mac (or any dev machine). It:
#
#   1. Runs the fast, local static checks (formatting + banned constructs).
#   2. Syncs the working tree to a native Linux x86 box over SSH (so even
#      uncommitted changes are exercised), preserving the box's large model /
#      test assets.
#   3. Runs scripts/reliability-remote.sh there (ASan/UBSan build + test suite,
#      clang-tidy, and time-boxed per-module fuzzing).
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
#   TIDY_STRICT        1 => clang-tidy findings fail the run (default 0)
#   FORMAT_STRICT      1 => formatting drift fails the run (default 0, advisory).
#                      Run scripts/format-core.sh to fix drift wholesale.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
CORE_DIR="${REPO_ROOT_DIR}/core"

RELIABILITY_HOST="${RELIABILITY_HOST:-petes-alienware-pc}"
REMOTE_DIR="${REMOTE_DIR:-moonshine}"
FUZZ_SECONDS="${FUZZ_SECONDS:-900}"
TIDY_STRICT="${TIDY_STRICT:-0}"
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
# Overlay our source onto the box's existing checkout without --delete so its
# LFS-managed models / test assets (excluded below) are preserved.
ssh "${RELIABILITY_HOST}" "mkdir -p '${REMOTE_DIR}'"
rsync -az --info=stats1 \
  --exclude '.git/' \
  --exclude '.env' \
  --exclude '**/build/' \
  --exclude 'core/reliability/artifacts/' \
  --exclude 'core/reliability/corpus/' \
  --exclude 'test-assets/' \
  --exclude 'core/moonshine-tts/data/' \
  --exclude '**/__pycache__/' \
  --exclude '*.dylib' \
  --exclude '*.so' \
  "${REPO_ROOT_DIR}/" \
  "${RELIABILITY_HOST}:${REMOTE_DIR}/"

echo ""
echo "=== Running remote reliability checks ==="
remote_status=0
ssh "${RELIABILITY_HOST}" \
  "cd '${REMOTE_DIR}' && FUZZ_SECONDS='${FUZZ_SECONDS}' TIDY_STRICT='${TIDY_STRICT}' bash scripts/reliability-remote.sh" \
  || remote_status=$?

echo ""
echo "=== Copying artifacts back ==="
mkdir -p "${CORE_DIR}/reliability/artifacts"
rsync -az \
  "${RELIABILITY_HOST}:${REMOTE_DIR}/core/reliability/artifacts/" \
  "${CORE_DIR}/reliability/artifacts/" \
  || echo "warning: no artifacts to copy back" >&2

if [[ "${remote_status}" -eq 0 ]]; then
  echo ""
  echo "Reliability run PASSED. Artifacts in core/reliability/artifacts/."
else
  echo ""
  echo "Reliability run FAILED (exit ${remote_status}). See core/reliability/artifacts/." >&2
fi
exit "${remote_status}"
