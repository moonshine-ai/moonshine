#!/bin/bash

# Upload one or more assets to a GitHub release, creating the release if it does
# not yet exist, with a bounded per-attempt timeout and retries.
#
# `gh release upload` has no client-side timeout, so a stalled TLS connection to
# GitHub can leave it hanging indefinitely. Each attempt is wrapped with
# `timeout` (GNU `timeout`, or `gtimeout` from coreutils on macOS) and retried a
# few times before giving up. This is the bash companion to
# scripts/gh-upload-retry.ps1 (used by the Windows release path) and is shared by
# scripts/publish-binary.sh and scripts/build-pip-docker.sh so the retry
# behaviour lives in exactly one place.
#
# Usage:
#   scripts/gh-upload-retry.sh <version> <asset-path> [<asset-path> ...]
#
# Environment:
#   REPO   GitHub repository (default: moonshine-ai/moonshine)

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <version> <asset-path> [<asset-path> ...]" >&2
    exit 1
fi

VERSION="$1"
shift
REPO="${REPO:-moonshine-ai/moonshine}"

# Create the release if it is missing. Pass -R explicitly because callers may
# invoke this from outside a git checkout (e.g. a temp dir), so gh can't infer
# the repo from the working directory.
if ! gh release view "v${VERSION}" -R "${REPO}" >/dev/null 2>&1; then
    gh release create "v${VERSION}" -R "${REPO}" --title "v${VERSION}" --notes "Release v${VERSION}"
fi

if command -v timeout >/dev/null 2>&1; then
    TIMEOUT_CMD=timeout
elif command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_CMD=gtimeout
else
    TIMEOUT_CMD=""
fi

UPLOAD_TIMEOUT_SEC=180
UPLOAD_RETRIES=5

for ASSET in "$@"; do
    ASSET_NAME="$(basename "${ASSET}")"
    upload_ok=0
    for attempt in $(seq 1 ${UPLOAD_RETRIES}); do
        echo "[gh-upload-retry] Attempt ${attempt}/${UPLOAD_RETRIES}: uploading ${ASSET_NAME} to release v${VERSION}..."
        # `set -e` would otherwise abort the script on a failed attempt before we
        # get a chance to retry, so guard the call and inspect its status.
        rc=0
        if [ -n "${TIMEOUT_CMD}" ]; then
            "${TIMEOUT_CMD}" "${UPLOAD_TIMEOUT_SEC}" gh release upload "v${VERSION}" "${ASSET}" -R "${REPO}" --clobber || rc=$?
        else
            gh release upload "v${VERSION}" "${ASSET}" -R "${REPO}" --clobber || rc=$?
        fi
        if [ ${rc} -eq 0 ]; then
            upload_ok=1
            break
        fi
        echo "[gh-upload-retry] Attempt ${attempt} failed or timed out (exit ${rc})."
        if [ ${attempt} -lt ${UPLOAD_RETRIES} ]; then
            sleep 5
        fi
    done
    if [ ${upload_ok} -ne 1 ]; then
        echo "[gh-upload-retry] ERROR: upload of ${ASSET_NAME} to v${VERSION} failed after ${UPLOAD_RETRIES} attempts." >&2
        exit 1
    fi
    echo "[gh-upload-retry] Uploaded ${ASSET_NAME} to release v${VERSION}."
done
