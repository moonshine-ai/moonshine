#!/bin/bash
set -euo pipefail

# Copy model directories that live under this example only (SRCROOT/moonshine-models)
# into the app bundle. Uses rsync so the bundle contains real files, not symlinks.

LOCAL="${SRCROOT}/moonshine-models"
DEST="${TARGET_BUILD_DIR}/${UNLOCALIZED_RESOURCES_FOLDER_PATH}/moonshine-models"

mkdir -p "${DEST}"

if [[ -d "${LOCAL}/tiny-en" ]] && [[ -n "$(ls -A "${LOCAL}/tiny-en" 2>/dev/null)" ]]; then
  mkdir -p "${DEST}/tiny-en"
  rsync -a "${LOCAL}/tiny-en/" "${DEST}/tiny-en/"
else
  echo "warning: ${LOCAL}/tiny-en is missing or empty. Ensure moonshine-models is checked out (Git LFS)." >&2
fi

if [[ -d "${LOCAL}/embeddinggemma-300m" ]] && [[ -n "$(ls -A "${LOCAL}/embeddinggemma-300m" 2>/dev/null)" ]]; then
  mkdir -p "${DEST}/embeddinggemma-300m"
  rsync -a "${LOCAL}/embeddinggemma-300m/" "${DEST}/embeddinggemma-300m/"
else
  echo "warning: ${LOCAL}/embeddinggemma-300m is missing or empty. Ensure moonshine-models is checked out (Git LFS)." >&2
fi
