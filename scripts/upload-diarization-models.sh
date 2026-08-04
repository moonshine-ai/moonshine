#!/usr/bin/env bash
# Publish the speaker diarization models to the Moonshine download bucket, so
# that the URLs in the native catalog resolve:
#   https://download.moonshine.ai/model/diarization-community1/segmentation.ort
#   https://download.moonshine.ai/model/diarization-community1/embedding.ort
#
# These two were compiled into the library until version 26.8; see
# docs/diarization-models.md for why they became a download. Local copies under
# test-assets/diarization are fetched from the CDN (scripts/fetch-voice-assets.sh)
# so tests run against exactly the bytes clients download.
#
# The catalog pins a directory rather than overwriting one, because the
# clustering parameters still compiled into the library were fitted against this
# exact pair of models. A new pair means a new directory in
# core/moonshine-model-catalog.cpp and a matching library release, not an
# in-place overwrite. That also means this script should never need a cache
# purge: it writes objects that did not exist before. Set
# MOONSHINE_INVALIDATE_CDN if you are overwriting anyway and know why.
#
# After running this, regenerate the integrity metadata, which reads sizes and
# checksums back off the CDN, and commit the result:
#   python3 scripts/generate-model-file-metadata.py
#
# Prerequisites: rclone, and the Cloudflare R2 credentials described in
# cdn-publish-common.sh. The old Google Cloud Storage bucket behind this
# hostname has been deleted, so gcloud is no longer involved.
#
# Environment:
#   MOONSHINE_R2_BUCKET         Bucket name (default: download-moonshine-ai)
#   MOONSHINE_MODEL_CACHE_CONTROL  Cache-Control for uploaded objects
#                               (default: "public, max-age=2592000"; 30 days).
#   MOONSHINE_INVALIDATE_CDN    When non-empty, purge the uploaded URLs from the
#                               Cloudflare edge cache after upload.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/test-assets/diarization"
REMOTE_DIR="model/diarization-community1"
CACHE_CONTROL="${MOONSHINE_MODEL_CACHE_CONTROL:-public, max-age=2592000}"

source "${ROOT}/scripts/cdn-publish-common.sh"
cdn_setup_r2

DEST="r2:${CDN_R2_BUCKET}/${REMOTE_DIR}"

# The file list comes from the library rather than a glob, so that this script
# and the manifest clients download cannot drift apart.
FILES=(segmentation.ort embedding.ort)

if [[ ! -f "${SRC}/segmentation.ort" || ! -f "${SRC}/embedding.ort" ]]; then
  echo "Fetching diarization fixtures via scripts/fetch-voice-assets.sh..." >&2
  "${ROOT}/scripts/fetch-voice-assets.sh" test-assets
fi

for file in "${FILES[@]}"; do
  if [[ ! -f "${SRC}/${file}" ]]; then
    echo "Missing ${SRC}/${file} after fetch." >&2
    exit 1
  fi
  # Guard against uploading a truncated/corrupt file. ORT flatbuffers carry
  # "ORTM" as the file identifier at offset 4.
  if [[ "$(dd if="${SRC}/${file}" bs=1 skip=4 count=4 2>/dev/null)" != "ORTM" ]]; then
    echo "${SRC}/${file} is not an ORT model (no ORTM magic)." >&2
    exit 1
  fi
done

echo "Upload ${SRC} -> ${DEST} (Cache-Control: ${CACHE_CONTROL})" >&2
# --header-upload is how the caching header reaches R2; rclone drops metadata
# that is not asked for explicitly, and an object uploaded without it is served
# with no Cache-Control at all.
for file in "${FILES[@]}"; do
  rclone copyto "${SRC}/${file}" "${DEST}/${file}" \
    --header-upload "Cache-Control: ${CACHE_CONTROL}"
done

if [[ -n "${MOONSHINE_INVALIDATE_CDN:-}" ]]; then
  echo "Purging the Cloudflare cache for the uploaded objects..." >&2
  urls=()
  for file in "${FILES[@]}"; do
    urls+=("https://${CDN_HOST}/${REMOTE_DIR}/${file}")
  done
  cdn_purge_urls "${urls[@]}"
fi

echo "Done. Now run: python3 scripts/generate-model-file-metadata.py" >&2
