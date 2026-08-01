#!/usr/bin/env bash
# Publish the speaker diarization models to the Moonshine download bucket, so
# that the URLs in the native catalog resolve:
#   https://download.moonshine.ai/model/diarization-community1/segmentation.ort
#   https://download.moonshine.ai/model/diarization-community1/embedding.ort
#
# These two were compiled into the library until version 26.8; see
# docs/diarization-models.md for why they became a download. The copies under
# test-assets/diarization are the source of truth, which is deliberate: the
# tests run against exactly the bytes clients download, so a mismatch is not
# expressible.
#
# The catalog pins a directory rather than overwriting one, because the
# clustering parameters still compiled into the library were fitted against this
# exact pair of models. A new pair means a new directory in
# core/moonshine-model-catalog.cpp and a matching library release, not an
# in-place overwrite. That also means this script should never need a CDN
# invalidation: it writes objects that did not exist before. Set
# MOONSHINE_INVALIDATE_CDN if you are overwriting anyway and know why.
#
# After running this, regenerate the integrity metadata, which reads sizes and
# checksums back off the CDN, and commit the result:
#   python3 scripts/generate-model-file-metadata.py
#
# Prerequisites: Google Cloud SDK (gcloud) and credentials with
# storage.objects.create/list on the bucket.
#
# Environment:
#   MOONSHINE_MODEL_GCS_BUCKET  Bucket name (default: download.moonshine.ai)
#   MOONSHINE_MODEL_CACHE_CONTROL  Cache-Control for uploaded objects
#                               (default: "public, max-age=2592000"; 30 days).
#   MOONSHINE_INVALIDATE_CDN    When non-empty, invalidate
#                               /model/diarization-community1/* after upload.
#   MOONSHINE_DOWNLOAD_URL_MAP  URL map to invalidate (default: download-lb).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/test-assets/diarization"
BUCKET="${MOONSHINE_MODEL_GCS_BUCKET:-download.moonshine.ai}"
REMOTE_DIR="model/diarization-community1"
DEST="gs://${BUCKET}/${REMOTE_DIR}"
CACHE_CONTROL="${MOONSHINE_MODEL_CACHE_CONTROL:-public, max-age=2592000}"
URL_MAP="${MOONSHINE_DOWNLOAD_URL_MAP:-download-lb}"

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install Google Cloud SDK: https://cloud.google.com/sdk" >&2
  exit 1
fi

# The file list comes from the library rather than a glob, so that this script
# and the manifest clients download cannot drift apart.
FILES=(segmentation.ort embedding.ort)

for file in "${FILES[@]}"; do
  if [[ ! -f "${SRC}/${file}" ]]; then
    echo "Missing ${SRC}/${file}." >&2
    echo "It is tracked with Git LFS; try 'git lfs pull'." >&2
    exit 1
  fi
  # Guard against uploading an LFS pointer, which is a small text file that
  # would otherwise publish cleanly and fail for every client. ORT flatbuffers
  # carry "ORTM" as the file identifier at offset 4.
  if [[ "$(dd if="${SRC}/${file}" bs=1 skip=4 count=4 2>/dev/null)" != "ORTM" ]]; then
    echo "${SRC}/${file} is not an ORT model (no ORTM magic)." >&2
    echo "If this is a Git LFS pointer, run 'git lfs pull'." >&2
    exit 1
  fi
done

echo "Upload ${SRC} -> ${DEST} (Cache-Control: ${CACHE_CONTROL})" >&2
for file in "${FILES[@]}"; do
  gcloud storage cp "${SRC}/${file}" "${DEST}/${file}" \
    --cache-control="${CACHE_CONTROL}"
done

if [[ -n "${MOONSHINE_INVALIDATE_CDN:-}" ]]; then
  echo "Invalidating Cloud CDN cache for /${REMOTE_DIR}/* on url-map ${URL_MAP}..." >&2
  gcloud compute url-maps invalidate-cdn-cache "${URL_MAP}" \
    --path "/${REMOTE_DIR}/*" --async
fi

echo "Done. Now run: python3 scripts/generate-model-file-metadata.py" >&2
