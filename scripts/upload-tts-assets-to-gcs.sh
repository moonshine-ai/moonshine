#!/usr/bin/env bash
# Upload core/moonshine-tts/data to the Moonshine download bucket under tts/, using
# `gcloud storage rsync` so incremental updates stay small (avoids legacy gsutil on Python 3,
# which can fail with: module 'sys' has no attribute 'maxint'). After upload, HTTPS URLs of the form
#   https://download.moonshine.ai/tts/<canonical-key>
# match object names gs://<bucket>/tts/<canonical-key> when the bucket is served from that host.
#
# Prerequisites: Google Cloud SDK (gcloud), and credentials with storage.objects.create/list on the bucket.
#
# Environment:
#   MOONSHINE_TTS_GCS_BUCKET  Bucket name (default: download.moonshine.ai)
#   GSUTIL_RSYNC_EXTRA         Extra flags passed to `gcloud storage rsync` (e.g.
#                              "--delete-unmatched-destination-objects" to remove remote objects absent locally;
#                              this replaces gsutil's "-d".)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/core/moonshine-tts/data"
BUCKET="${MOONSHINE_TTS_GCS_BUCKET:-download.moonshine.ai}"
DEST="gs://${BUCKET}/tts"
EXTRA="${GSUTIL_RSYNC_EXTRA:-}"

if [[ ! -d "${SRC}" ]]; then
  echo "Source directory not found: ${SRC}" >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install Google Cloud SDK: https://cloud.google.com/sdk" >&2
  exit 1
fi

echo "Sync ${SRC} -> ${DEST}" >&2
# --checksums-only matches gsutil rsync -c (compare hashes, not just mtime/size).
# shellcheck disable=SC2086
gcloud storage rsync "${SRC}" "${DEST}" --recursive --checksums-only ${EXTRA}
echo "Done." >&2
