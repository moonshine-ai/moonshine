#!/usr/bin/env bash
# Upload core/moonshine-tts/data to the Moonshine download bucket under tts/, using
# `gcloud storage rsync` so incremental updates stay small (avoids legacy gsutil on Python 3,
# which can fail with: module 'sys' has no attribute 'maxint'). After upload, HTTPS URLs of the form
#   https://download.moonshine.ai/tts/<canonical-key>
# match object names gs://<bucket>/tts/<canonical-key> when the bucket is served from that host.
#
# download.moonshine.ai is served by a Google Cloud external HTTPS load balancer with Cloud CDN
# enabled on the backing bucket, so uploaded objects are cached at the edge. We stamp a long
# Cache-Control on every uploaded object (see MOONSHINE_TTS_CACHE_CONTROL) so the CDN and clients
# keep them for a long time. Because TTS asset keys are path-based (not content-hashed), any
# in-place overwrite of an existing key must be followed by a CDN invalidation, otherwise stale
# content is served up to the TTL. Set MOONSHINE_INVALIDATE_CDN=1 to invalidate /tts/* after upload.
#
# Prerequisites: Google Cloud SDK (gcloud), and credentials with storage.objects.create/list on the bucket
# (plus compute.urlMaps.invalidateCache when MOONSHINE_INVALIDATE_CDN is set).
#
# Environment:
#   MOONSHINE_TTS_GCS_BUCKET   Bucket name (default: download.moonshine.ai)
#   MOONSHINE_TTS_CACHE_CONTROL Cache-Control applied to uploaded objects
#                              (default: "public, max-age=2592000"; 30 days).
#   MOONSHINE_INVALIDATE_CDN   When non-empty, run a Cloud CDN invalidation for /tts/* after upload.
#   MOONSHINE_DOWNLOAD_URL_MAP URL map to invalidate (default: download-lb).
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
CACHE_CONTROL="${MOONSHINE_TTS_CACHE_CONTROL:-public, max-age=2592000}"
URL_MAP="${MOONSHINE_DOWNLOAD_URL_MAP:-download-lb}"

if [[ ! -d "${SRC}" ]]; then
  echo "Source directory not found: ${SRC}" >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Install Google Cloud SDK: https://cloud.google.com/sdk" >&2
  exit 1
fi

echo "Sync ${SRC} -> ${DEST} (Cache-Control: ${CACHE_CONTROL})" >&2
# --checksums-only matches gsutil rsync -c (compare hashes, not just mtime/size).
# --cache-control stamps the long-lived caching header on newly uploaded/updated objects.
# shellcheck disable=SC2086
gcloud storage rsync "${SRC}" "${DEST}" --recursive --checksums-only \
  --cache-control="${CACHE_CONTROL}" ${EXTRA}

if [[ -n "${MOONSHINE_INVALIDATE_CDN:-}" ]]; then
  echo "Invalidating Cloud CDN cache for /tts/* on url-map ${URL_MAP}..." >&2
  gcloud compute url-maps invalidate-cdn-cache "${URL_MAP}" --path "/tts/*" --async
fi
echo "Done." >&2
