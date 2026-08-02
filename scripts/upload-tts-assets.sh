#!/usr/bin/env bash
# Upload core/moonshine-tts/data to the Moonshine download bucket under tts/, using
# `rclone copy` so incremental updates stay small. After upload, HTTPS URLs of the form
#   https://download.moonshine.ai/tts/<canonical-key>
# match object names <bucket>/tts/<canonical-key>.
#
# download.moonshine.ai is served from a Cloudflare R2 bucket, fronted by the Cloudflare
# cache. The Google Cloud Storage bucket and Cloud CDN load balancer that used to serve this
# hostname have been deleted, so uploading to GCS would reach nobody. We stamp a long
# Cache-Control on every uploaded object (see MOONSHINE_TTS_CACHE_CONTROL) so the edge and
# clients keep them for a long time. Because TTS asset keys are path-based (not
# content-hashed), any in-place overwrite of an existing key must be followed by a cache
# purge, otherwise stale content is served up to the TTL. Set MOONSHINE_INVALIDATE_CDN=1 to
# purge exactly the objects this run replaced.
#
# Prerequisites: rclone, and the Cloudflare R2 credentials described in
# cdn-publish-common.sh (plus CLOUDFLARE_API_TOKEN when MOONSHINE_INVALIDATE_CDN is set).
#
# Environment:
#   MOONSHINE_R2_BUCKET        Bucket name (default: download-moonshine-ai)
#   MOONSHINE_TTS_CACHE_CONTROL Cache-Control applied to uploaded objects
#                              (default: "public, max-age=2592000"; 30 days).
#   MOONSHINE_INVALIDATE_CDN   When non-empty, purge the objects this run replaced.
#   MOONSHINE_RCLONE_EXTRA     Extra flags passed to rclone (e.g. "--dry-run", or
#                              "--delete-excluded" if you really mean to remove things).
#
# Do not turn this into `rclone sync`, which deletes destination objects that are absent
# locally. Piper voices ship as ORT now and their `.onnx` originals are no longer in this
# tree, but released clients still ask for `.onnx` keys, so deleting remote objects absent
# locally would break them. `copy` is what keeps them working.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/core/moonshine-tts/data"
EXTRA="${MOONSHINE_RCLONE_EXTRA:-}"
CACHE_CONTROL="${MOONSHINE_TTS_CACHE_CONTROL:-public, max-age=2592000}"

if [[ ! -d "${SRC}" ]]; then
  echo "Source directory not found: ${SRC}" >&2
  exit 1
fi

source "${ROOT}/scripts/cdn-publish-common.sh"
cdn_setup_r2

DEST="r2:${CDN_R2_BUCKET}/tts"
LOG=$(mktemp)
trap 'rm -f "${LOG}"' EXIT

echo "Copy ${SRC} -> ${DEST} (Cache-Control: ${CACHE_CONTROL})" >&2
# --checksum compares hashes rather than mtime/size, matching what gsutil rsync -c did.
# --header-upload is how the caching header reaches R2; without it objects are served with
# no Cache-Control at all. The JSON log is parsed below to find what actually changed.
# shellcheck disable=SC2086
rclone copy "${SRC}" "${DEST}" --checksum \
  --header-upload "Cache-Control: ${CACHE_CONTROL}" \
  --use-json-log --log-level INFO --log-file "${LOG}" ${EXTRA}

TRANSFERRED=$(python3 - "${LOG}" <<'PY'
import json, sys
objects = []
for line in open(sys.argv[1], errors="replace"):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        entry = json.loads(line)
    except ValueError:
        continue
    # rclone logs "Copied (new)", "Copied (replaced existing)", "Updated" and similar.
    if entry.get("object") and entry.get("msg", "").startswith(("Copied", "Updated")):
        objects.append(entry["object"])
print("\n".join(sorted(set(objects))))
PY
)

if [[ -n "${TRANSFERRED}" ]]; then
  echo "Uploaded $(printf '%s\n' "${TRANSFERRED}" | grep -c .) object(s)." >&2
else
  echo "Nothing changed; every object already matched." >&2
fi

if [[ -n "${MOONSHINE_INVALIDATE_CDN:-}" && -n "${TRANSFERRED}" ]]; then
  echo "Purging the Cloudflare cache for the objects this run replaced..." >&2
  urls=()
  while IFS= read -r object; do
    [[ -n "${object}" ]] && urls+=("https://${CDN_HOST}/tts/${object}")
  done <<<"${TRANSFERRED}"
  cdn_purge_urls "${urls[@]}"
fi
echo "Done." >&2
