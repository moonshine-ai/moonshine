#!/usr/bin/env bash
# Publish examples/web to the R2 bucket behind the moonshine.ai static site.
#
# Staging is live at https://staging.moonshine.ai (custom domain on the
# www-moonshine-ai bucket). Production cutover points moonshine.ai /
# www.moonshine.ai at the same bucket once the wasm library is ready.
#
# R2 has no index-document feature, so directory URLs rely on Transform Rules
# in the moonshine.ai zone (see scripts/setup-web-site-rules.sh). Those rules
# also attach the COOP/COEP headers SharedArrayBuffer needs.
#
# Prerequisites: rclone, and the Cloudflare R2 credentials described in
# cdn-publish-common.sh (plus CLOUDFLARE_API_TOKEN when MOONSHINE_INVALIDATE_CDN
# is set).
#
# Environment:
#   MOONSHINE_WWW_R2_BUCKET   Bucket name (default: www-moonshine-ai)
#   MOONSHINE_WWW_HOST        Hostname used in purge URLs
#                             (default: staging.moonshine.ai)
#   MOONSHINE_WWW_CACHE_CONTROL
#                             Cache-Control for uploaded objects
#                             (default: "public, max-age=300")
#   MOONSHINE_INVALIDATE_CDN  When non-empty, purge transferred object URLs
#                             from the edge cache after upload.
#   MOONSHINE_RCLONE_EXTRA    Extra flags passed to rclone (e.g. "--dry-run").
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/examples/web"
EXTRA="${MOONSHINE_RCLONE_EXTRA:-}"
CACHE_CONTROL="${MOONSHINE_WWW_CACHE_CONTROL:-public, max-age=300}"
WWW_BUCKET="${MOONSHINE_WWW_R2_BUCKET:-www-moonshine-ai}"
WWW_HOST="${MOONSHINE_WWW_HOST:-staging.moonshine.ai}"

if [[ ! -f "${SRC}/index.html" ]]; then
  echo "Missing ${SRC}/index.html" >&2
  exit 1
fi

# Reuse the CDN rclone remote helpers; override the bucket/host for this site.
# shellcheck source=cdn-publish-common.sh
source "${ROOT}/scripts/cdn-publish-common.sh"
CDN_R2_BUCKET="${WWW_BUCKET}"
CDN_HOST="${WWW_HOST}"
cdn_setup_r2

DEST="r2:${WWW_BUCKET}"
LOG=$(mktemp)
FILTER=$(mktemp)
trap 'rm -f "${LOG}" "${FILTER}"' EXIT

# Keep the bucket identical to the publishable tree. Local-only helpers and
# design drafts stay out of the public site.
cat >"${FILTER}" <<'EOF'
- serve.mjs
- home-cta-final.png
- wrangler.jsonc
- wrangler.toml
- .assetsignore
- _headers
- .DS_Store
- **/.DS_Store
+ /**
EOF

echo "Sync ${SRC} -> ${DEST} (Cache-Control: ${CACHE_CONTROL})" >&2
# --checksum matches upload-tts-assets.sh. sync (not copy) is correct here:
# removed demos should disappear from the site.
# shellcheck disable=SC2086
rclone sync "${SRC}" "${DEST}" --checksum \
  --filter-from "${FILTER}" \
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
    # rclone JSON logs use "object" for the remote key on transfers.
    obj = entry.get("object")
    msg = entry.get("msg", "")
    if obj and ("Copied" in msg or "Updated" in msg or "Deleted" in msg):
        objects.append(obj)
print("\n".join(objects))
PY
)

count=$(grep -c . <<<"${TRANSFERRED}" || true)
echo "Transferred/deleted ${count} object(s)." >&2

if [[ -n "${MOONSHINE_INVALIDATE_CDN:-}" ]]; then
  urls=()
  while IFS= read -r key; do
    [[ -n "${key}" ]] || continue
    urls+=("https://${WWW_HOST}/${key}")
  done <<<"${TRANSFERRED}"
  # Root is rewritten to /index.html; purge both so either URL is fresh.
  urls+=("https://${WWW_HOST}/" "https://${WWW_HOST}/index.html")
  if ((${#urls[@]})); then
    echo "Purging ${#urls[@]} URL(s) on ${WWW_HOST}..." >&2
    cdn_purge_urls "${urls[@]}"
  fi
else
  echo "Edge cache not purged. Re-run with MOONSHINE_INVALIDATE_CDN=1 after" >&2
  echo "replacing bytes that browsers may already hold." >&2
fi

echo "Site: https://${WWW_HOST}/" >&2
