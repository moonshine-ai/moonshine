#!/usr/bin/env bash
# Publish examples/web to the R2 bucket behind the moonshine.ai static site.
#
# Served from the www-moonshine-ai R2 bucket on:
#   https://moonshine.ai
#   https://www.moonshine.ai
#   https://staging.moonshine.ai
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
#                             (default: moonshine.ai)
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
WWW_HOST="${MOONSHINE_WWW_HOST:-moonshine.ai}"

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
#
# Marketing/legal pages (license, enterprise, …) are published to this bucket
# separately and are intentionally not part of the open-source examples/web
# tree. Excluding them here means rclone sync will neither upload nor delete
# those remote objects.
cat >"${FILTER}" <<'EOF'
- serve.mjs
- home-cta-final.png
- wrangler.jsonc
- wrangler.toml
- .assetsignore
- _headers
- .DS_Store
- **/.DS_Store
- /license/**
- /enterprise/**
- /use-policy/**
- /community-license/**
- /moonshine_community_license.txt
- /assets/legacy-site.css
+ /**
EOF

collect_transferred() {
  python3 - "$1" <<'PY'
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
    obj = entry.get("object")
    msg = entry.get("msg", "")
    if obj and ("Copied" in msg or "Updated" in msg or "Deleted" in msg):
        objects.append(obj)
print("\n".join(objects))
PY
}

echo "Sync ${SRC} -> ${DEST} (Cache-Control: ${CACHE_CONTROL})" >&2
# --checksum matches upload-tts-assets.sh. sync (not copy) is correct here:
# removed demos should disappear from the site.
# shellcheck disable=SC2086
rclone sync "${SRC}" "${DEST}" --checksum \
  --filter-from "${FILTER}" \
  --header-upload "Cache-Control: ${CACHE_CONTROL}" \
  --use-json-log --log-level INFO --log-file "${LOG}" ${EXTRA}

TRANSFERRED=$(collect_transferred "${LOG}")

# Same-origin copy of the WASM binding. Deployed demos load /wasm/dist so
# pthread Workers are not cross-origin under COOP/COEP (see pageConfig).
WASM_DIST="${ROOT}/language-bindings/wasm/dist"
if [[ -f "${WASM_DIST}/index.js" && -f "${WASM_DIST}/moonshine.wasm" ]]; then
  : >"${LOG}"
  echo "Sync ${WASM_DIST} -> ${DEST}/wasm/dist" >&2
  # shellcheck disable=SC2086
  rclone sync "${WASM_DIST}" "${DEST}/wasm/dist" --checksum \
    --exclude '*.map' \
    --header-upload "Cache-Control: ${CACHE_CONTROL}" \
    --use-json-log --log-level INFO --log-file "${LOG}" ${EXTRA}
  wasm_transferred=$(collect_transferred "${LOG}")
  if [[ -n "${wasm_transferred}" ]]; then
    while IFS= read -r key; do
      [[ -n "${key}" ]] || continue
      TRANSFERRED+=$'\n'"wasm/dist/${key}"
    done <<<"${wasm_transferred}"
  fi
else
  echo "Warning: ${WASM_DIST} is incomplete; demos expect /wasm/dist/index.js." >&2
  echo "Build with: (cd wasm && npm run build) && scripts/build-wasm.sh" >&2
fi

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
  urls+=("https://staging.moonshine.ai/" "https://www.moonshine.ai/")
  if ((${#urls[@]})); then
    echo "Purging ${#urls[@]} URL(s)..." >&2
    cdn_purge_urls "${urls[@]}"
  fi
else
  echo "Edge cache not purged. Re-run with MOONSHINE_INVALIDATE_CDN=1 after" >&2
  echo "replacing bytes that browsers may already hold." >&2
fi

echo "Site: https://${WWW_HOST}/" >&2
