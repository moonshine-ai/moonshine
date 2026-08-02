#!/usr/bin/env bash
# Shared helpers for publishing assets to the Moonshine download CDN.
#
# download.moonshine.ai is served from the Cloudflare R2 bucket
# `download-moonshine-ai`. It used to be a Google Cloud Storage bucket behind a
# Cloud CDN load balancer, but that stack was retired and gs://download.moonshine.ai
# no longer exists, so uploading there reaches nobody.
#
# Sourced by upload-diarization-models.sh and upload-tts-assets.sh.
#
# Environment (all four normally live in your shell profile):
#   CLOUDFLARE_ACCOUNT_ID         R2 account, used to build the S3 endpoint.
#   CLOUDFLARE_ACCESS_KEY_ID      R2 access key with write access to the bucket.
#   CLOUDFLARE_SECRET_ACCESS_KEY  Matching secret.
#   CLOUDFLARE_API_TOKEN          Only needed for cache purges.

CDN_R2_BUCKET="${MOONSHINE_R2_BUCKET:-download-moonshine-ai}"
CDN_ZONE_NAME="${MOONSHINE_CDN_ZONE:-moonshine.ai}"
# Read by the scripts that source this file, not here.
# shellcheck disable=SC2034
CDN_HOST="${MOONSHINE_CDN_HOST:-download.moonshine.ai}"

# Points an `r2:` rclone remote at the bucket using environment variables only,
# so this works without an rclone config file on any machine that has the
# credentials.
cdn_setup_r2() {
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found. Install it: https://rclone.org/install/" >&2
    return 1
  fi
  local missing=()
  for var in CLOUDFLARE_ACCOUNT_ID CLOUDFLARE_ACCESS_KEY_ID CLOUDFLARE_SECRET_ACCESS_KEY; do
    [[ -n "${!var:-}" ]] || missing+=("$var")
  done
  if (( ${#missing[@]} )); then
    echo "Missing required environment: ${missing[*]}" >&2
    echo "These are the R2 credentials for the ${CDN_R2_BUCKET} bucket." >&2
    return 1
  fi

  export RCLONE_CONFIG_R2_TYPE=s3
  export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
  export RCLONE_CONFIG_R2_REGION=auto
  export RCLONE_CONFIG_R2_ACCESS_KEY_ID="${CLOUDFLARE_ACCESS_KEY_ID}"
  export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="${CLOUDFLARE_SECRET_ACCESS_KEY}"
  export RCLONE_CONFIG_R2_ENDPOINT="https://${CLOUDFLARE_ACCOUNT_ID}.r2.cloudflarestorage.com"
}

cdn_zone_id() {
  if [[ -n "${CLOUDFLARE_ZONE_ID:-}" ]]; then
    echo "${CLOUDFLARE_ZONE_ID}"
    return 0
  fi
  local response
  response=$(curl -fsS \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    "https://api.cloudflare.com/client/v4/zones?name=${CDN_ZONE_NAME}") || return 1
  python3 -c '
import json, sys
data = json.load(sys.stdin)
zones = data.get("result") or []
if not zones:
    sys.exit("no zone found; check CLOUDFLARE_API_TOKEN has Zone:Read")
print(zones[0]["id"])
' <<<"${response}"
}

# Purges specific URLs from the Cloudflare edge cache. Cloudflare accepts at most
# 30 files per request, and purge-by-prefix needs an Enterprise plan, so callers
# pass the exact URLs they overwrote rather than a wildcard.
cdn_purge_urls() {
  (( $# )) || return 0
  if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
    echo "CLOUDFLARE_API_TOKEN is not set, so the cache cannot be purged." >&2
    echo "The new bytes will still be served once the old cache entry expires." >&2
    return 1
  fi
  local zone
  zone=$(cdn_zone_id) || return 1

  local batch=()
  _cdn_flush_batch() {
    (( ${#batch[@]} )) || return 0
    local payload response
    payload=$(printf '%s\n' "${batch[@]}" | python3 -c '
import json, sys
print(json.dumps({"files": [l.strip() for l in sys.stdin if l.strip()]}))
')
    # Deliberately not -f: on a failure we want to read Cloudflare's error body,
    # which names the missing permission, rather than just a status code.
    if ! response=$(curl -sS -X POST \
      -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
      -H "Content-Type: application/json" \
      --data "${payload}" \
      "https://api.cloudflare.com/client/v4/zones/${zone}/purge_cache"); then
      echo "Cache purge request failed to reach Cloudflare." >&2
      return 1
    fi
    if ! python3 -c '
import json, sys
body = json.load(sys.stdin)
if body.get("success"):
    sys.exit(0)
for error in body.get("errors") or [{}]:
    message = error.get("message") or json.dumps(body)
    sys.stderr.write("  Cloudflare refused the purge: " + str(message) + "\n")
sys.stderr.write("  The API token needs the Zone / Cache Purge permission.\n")
sys.exit(1)
' <<<"${response}"; then
      return 1
    fi
    echo "  purged ${#batch[@]} URL(s)" >&2
    batch=()
  }

  local url
  for url in "$@"; do
    batch+=("${url}")
    if (( ${#batch[@]} == 30 )); then
      _cdn_flush_batch || return 1
    fi
  done
  _cdn_flush_batch || return 1
}
