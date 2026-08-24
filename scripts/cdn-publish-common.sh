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
#
# Published objects are immutable. cdn_publish_file below refuses to change the
# bytes behind an existing URL, because released clients pin paths and would
# otherwise start downloading files they were never tested against. Ship updates
# as a new dated path.

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

# Uploads one file, and refuses to change the bytes behind a URL that already
# exists.
#
# Every client keys its download cache off the URL, and released libraries pin
# specific paths, so replacing an object retroactively changes what an old
# version downloads. A user who installed last year's wheel and never upgraded
# would silently start fetching graphs their runtime was never tested against —
# and because these objects carry a 30-day Cache-Control, the two populations
# disagree for a month. New bytes therefore go to a new path (a dated directory
# for models), and old paths keep serving what they always served.
#
# Re-running a publish is still safe. An object already present with identical
# content is skipped rather than refused, so a partial upload can be finished by
# repeating the command; only a *different* body is an error. That is why this
# is a comparison and not an existence check, which would make repair impossible.
#
# The comparison is explicit because rclone's `--immutable` does not cover
# `copyto`: with a named destination it overwrites regardless, which was
# verified against the live bucket rather than assumed. `--immutable` does work
# for directory-mode `copy`, which is what upload-tts-assets.sh uses.
#
# An object whose remote hash cannot be read is refused rather than replaced.
# R2 returns no usable MD5 for multipart uploads, and "cannot prove it is the
# same" has to fail closed or the guard is decorative.
#
# MOONSHINE_CDN_ALLOW_OVERWRITE=1 disables the check. It exists because a
# genuinely bad object (wrong bytes, a licence problem) must be retractable, but
# it is never the way to ship an update: it breaks old clients by design, and it
# needs a cache purge to take effect at all.
cdn_publish_file() {
  local src="$1" dest="$2" cache_control="$3"
  local remote_md5 local_md5

  if [[ -n "${MOONSHINE_CDN_ALLOW_OVERWRITE:-}" ]]; then
    echo "  WARNING overwriting ${dest} in place; clients pinned to this path" >&2
    echo "  will receive the new bytes, and the edge cache needs purging." >&2
  elif [[ -n "$(rclone lsf "${dest}" 2>/dev/null)" ]]; then
    remote_md5="$(rclone hashsum md5 "${dest}" 2>/dev/null | awk 'NR==1{print $1}')"
    local_md5="$(rclone hashsum md5 "${src}" 2>/dev/null | awk 'NR==1{print $1}')"
    if [[ -n "${remote_md5}" && "${remote_md5}" == "${local_md5}" ]]; then
      echo "  unchanged, already published: ${dest}" >&2
      return 0
    fi
    echo >&2
    echo "Refusing to modify ${dest}: it already exists on the CDN" >&2
    if [[ -z "${remote_md5}" ]]; then
      echo "and its content could not be verified as identical." >&2
    else
      echo "with different content (${remote_md5} vs ${local_md5})." >&2
    fi
    echo "Publish to a new dated path rather than rewriting a pinned one, so" >&2
    echo "clients on older library versions keep resolving the bytes they were" >&2
    echo "tested against. Override with MOONSHINE_CDN_ALLOW_OVERWRITE=1 only to" >&2
    echo "retract a bad object, and purge the cache afterwards." >&2
    return 1
  fi

  rclone copyto "${src}" "${dest}" \
    --header-upload "Cache-Control: ${cache_control}"
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
#
# This clears less than it looks like it does. The buckets now carry a CORS
# policy, so R2 answers any request that has an Origin header with
# `Vary: Origin`, and Cloudflare then stores one cache entry per origin. A purge
# by URL carries no Origin of its own and so only evicts the entry for requests
# that had none, which is the one native clients and curl use. Every browser
# origin keeps its own copy, for the full 30 days these objects are cached.
#
# So this is the right call after replacing bytes that only native clients
# fetch, and the wrong one if a browser might hold a stale copy. Use
# cdn_purge_everything for that.
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

# Drops the whole zone from the Cloudflare edge cache.
#
# Blunt, and deliberately so: it is the only purge that reaches every per-origin
# variant of an object, for the reason described above cdn_purge_urls. Naming
# the origins instead is not an option, because there is no way to enumerate the
# sites that have already fetched a file. The cost is a period of cache misses
# across the zone, which the origin absorbs.
cdn_purge_everything() {
  if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
    echo "CLOUDFLARE_API_TOKEN is not set, so the cache cannot be purged." >&2
    return 1
  fi
  local zone response
  zone=$(cdn_zone_id) || return 1

  if ! response=$(curl -sS -X POST \
    -H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" \
    -H "Content-Type: application/json" \
    --data '{"purge_everything":true}' \
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
sys.stderr.write("  The API token needs the Zone / Cache Purge permission,\n")
sys.stderr.write("  which is a zone policy: an account-scoped policy cannot hold it.\n")
sys.exit(1)
' <<<"${response}"; then
    return 1
  fi
  echo "  purged the whole ${CDN_ZONE_NAME} zone" >&2
}
