#!/usr/bin/env bash
# Apply the CORS policy in scripts/cdn-cors.json to the R2 buckets behind
# download.moonshine.ai and webrtc.moonshine.ai.
#
# An R2 bucket serves no Access-Control-Allow-Origin header at all until it has
# a CORS policy, and a bucket's policy is account state rather than anything in
# this repository, so it survives no reinstall and appears in no diff. Both
# buckets went live without one, which left every browser fetch of a model
# failing on a CORS error while curl and the native clients saw nothing wrong.
# This script exists so that the policy is reviewable, reproducible, and can be
# reapplied to a bucket rebuilt from scratch.
#
# Origins are "*" deliberately. These are public read-only assets that any
# developer's page is meant to be able to download, so naming origins would
# break every third-party user of the npm package. It also keeps the response
# byte-identical for every caller, which matters because Cloudflare caches
# these objects at the edge: a policy that echoed the requesting origin back
# would risk a cached response carrying some other site's origin.
#
# Setting the policy is only half the job on a bucket that has been serving
# traffic. Anything already cached at the edge keeps the headerless response it
# was stored with, and these objects are cached for 30 days, so a browser that
# asked before the change keeps failing for a month. Worse, those stale copies
# cannot be picked off by URL; see cdn_purge_urls in cdn-publish-common.sh for
# why. Hence MOONSHINE_INVALIDATE_CDN, which drops the entire zone.
#
# Prerequisites: wrangler (npx fetches it) and CLOUDFLARE_API_TOKEN with the
# Workers R2 Storage edit permission. Purging additionally needs the Zone /
# Cache Purge permission, which is a *zone* policy on the token: an
# account-scoped policy cannot hold it, and the dashboard will not offer it
# until the policy's scope is changed off "Entire Account".
#
# Environment:
#   MOONSHINE_INVALIDATE_CDN  When non-empty, purge the edge cache after setting
#                             the policy. Needed whenever the policy changes on
#                             a bucket that is already serving traffic.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POLICY="${ROOT}/scripts/cdn-cors.json"

# The download bucket also answers on download.skyshine-ai.com, which is a
# different zone and is left alone here; it is a spare name, not a path
# anything ships against.
BUCKETS=(download-moonshine-ai webrtc-moonshine-ai)

source "${ROOT}/scripts/cdn-publish-common.sh"

if [[ ! -f "${POLICY}" ]]; then
  echo "Missing ${POLICY}." >&2
  exit 1
fi

for bucket in "${BUCKETS[@]}"; do
  echo "Applying ${POLICY} to ${bucket}..." >&2
  npx --yes wrangler r2 bucket cors set "${bucket}" --file "${POLICY}" --force
  npx --yes wrangler r2 bucket cors list "${bucket}"
done

echo >&2
echo "A CORS policy change can take up to 30 seconds to propagate." >&2

# Both buckets are served from the one zone, so this purges for all of them.
if [[ -n "${MOONSHINE_INVALIDATE_CDN:-}" ]]; then
  echo "Purging the ${CDN_ZONE_NAME} edge cache..." >&2
  cdn_purge_everything
  # Worth knowing if you go on to check the result in bulk: with the cache
  # empty every request is a cold read of R2, and a request that fails is
  # cached like any other response. A parallel sweep of the bucket can talk
  # itself into a handful of 520s that then persist. Re-purge those URLs.
  echo "  the cache is now cold; expect slow first reads" >&2
else
  echo "Objects already cached at the edge keep their headerless response until" >&2
  echo "they expire. Re-run with MOONSHINE_INVALIDATE_CDN=1 to purge them." >&2
fi
