#!/usr/bin/env bash
# Apply (or re-apply) the Transform Rules the moonshine.ai static site needs.
#
# R2 custom domains serve objects by exact key. There is no mainPageSuffix, so
# "/" and "/stt/" must be rewritten to their index.html objects. The WASM demos
# also need Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy so the
# browser exposes SharedArrayBuffer — the same headers examples/web/serve.mjs
# sets locally.
#
# Hosts covered:
#   staging.moonshine.ai   — live staging (R2 custom domain on www-moonshine-ai)
#   moonshine.ai           — production cutover (rules inert until DNS is
#                            orange-clouded onto the same bucket)
#   www.moonshine.ai       — same
#
# Also preserves the existing webrtc.moonshine.ai root rewrite.
#
# Prerequisites: CLOUDFLARE_API_TOKEN with Zone Transform Rules edit on
# moonshine.ai (the token used for R2 publish is usually enough).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=cdn-publish-common.sh
source "${ROOT}/scripts/cdn-publish-common.sh"

if [[ -z "${CLOUDFLARE_API_TOKEN:-}" ]]; then
  echo "CLOUDFLARE_API_TOKEN is required." >&2
  exit 1
fi

ZONE_ID=$(cdn_zone_id)
API="https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/rulesets"
AUTH=(-H "Authorization: Bearer ${CLOUDFLARE_API_TOKEN}" -H "Content-Type: application/json")

# Find or create a phase ruleset, then PUT its rules (full replace).
upsert_phase_rules() {
  local phase="$1"
  local rules_json="$2"
  local ruleset_id
  ruleset_id=$(curl -fsS "${AUTH[@]}" "${API}" | python3 -c '
import json, sys
phase = sys.argv[1]
for r in json.load(sys.stdin)["result"]:
    if r.get("phase") == phase and r.get("kind") == "zone":
        print(r["id"])
        break
' "${phase}")

  if [[ -z "${ruleset_id}" ]]; then
    echo "Creating ${phase} ruleset..." >&2
    payload=$(RULES_JSON="${rules_json}" PHASE="${phase}" python3 -c '
import json, os
print(json.dumps({
  "name": "default",
  "kind": "zone",
  "phase": os.environ["PHASE"],
  "rules": json.loads(os.environ["RULES_JSON"]),
}))
')
    curl -fsS "${AUTH[@]}" -X POST "${API}" --data "${payload}" >/dev/null
  else
    echo "Updating ${phase} ruleset ${ruleset_id}..." >&2
    payload=$(RULES_JSON="${rules_json}" python3 -c '
import json, os
print(json.dumps({"rules": json.loads(os.environ["RULES_JSON"])}))
')
    curl -fsS "${AUTH[@]}" -X PUT "${API}/${ruleset_id}" --data "${payload}" >/dev/null
  fi
  echo "  ok: ${phase}" >&2
}

REQUEST_TRANSFORM_RULES=$(python3 <<'PY'
import json
hosts = '(http.host in {"staging.moonshine.ai" "moonshine.ai" "www.moonshine.ai"})'
print(json.dumps([
  {
    "ref": "webrtc_root_index",
    "expression": '(http.host eq "webrtc.moonshine.ai" and http.request.uri.path eq "/")',
    "description": "Stand in for the GCS mainPageSuffix that served index.html at /",
    "action": "rewrite",
    "action_parameters": {"uri": {"path": {"value": "/index.html"}}},
    "enabled": True,
  },
  {
    "ref": "www_root_index",
    "expression": f'{hosts} and http.request.uri.path eq "/"',
    "description": "Serve examples/web index.html at site root",
    "action": "rewrite",
    "action_parameters": {"uri": {"path": {"value": "/index.html"}}},
    "enabled": True,
  },
  {
    "ref": "www_dir_index",
    "expression": f'{hosts} and ends_with(http.request.uri.path, "/") and http.request.uri.path ne "/"',
    "description": "Append index.html for directory URLs on the static site",
    "action": "rewrite",
    "action_parameters": {
      "uri": {"path": {"expression": 'concat(http.request.uri.path, "index.html")'}}
    },
    "enabled": True,
  },
]))
PY
)

RESPONSE_HEADER_RULES=$(python3 <<'PY'
import json
hosts = '(http.host in {"staging.moonshine.ai" "moonshine.ai" "www.moonshine.ai"})'
print(json.dumps([
  {
    "ref": "www_cross_origin_isolation",
    "expression": hosts,
    "description": "COOP/COEP so SharedArrayBuffer works for WASM demos",
    "action": "rewrite",
    "action_parameters": {
      "headers": {
        "Cross-Origin-Opener-Policy": {"operation": "set", "value": "same-origin"},
        "Cross-Origin-Embedder-Policy": {"operation": "set", "value": "require-corp"},
      }
    },
    "enabled": True,
  },
]))
PY
)

upsert_phase_rules "http_request_transform" "${REQUEST_TRANSFORM_RULES}"
upsert_phase_rules "http_response_headers_transform" "${RESPONSE_HEADER_RULES}"

echo "Transform rules applied for staging/production www hosts." >&2
