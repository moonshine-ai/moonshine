#!/usr/bin/env bash
# Publish @moonshine-ai/moonshine-wasm to npm.
#
# Kept separate from scripts/build-all-platforms.sh so npm's interactive login
# (or an expired token) cannot stall a multi-hour release mid-flight. Build the
# package as part of the release first:
#
#   scripts/build-all-platforms.sh publish   # builds wasm + attaches the tarball
#   scripts/publish-wasm-npm.sh              # then push to npm when ready
#
# Prerequisites: language-bindings/wasm/dist must already contain the built
# package (index.js + moonshine.wasm). Re-build with scripts/build-wasm.sh if
# needed. Requires an authenticated npm CLI (`npm whoami`).
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
WASM_DIR="${REPO_ROOT_DIR}/language-bindings/wasm"
DIST_DIR="${WASM_DIR}/dist"

if [[ ! -f "${DIST_DIR}/index.js" || ! -f "${DIST_DIR}/moonshine.wasm" ]]; then
	echo "Missing ${DIST_DIR}/index.js or moonshine.wasm." >&2
	echo "Build first: scripts/build-wasm.sh" >&2
	exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
	echo "npm is not installed." >&2
	exit 1
fi

if ! npm whoami >/dev/null 2>&1; then
	echo "npm is not authenticated. Run: npm login" >&2
	exit 1
fi

VERSION="$(node -p "require('${WASM_DIR}/package.json').version")"
echo "Publishing @moonshine-ai/moonshine-wasm@${VERSION} to npm as $(npm whoami)..."
(cd "${WASM_DIR}" && npm publish --access public)
echo "Published. CDN import: https://cdn.jsdelivr.net/npm/@moonshine-ai/moonshine-wasm@${VERSION}/dist/index.js"
