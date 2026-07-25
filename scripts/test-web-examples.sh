#! /bin/bash -e
set -o pipefail

# Runs the browser integration test for the web examples (examples/web/): starts
# examples/web/serve.mjs, opens each example page in headless Chrome via
# puppeteer, and drives it through its file-based I/O against the locally-built
# /wasm/dist binding and the in-repo model assets (fully offline).
#
# Requires the binding to be built (scripts/build-wasm.sh) and puppeteer-core +
# a Chrome/Chromium binary to be available. Pass `build` to build first.
#
# Environment:
#   PUPPETEER_EXECUTABLE_PATH / CHROME_PATH - override the browser binary.

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname "${SCRIPTS_DIR}")
WASM_DIR="${REPO_ROOT_DIR}/wasm"
DIST_DIR="${WASM_DIR}/dist"

DO_BUILD=""
for arg in "$@"; do
    case "$arg" in
        build) DO_BUILD=1 ;;
        *) echo "Unknown argument: '$arg'" >&2; exit 1 ;;
    esac
done

if [ -n "${DO_BUILD}" ]; then
    "${SCRIPTS_DIR}/build-wasm.sh"
fi

if [ ! -f "${DIST_DIR}/moonshine.mjs" ] || [ ! -f "${DIST_DIR}/index.js" ]; then
    echo "[test-web-examples] built artifacts not found in ${DIST_DIR}." >&2
    echo "[test-web-examples] Run: scripts/test-web-examples.sh build" >&2
    exit 1
fi

echo "[test-web-examples] running browser integration test..."
(
    cd "${WASM_DIR}"
    if [ ! -d node_modules ]; then npm install; fi
    MOONSHINE_BROWSER_TESTS=1 node --test tests/web-examples.integration.test.mjs
)

echo "[test-web-examples] done."
