#! /bin/bash -e
set -o pipefail

# Runs the Moonshine WASM binding test suite:
#   1. a Node smoke test that loads the module and checks the low-level embind
#      surface (version, manifest helpers, class registration),
#   2. the TypeScript-layer unit tests (node --test).
#
# Assumes the binding has already been built (scripts/build-wasm.sh); pass
# `build` to build it first.
#
# Arguments (order-independent):
#   build   - run scripts/build-wasm.sh before testing.

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
    echo "[test-wasm] built artifacts not found in ${DIST_DIR}." >&2
    echo "[test-wasm] Run: scripts/test-wasm.sh build" >&2
    exit 1
fi

echo "[test-wasm] running Node test suite..."
(
    cd "${WASM_DIR}"
    if [ ! -d node_modules ]; then npm install; fi
    # Pass explicit test files (a glob the shell expands) rather than the bare
    # `tests` directory, which Node's runner otherwise tries to load as a module.
    node --test tests/*.test.mjs
)

echo "[test-wasm] done."
