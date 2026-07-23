#! /bin/bash -e
set -o pipefail

# Builds the Moonshine WebAssembly binding:
#   1. ensures the vendored ORT-wasm static library exists
#      (scripts/build-ort-wasm.sh),
#   2. configures + builds the embind module with Emscripten (emcmake),
#      emitting moonshine.mjs + moonshine.wasm,
#   3. compiles the idiomatic TypeScript layer (tsc) into wasm/dist,
#   4. copies the generated wasm artifacts alongside the compiled JS.
#
# We deliberately do NOT upload the library to download.moonshine.ai (that CDN
# hosts *model assets*, which the binding downloads at runtime). Distribution of
# the library itself is via npm and (optionally) a tarball on the GitHub
# release, mirroring scripts/publish-binary.sh.
#
# Arguments (order-independent):
#   publish-npm    - run `npm publish` from wasm/ after a successful build.
#   upload         - attach a wasm/dist tarball to the GitHub release v<VERSION>.
#   single-thread  - build the SIMD-only (no pthreads) variant for pages that
#                    can't be cross-origin isolated. Default is SIMD + threads.
#   skip-ort       - assume the ORT-wasm archive is already vendored.
#   skip-core      - reuse an existing wasm build dir (skip emcmake/cmake build).

VERSION=0.0.71
REPO="moonshine-ai/moonshine"

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname "${SCRIPTS_DIR}")
CORE_DIR="${REPO_ROOT_DIR}/core"
WASM_DIR="${REPO_ROOT_DIR}/wasm"
BUILD_DIR="${WASM_DIR}/build"
DIST_DIR="${WASM_DIR}/dist"

PUBLISH_NPM=""
DO_UPLOAD=""
SINGLE_THREAD=""
SKIP_ORT=""
SKIP_CORE=""
for arg in "$@"; do
    case "$arg" in
        publish-npm|--publish-npm) PUBLISH_NPM=1 ;;
        upload) DO_UPLOAD=1 ;;
        single-thread|singlethread) SINGLE_THREAD=1 ;;
        skip-ort|--skip-ort) SKIP_ORT=1 ;;
        skip-core|--skip-core) SKIP_CORE=1 ;;
        *) echo "Unknown argument: '$arg'" >&2; exit 1 ;;
    esac
done

# --- Toolchain: ensure emcc/emcmake are on PATH ----------------------------
if ! command -v emcmake >/dev/null 2>&1; then
    if [ -f "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" ]; then
        # shellcheck disable=SC1090
        source "${EMSDK:-$HOME/emsdk}/emsdk_env.sh" >/dev/null 2>&1
    fi
fi
if ! command -v emcmake >/dev/null 2>&1; then
    echo "[build-wasm] emcmake not found. Install/activate emsdk (4.0.8) first:" >&2
    echo "  git clone https://github.com/emscripten-core/emsdk ~/emsdk && \\" >&2
    echo "  ~/emsdk/emsdk install 4.0.8 && ~/emsdk/emsdk activate 4.0.8" >&2
    exit 1
fi

CMAKE_WASM_FLAGS=()
ORT_ARGS=()
if [ -n "${SINGLE_THREAD}" ]; then
    CMAKE_WASM_FLAGS+=(-DMOONSHINE_WASM_SINGLE_THREAD=ON)
    ORT_ARGS+=(single-thread)
fi

# --- Step 1: vendored ORT-wasm static library ------------------------------
if [ -z "${SKIP_ORT}" ]; then
    echo "[build-wasm] ensuring ORT-wasm static library is vendored..."
    "${SCRIPTS_DIR}/build-ort-wasm.sh" "${ORT_ARGS[@]}"
fi

# --- Step 2: build the embind module with Emscripten -----------------------
if [ -z "${SKIP_CORE}" ]; then
    echo "[build-wasm] configuring + building the wasm module..."
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"
    (
        cd "${BUILD_DIR}"
        emcmake cmake "${CORE_DIR}" -DCMAKE_BUILD_TYPE=Release "${CMAKE_WASM_FLAGS[@]}"
        cmake --build . --target moonshine_wasm -j"$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
    )
fi

# The moonshine_wasm target (OUTPUT_NAME=moonshine) emits into wasm/build.
MJS=$(find "${BUILD_DIR}" -name 'moonshine.mjs' -print -quit)
WASM=$(find "${BUILD_DIR}" -name 'moonshine.wasm' -print -quit)
if [ -z "${MJS}" ] || [ -z "${WASM}" ]; then
    echo "[build-wasm] ERROR: could not find generated moonshine.mjs / moonshine.wasm under ${BUILD_DIR}" >&2
    exit 1
fi

# --- Step 3: compile the TypeScript layer ----------------------------------
echo "[build-wasm] compiling TypeScript layer..."
(
    cd "${WASM_DIR}"
    if [ ! -d node_modules ]; then npm install; fi
    npm run build:ts
)

# --- Step 4: place generated wasm artifacts next to the compiled JS --------
mkdir -p "${DIST_DIR}"
cp "${MJS}" "${DIST_DIR}/moonshine.mjs"
cp "${WASM}" "${DIST_DIR}/moonshine.wasm"
# pthreads builds may emit a worker helper; copy it if present.
WORKER=$(find "${BUILD_DIR}" -name 'moonshine.worker.js' -print -quit || true)
if [ -n "${WORKER}" ]; then
    cp "${WORKER}" "${DIST_DIR}/moonshine.worker.js"
fi
echo "[build-wasm] artifacts in ${DIST_DIR}:"
ls -la "${DIST_DIR}"

# --- Optional: publish to npm ----------------------------------------------
if [ -n "${PUBLISH_NPM}" ]; then
    echo "[build-wasm] publishing @moonshine-ai/moonshine-wasm@${VERSION} to npm..."
    (cd "${WASM_DIR}" && npm publish --access public)
fi

# --- Optional: attach a tarball to the GitHub release ----------------------
if [ -n "${DO_UPLOAD}" ]; then
    TMP_DIR=$(mktemp -d)
    TAR_NAME="moonshine-voice-wasm.tar.gz"
    tar -czf "${TMP_DIR}/${TAR_NAME}" -C "${WASM_DIR}" dist README.md
    if ! gh release view "v${VERSION}" -R "${REPO}" >/dev/null 2>&1; then
        gh release create "v${VERSION}" -R "${REPO}" --title "v${VERSION}" --notes "Release v${VERSION}"
    fi
    echo "[build-wasm] uploading ${TAR_NAME} to release v${VERSION}..."
    gh release upload "v${VERSION}" "${TMP_DIR}/${TAR_NAME}" -R "${REPO}" --clobber
    echo "[build-wasm] uploaded ${TAR_NAME}."
fi

echo "[build-wasm] done."
