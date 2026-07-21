#! /bin/bash -e
set -o pipefail

# Builds ONNX Runtime as a WebAssembly *static library*
# (libonnxruntime_webassembly.a) and vendors it under
#   core/third-party/onnxruntime/lib/wasm/
# so the Emscripten build of libmoonshine can link against it.
#
# Why we build from source: Microsoft does not publish a prebuilt ORT-wasm
# static library. The onnxruntime-web npm package only ships a fully-linked
# standalone .wasm module plus JS glue (no .a / .o / headers), which cannot be
# linked into another C++ program. The --build_wasm_static_lib option produces
# the archive we need, but "is not published by a pipeline" (ORT docs), so a
# manual build is required.
#
# The ORT C/C++ headers are already vendored (and shared with the native
# builds) under core/third-party/onnxruntime/include, so this script only needs
# to produce the .a. We pin ORT to the same version the native libraries use
# (see find-ort-library-path.cmake / ORT_VERSION below) so the archive is
# ABI-compatible with those headers.
#
# Toolchain: ORT self-manages emsdk via its cmake/external/emsdk submodule and
# installs/activates the version below. Build our own core with the SAME emsdk
# (see scripts/build-wasm.sh) to avoid link/ABI mismatch.
#
# The primary archive is built with BOTH WebAssembly SIMD and multithreading
# (pthreads) enabled -> libonnxruntime_webassembly.a. Consuming this requires
# the Emscripten build of libmoonshine to also use -pthread and the page to be
# cross-origin isolated (COOP/COEP) so SharedArrayBuffer is available.
#
# Arguments (order-independent):
#   single-thread - ALSO build a non-threaded SIMD fallback
#                   (libonnxruntime_webassembly_singlethread.a) for pages that
#                   can't set COOP/COEP. Off by default.
#   force         - rebuild even if the vendored archive already exists.
#
# Environment:
#   ORT_WASM_BUILD_DIR  Where to clone + build ORT (default: ~/moonshine-ort-wasm).
#   EMSDK_VERSION       emsdk version to pin (default: 4.0.8, ORT 1.23's pin).

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname "${SCRIPTS_DIR}")

# Keep this in lockstep with the native ORT version in
# core/third-party/onnxruntime/find-ort-library-path.cmake (e.g. 1.23.2).
ORT_VERSION="${ORT_VERSION:-1.23.2}"
EMSDK_VERSION="${EMSDK_VERSION:-4.0.8}"
ORT_WASM_BUILD_DIR="${ORT_WASM_BUILD_DIR:-${HOME}/moonshine-ort-wasm}"

DEST_DIR="${REPO_ROOT_DIR}/core/third-party/onnxruntime/lib/wasm"

BUILD_SINGLE_THREAD=""
FORCE=""
for arg in "$@"; do
    case "$arg" in
        single-thread|singlethread) BUILD_SINGLE_THREAD=1 ;;
        threads) : ;;  # accepted for backwards-compat; threaded is now the default
        force|--force) FORCE=1 ;;
        *) echo "Unknown argument: '$arg'" >&2; exit 1 ;;
    esac
done

echo "[build-ort-wasm] ORT ${ORT_VERSION}, emsdk ${EMSDK_VERSION}"
echo "[build-ort-wasm] build dir: ${ORT_WASM_BUILD_DIR}"
echo "[build-ort-wasm] dest dir:  ${DEST_DIR}"

mkdir -p "${DEST_DIR}"

if [ -z "${FORCE}" ] && [ -f "${DEST_DIR}/libonnxruntime_webassembly.a" ]; then
    echo "[build-ort-wasm] ${DEST_DIR}/libonnxruntime_webassembly.a already exists; pass 'force' to rebuild."
    if [ -z "${BUILD_SINGLE_THREAD}" ] || [ -f "${DEST_DIR}/libonnxruntime_webassembly_singlethread.a" ]; then
        exit 0
    fi
    echo "[build-ort-wasm] single-thread fallback missing; building it."
fi

# cmake_extra_defines applied to every variant:
#   - onnxruntime_USE_KLEIDIAI=OFF: on an Apple-Silicon host, cross-compiling
#     ORT 1.23 to wasm otherwise fails with "no member named 'HasArm_SME' in
#     'MLASCPUIDInfo'" because KleidiAI (an arm64-host feature) gets
#     auto-enabled for the wasm target.
#     See https://github.com/microsoft/onnxruntime/issues/26175
#   - onnxruntime_BUILD_UNIT_TESTS=OFF: --skip_tests only skips *running* tests;
#     the wasm build still tries to build (and file_package the testdata for)
#     the onnxruntime_webassembly_test target, which fails with
#     "testdata does not exist". We only need the static library, so drop the
#     test targets entirely (this also makes the build much faster).
EXTRA_DEFINES=(onnxruntime_USE_KLEIDIAI=OFF onnxruntime_BUILD_UNIT_TESTS=OFF)

ORT_SRC="${ORT_WASM_BUILD_DIR}/onnxruntime"
if [ ! -d "${ORT_SRC}/.git" ]; then
    mkdir -p "${ORT_WASM_BUILD_DIR}"
    echo "[build-ort-wasm] cloning onnxruntime v${ORT_VERSION} (recursive, for cmake/external/emsdk)..."
    git clone --recursive --depth 1 --branch "v${ORT_VERSION}" \
        https://github.com/microsoft/onnxruntime.git "${ORT_SRC}"
else
    echo "[build-ort-wasm] reusing existing ORT checkout at ${ORT_SRC}"
    git -C "${ORT_SRC}" submodule update --init --recursive
fi

# Builds one variant and vendors its static library to $2.
#   $1 = variant tag (simd|simd-threaded)
#   $2 = destination path for the resulting libonnxruntime_webassembly.a
#   remaining args = extra flags for build.sh
# Runs under `set -e -o pipefail`, so a failed build.sh aborts the whole script
# instead of silently vendoring a stale/absent archive.
build_variant() {
    local tag="$1"; local dest="$2"; shift 2
    local build_dir="${ORT_WASM_BUILD_DIR}/build-${tag}"
    echo "[build-ort-wasm] building variant '${tag}' -> ${build_dir}"
    (
        cd "${ORT_SRC}"
        ./build.sh \
            --build_dir "${build_dir}" \
            --config Release \
            --build_wasm_static_lib \
            --enable_wasm_simd \
            --skip_tests \
            --parallel \
            --emsdk_version "${EMSDK_VERSION}" \
            --cmake_extra_defines "${EXTRA_DEFINES[@]}" \
            "$@"
    )
    local lib="${build_dir}/Release/libonnxruntime_webassembly.a"
    if [ ! -f "${lib}" ]; then
        echo "[build-ort-wasm] ERROR: expected archive not found at ${lib}" >&2
        exit 1
    fi
    cp "${lib}" "${dest}"
    echo "[build-ort-wasm] vendored ${dest}"
}

# Primary variant: SIMD + multithreading.
build_variant simd-threaded \
    "${DEST_DIR}/libonnxruntime_webassembly.a" \
    --enable_wasm_threads

# Optional non-threaded SIMD fallback for pages without COOP/COEP.
if [ -n "${BUILD_SINGLE_THREAD}" ]; then
    build_variant simd \
        "${DEST_DIR}/libonnxruntime_webassembly_singlethread.a"
fi

echo "[build-ort-wasm] done."
