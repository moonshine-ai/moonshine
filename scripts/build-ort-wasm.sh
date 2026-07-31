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
# Exception handling is the single biggest performance lever in this build, so
# it is worth spelling out. ORT's default wasm configuration turns on
# `-s DISABLE_EXCEPTION_CATCHING=0`, i.e. Emscripten's *JavaScript* exception
# handling. Under that model every C++ call that needs an unwind landing pad is
# lowered to an `invoke_*` trampoline that leaves wasm, runs a JS function, and
# re-enters wasm - three boundary crossings per call.
#
# That is ruinous for kernels that call a small throwing helper in an inner
# loop. The worst offender for us is ConvInteger (the quantized convolutions in
# the Kokoro TTS model): for 1-D kernels it falls back to ORT's generic im2col,
# which calls math::NextPosition() once per output element, and NextPosition
# contains an ORT_ENFORCE and so can throw. Profiling browser TTS showed ~66% of
# total synthesis time spent in those trampolines, and building with the flags
# below took Kokoro from a real-time factor of 2.6 to 0.59 (4.4x) with no change
# in output. See scripts/bench-wasm-tts.mjs / scripts/profile-wasm-tts.mjs.
#
# Disabling exception *catching* while keeping it at the API boundary is the
# same configuration Microsoft ships for the onnxruntime-web packages: ORT
# internals lose their landing pads, while errors surfaced through the ORT API
# are still reported to callers rather than aborting.
#
# Size is the other lever. This is a *minimal* build restricted to the
# operators our models actually use
# (core/third-party/onnxruntime/moonshine-required-operators.config, produced by
# scripts/generate-ort-op-config.py). That drops roughly two thirds of ORT's
# code from the linked .wasm. Two consequences follow from --minimal_build:
#
#   - The runtime can only load ORT-format models. A .onnx fails at session
#     creation with a clear error, which is why everything we ship or download
#     is converted (see scripts/convert-models-to-ort.py).
#   - Any operator missing from the config fails at session creation too, so
#     the config has to be regenerated whenever a model changes. CI enforces
#     this; see scripts/check-ort-op-config.sh.
#
# `extended custom_ops` keeps the runtime-level optimizers a minimal build can
# still apply, and keeps custom-op registration available for ZipVoice.
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
OP_CONFIG="${REPO_ROOT_DIR}/core/third-party/onnxruntime/moonshine-required-operators.config"

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
    echo "[build-ort-wasm] NOTE: archives vendored before the exception-handling"
    echo "[build-ort-wasm] change (see the header comment) are ~4x slower at TTS;"
    echo "[build-ort-wasm] rebuild with 'force' if yours predates it."
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

if [ ! -f "${OP_CONFIG}" ]; then
    echo "[build-ort-wasm] ERROR: operator config not found at ${OP_CONFIG}" >&2
    echo "[build-ort-wasm] Generate it with scripts/generate-ort-op-config.py" >&2
    exit 1
fi

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

# --disable_ml_ops drops the classical-ML (ai.onnx.ml) kernels, which no
# Moonshine model uses.
#
# --compile_no_warning_as_error works around ORT 1.23: in a minimal build
# `min_ort_version_with_shape_inference` in core/session/custom_ops.cc is left
# unused, and the build otherwise fails on -Werror,-Wunused-const-variable.
# Drop this once the upstream fix lands in a version we pin to.
#
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
            --disable_wasm_exception_catching \
            --enable_wasm_api_exception_catching \
            --minimal_build extended custom_ops \
            --include_ops_by_config "${OP_CONFIG}" \
            --disable_ml_ops \
            --skip_tests \
            --parallel \
            --compile_no_warning_as_error \
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
