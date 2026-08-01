#!/usr/bin/env bash
# Build and vendor the minimal ONNX Runtime shared libraries for Android.
#
# Android ships ORT as a .so, so unlike the static-linked platforms nothing is
# dropped at app link time: whatever is in the library is in the install. That
# makes operator reduction worth more here than anywhere else. The build is
# restricted to core/third-party/onnxruntime/moonshine-required-operators.config
# (see scripts/generate-ort-op-config.py), which drops roughly two thirds of
# ORT's code.
#
# Two consequences follow from --minimal_build, and both fail at session
# creation rather than at build time:
#
#   - Only ORT-format models load. A .onnx is rejected with a clear error; see
#     docs/ort-only-models.md.
#   - An operator missing from the config fails too, so the config must be
#     regenerated whenever a model changes. CI enforces this; see the
#     check-ort-op-config test.
#
# NNAPI is left out, which is a measurement rather than an oversight. It costs
# 0.55 MB of the 6.5 MB library, and against the models we ship it takes the
# graph in 1 to 7 node pieces — 220 separate partitions for a Piper voice — so
# the boundary crossings cost more than the acceleration saves. The cause is
# that our .ort files are converted at full optimization, which fuses regions
# into com.microsoft ops no compiling provider recognises;
# docs/execution-providers.md has the numbers and
# scripts/check-ep-partitioning.py reproduces them. Pass 'with-nnapi' to build
# it back in, which is worth doing if that ever changes.
#
# Usage:
#   scripts/build-ort-android.sh [force] [with-nnapi] [abi ...]
#
#   force       rebuild and re-vendor even if a library is already in place
#   with-nnapi  include the NNAPI execution provider (see above)
#   abi         one or more of arm64-v8a, armeabi-v7a, x86_64 (default: all)
#
# Environment:
#   ANDROID_SDK_ROOT / ANDROID_HOME  Android SDK (default: ~/Library/Android/sdk)
#   ANDROID_NDK_VERSION              NDK to build with (default: matches build.gradle.kts)
#   ORT_ANDROID_CONFIG               Release (default) or MinSizeRel
#   MOONSHINE_ORT_ROOT               where the ORT checkout and build trees live
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/ort-build-common.sh
source "${REPO_ROOT_DIR}/scripts/ort-build-common.sh"

DEST_ROOT="${REPO_ROOT_DIR}/core/third-party/onnxruntime/lib/android"

# minSdk in build.gradle.kts. Building against a higher API than the app
# supports would leave the library referencing symbols absent on older devices.
ANDROID_API="${ANDROID_API:-26}"
ANDROID_NDK_VERSION="${ANDROID_NDK_VERSION:-28.2.13676358}"
ANDROID_SDK="${ANDROID_SDK_ROOT:-${ANDROID_HOME:-${HOME}/Library/Android/sdk}}"
ANDROID_NDK="${ANDROID_SDK}/ndk/${ANDROID_NDK_VERSION}"

# Release matches the wasm build and the prebuilt libraries this replaces.
# MinSizeRel builds arm64 at 5.3 MB against Release's 6.0 MB, so it is worth
# 0.7 MB — 3% of the arm64 install. It buys that by compiling -Os, which is a
# speed trade we cannot price without a physical device (an emulator's timings
# say nothing), and TTS is the tightest real-time budget we have. So it stays
# opt-in: measure it on a device before shipping it.
ORT_ANDROID_CONFIG="${ORT_ANDROID_CONFIG:-Release}"

FORCE=""
USE_NNAPI=""
ABIS=()
for arg in "$@"; do
    case "$arg" in
        force|--force) FORCE=1 ;;
        with-nnapi|--with-nnapi) USE_NNAPI=1 ;;
        no-nnapi|--no-nnapi) USE_NNAPI="" ;;
        arm64-v8a|armeabi-v7a|x86_64) ABIS+=("$arg") ;;
        *) echo "Unknown argument: '$arg'" >&2; exit 1 ;;
    esac
done
if [ "${#ABIS[@]}" -eq 0 ]; then
    ABIS=(arm64-v8a armeabi-v7a x86_64)
fi

# The vendored tree calls arm64-v8a "arm64"; the other two match the ABI name.
dest_dir_for_abi() {
    case "$1" in
        arm64-v8a) echo "${DEST_ROOT}/arm64" ;;
        *) echo "${DEST_ROOT}/$1" ;;
    esac
}

if [ ! -d "${ANDROID_NDK}" ]; then
    echo "ERROR: NDK ${ANDROID_NDK_VERSION} not found at ${ANDROID_NDK}" >&2
    echo "Install it, or set ANDROID_NDK_VERSION to one you have." >&2
    exit 1
fi

ort_require_op_config

echo "[build-ort-android] ORT ${ORT_VERSION}, NDK ${ANDROID_NDK_VERSION}, API ${ANDROID_API}"
echo "[build-ort-android] config: ${ORT_ANDROID_CONFIG}, NNAPI: ${USE_NNAPI:+on}${USE_NNAPI:-off (see the header comment)}"
echo "[build-ort-android] ABIs: ${ABIS[*]}"

needs_build=""
for abi in "${ABIS[@]}"; do
    if [ ! -f "$(dest_dir_for_abi "${abi}")/libonnxruntime.so" ]; then
        needs_build=1
    fi
done
if [ -z "${FORCE}" ] && [ -z "${needs_build}" ]; then
    echo "[build-ort-android] every requested ABI is already vendored; pass 'force' to rebuild."
    exit 0
fi

ort_prepare_checkout

# The NDK ships the only strip that understands its own output. ORT's build
# leaves debug info in, and an unstripped arm64 library is around 675 MB
# against 18 MB stripped; Gradle would strip it when packaging, but the repo
# would still carry the whole thing through Git LFS on every clone.
HOST_TAG="darwin-x86_64"
case "$(uname -s)" in
    Linux) HOST_TAG="linux-x86_64" ;;
esac
STRIP="${ANDROID_NDK}/toolchains/llvm/prebuilt/${HOST_TAG}/bin/llvm-strip"
if [ ! -x "${STRIP}" ]; then
    echo "ERROR: llvm-strip not found at ${STRIP}" >&2
    exit 1
fi

build_abi() {
    local abi="$1"
    local build_dir="${MOONSHINE_ORT_ROOT}/build-android-${abi}"
    local dest; dest="$(dest_dir_for_abi "${abi}")"
    local nnapi=()
    [ -n "${USE_NNAPI}" ] && nnapi=(--use_nnapi)

    # KleidiAI is aarch64-only, but ORT 1.23 enables it whenever the *host*
    # machine is arm64, without checking the cross-compilation target (see the
    # last clause of the onnxruntime_USE_KLEIDIAI logic in
    # tools/ci_build/build.py). On an Apple Silicon machine that turns on the
    # KleidiAI code paths for armeabi-v7a and x86_64 while the sources
    # themselves are excluded, and the link fails on undefined
    # ArmKleidiAI::Mlas* symbols. Turn it off for every target that is not
    # arm64, where it would do nothing anyway.
    local kleidiai=()
    case "${abi}" in
        arm64-v8a) ;;
        *) kleidiai=(--no_kleidiai) ;;
    esac

    # A forced rebuild starts from an empty build directory. ORT's build.py
    # only *adds* a -D when a feature is enabled, so turning one off leaves the
    # previous ON in CMakeCache.txt and the flag silently does nothing: passing
    # --no_kleidiai to a directory built with it enabled still fails on
    # undefined ArmKleidiAI symbols. This, rather than anything in the source
    # tree, is where stale state actually comes from.
    if [ -n "${FORCE}" ] && [ -d "${build_dir}" ]; then
        rm -rf "${build_dir}"
    fi

    echo "[build-ort-android] building ${abi} -> ${build_dir}"
    local minimal_flags=()
    while IFS= read -r flag; do minimal_flags+=("${flag}"); done < <(ort_minimal_flags)
    (
        cd "${ORT_SRC}"
        ./build.sh \
            --build_dir "${build_dir}" \
            --config "${ORT_ANDROID_CONFIG}" \
            --android \
            --android_sdk_path "${ANDROID_SDK}" \
            --android_ndk_path "${ANDROID_NDK}" \
            --android_abi "${abi}" \
            --android_api "${ANDROID_API}" \
            --build_shared_lib \
            "${minimal_flags[@]}" \
            ${nnapi[@]+"${nnapi[@]}"} \
            ${kleidiai[@]+"${kleidiai[@]}"} \
            --skip_tests \
            --parallel
    )

    local built="${build_dir}/${ORT_ANDROID_CONFIG}/libonnxruntime.so"
    if [ ! -f "${built}" ]; then
        echo "ERROR: expected library not produced at ${built}" >&2
        exit 1
    fi
    mkdir -p "${dest}"
    "${STRIP}" --strip-unneeded "${built}" -o "${dest}/libonnxruntime.so"
    echo "[build-ort-android] vendored ${abi}:"
    ort_report_size "unstripped" "${built}"
    ort_report_size "vendored" "${dest}/libonnxruntime.so"
}

for abi in "${ABIS[@]}"; do
    build_abi "${abi}"
done

echo "[build-ort-android] done. Measure the result with:"
echo "  scripts/build-android.sh local && scripts/measure-mobile-size.sh android"
