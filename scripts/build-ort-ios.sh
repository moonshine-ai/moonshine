#!/usr/bin/env bash
# Build and vendor the minimal ONNX Runtime static libraries for iOS.
#
# Restricted to core/third-party/onnxruntime/moonshine-required-operators.config
# (see scripts/generate-ort-op-config.py), the same config every other platform
# builds against.
#
# Expect a smaller win here than on Android. iOS links ORT statically, so the
# app linker already drops object files nothing references; operator reduction
# removes code that dead-stripping could not prove unreachable, which is a
# subset of what it removes from a shared library. Judge it with
# scripts/measure-mobile-size.sh ios, which links a real binary — the .a on
# disk is a poor proxy for what an app installs.
#
# The two consequences of --minimal_build apply here as on Android: only
# ORT-format models load (docs/ort-only-models.md), and an operator missing
# from the config fails at session creation, so the config must be regenerated
# whenever a model changes.
#
# CoreML does not build here, and that is an upstream limitation rather than a
# choice. ORT 1.23's CoreML provider calls Graph::GetModel() unconditionally
# (coreml_execution_provider.cc, in GetCapability), but that method is compiled
# out of any minimal build — include/onnxruntime/core/graph/graph.h guards it
# with `#if !defined(ORT_MINIMAL_BUILD)`, and `extended` still defines that.
# The build fails with "no member named 'GetModel' in 'onnxruntime::Graph'".
#
# So a minimal iOS build means no CoreML execution provider. Passing
# 'with-coreml' attempts it anyway, which is useful for checking whether a
# later ORT has fixed this; expect a compile error until one does. NNAPI has
# no such problem, so the Android build keeps it.
#
# Usage:
#   scripts/build-ort-ios.sh [force] [with-coreml] [device|simulator]
#
#   force        rebuild and re-vendor even if a library is already in place
#   with-coreml  attempt to include CoreML (see above; expected to fail)
#   device      build only the arm64 device slice
#   simulator   build only the fat x86_64+arm64 simulator slice
#
# Environment:
#   ORT_IOS_CONFIG      Release (default) or MinSizeRel
#   MOONSHINE_ORT_ROOT  where the ORT checkout and build trees live
set -euo pipefail

REPO_ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/ort-build-common.sh
source "${REPO_ROOT_DIR}/scripts/ort-build-common.sh"

DEST_ROOT="${REPO_ROOT_DIR}/core/third-party/onnxruntime/lib/ios"

# Matches IOS_VERSION in scripts/build-swift.sh. A library built for a newer
# minimum than the framework claims to support links, then fails on a device
# running the older OS.
IOS_DEPLOY_TARGET="${IOS_DEPLOY_TARGET:-15.1}"

# Release for the same reason as Android, where the two were compared:
# MinSizeRel saved 3% of the library and cost an unmeasured amount of speed.
# See the note on ORT_ANDROID_CONFIG in scripts/build-ort-android.sh.
ORT_IOS_CONFIG="${ORT_IOS_CONFIG:-Release}"

FORCE=""
USE_COREML=""
SLICES=()
for arg in "$@"; do
    case "$arg" in
        force|--force) FORCE=1 ;;
        with-coreml|--with-coreml) USE_COREML=1 ;;
        no-coreml|--no-coreml) USE_COREML="" ;;
        device|simulator) SLICES+=("$arg") ;;
        *) echo "Unknown argument: '$arg'" >&2; exit 1 ;;
    esac
done
if [ "${#SLICES[@]}" -eq 0 ]; then
    SLICES=(device simulator)
fi

# find-ort-library-path.cmake selects by sysroot: iphonesimulator picks
# lib/ios/simulator, anything else iOS picks lib/ios/arm64.
dest_dir_for_slice() {
    case "$1" in
        device) echo "${DEST_ROOT}/arm64" ;;
        simulator) echo "${DEST_ROOT}/simulator" ;;
    esac
}

if ! xcrun --sdk iphoneos --show-sdk-path >/dev/null 2>&1; then
    echo "ERROR: no iphoneos SDK; install Xcode and run xcode-select." >&2
    exit 1
fi

ort_require_op_config

echo "[build-ort-ios] ORT ${ORT_VERSION}, deploy target ${IOS_DEPLOY_TARGET}"
echo "[build-ort-ios] config: ${ORT_IOS_CONFIG}, CoreML: ${USE_COREML:+attempted}${USE_COREML:-off (not buildable in a minimal ORT)}"
echo "[build-ort-ios] slices: ${SLICES[*]}"

needs_build=""
for slice in "${SLICES[@]}"; do
    if [ ! -f "$(dest_dir_for_slice "${slice}")/libonnxruntime.a" ]; then
        needs_build=1
    fi
done
if [ -z "${FORCE}" ] && [ -z "${needs_build}" ]; then
    echo "[build-ort-ios] every requested slice is already vendored; pass 'force' to rebuild."
    exit 0
fi

ort_prepare_checkout

# The merged static library ORT produced, for one sysroot and architecture.
#
# ORT compiles one static library per component. --build_apple_framework has it
# merge them into a static framework, whose binary is exactly the single
# archive we want, so we take that rather than re-merging the parts ourselves.
# It is also what the prebuilt pod archive this replaces contained.
framework_binary() {
    local build_config_dir="$1" sysroot="$2"
    local bin="${build_config_dir}/${ORT_IOS_CONFIG}-${sysroot}/static_framework/onnxruntime.framework/onnxruntime"
    if [ ! -f "${bin}" ]; then
        # Layout has moved between ORT versions; look before giving up.
        bin="$(find "${build_config_dir}" -path '*static_framework*' \
            -name 'onnxruntime' -type f 2>/dev/null | head -1)"
    fi
    if [ -z "${bin}" ] || [ ! -f "${bin}" ]; then
        echo "ERROR: no static framework binary under ${build_config_dir}" >&2
        exit 1
    fi
    echo "${bin}"
}

# Builds one sysroot/architecture pair and echoes the resulting archive.
#
# One architecture at a time is not a simplification: --osx_arch takes a single
# choice from arm64, arm64e and x86_64, so a fat library has to be built twice
# and combined with lipo.
build_one() {
    local sysroot="$1" arch="$2"
    local build_dir="${MOONSHINE_ORT_ROOT}/build-ios-${sysroot}-${arch}"

    local coreml=()
    [ -n "${USE_COREML}" ] && coreml=(--use_coreml)

    # KleidiAI is aarch64-only, but ORT 1.23 turns it on whenever the *host* is
    # arm64 without checking the target, so an x86_64 simulator build on an
    # Apple Silicon machine fails to link on undefined ArmKleidiAI symbols.
    local kleidiai=()
    case "${arch}" in
        arm64|arm64e) ;;
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

    echo "[build-ort-ios] building ${sysroot}/${arch} -> ${build_dir}" >&2
    local minimal_flags=()
    while IFS= read -r flag; do minimal_flags+=("${flag}"); done < <(ort_minimal_flags)
    # --cmake_generator Xcode is not optional: ORT refuses an iOS build with
    # any other generator. --build_apple_framework gives us the merged static
    # library; its test step needs a simulator but is skipped by --skip_tests.
    #
    # CMAKE_POLICY_VERSION_MINIMUM=3.5 is for psimd, fetched transitively by
    # ORT 1.23's Apple build. Its CMakeLists declares a minimum below 3.5,
    # which CMake 4 refuses outright, so configuration fails before any of our
    # code is reached. Harmless on CMake 3.x. Remove when ORT stops pulling in
    # a dependency that old.
    (
        cd "${ORT_SRC}"
        ./build.sh \
            --build_dir "${build_dir}" \
            --config "${ORT_IOS_CONFIG}" \
            --ios \
            --cmake_generator Xcode \
            --build_apple_framework \
            --apple_sysroot "${sysroot}" \
            --osx_arch "${arch}" \
            --apple_deploy_target "${IOS_DEPLOY_TARGET}" \
            "${minimal_flags[@]}" \
            ${coreml[@]+"${coreml[@]}"} \
            ${kleidiai[@]+"${kleidiai[@]}"} \
            --skip_tests \
            --parallel \
            --cmake_extra_defines CMAKE_POLICY_VERSION_MINIMUM=3.5
    ) >&2

    framework_binary "${build_dir}/${ORT_IOS_CONFIG}" "${sysroot}"
}

build_slice() {
    local slice="$1"
    local dest; dest="$(dest_dir_for_slice "${slice}")"
    mkdir -p "${dest}"

    if [ "${slice}" = "device" ]; then
        local built; built="$(build_one iphoneos arm64)"
        cp "${built}" "${dest}/libonnxruntime.a"
    else
        # Both simulator architectures, so the library works on an Intel Mac as
        # well as Apple Silicon; lipo reports the archive this replaces as
        # x86_64+arm64 and find-ort-library-path.cmake expects the same.
        local sim_arm64 sim_x86
        sim_arm64="$(build_one iphonesimulator arm64)"
        sim_x86="$(build_one iphonesimulator x86_64)"
        lipo -create "${sim_arm64}" "${sim_x86}" \
            -output "${dest}/libonnxruntime.a"
    fi

    echo "[build-ort-ios] vendored ${slice}:"
    ort_report_size "archive" "${dest}/libonnxruntime.a"
    lipo -info "${dest}/libonnxruntime.a" | sed 's/^/  /'
}

for slice in "${SLICES[@]}"; do
    build_slice "${slice}"
done

echo "[build-ort-ios] done. The archive size is not the shipped cost; measure with:"
echo "  scripts/build-swift.sh && scripts/measure-mobile-size.sh ios"
