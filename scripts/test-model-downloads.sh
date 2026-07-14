#!/bin/bash -e

# Download-and-run sampling tests for the native model catalog, plus the Swift
# and Android framework download tests.
#
# For a strategically sampled set of configurations (one per code path, biased
# toward the smallest model that exercises it), this script:
#   1. asks the C API (via moonshine-download-smoke) for the download manifest,
#   2. downloads every listed file into a fresh temp directory with curl,
#   3. loads the corresponding engine from that directory and runs one trivial
#      inference.
#
# If a required file is missing from a manifest, step 3 fails - which is the
# whole point: this verifies the catalog lists everything each model needs.
#
# After the native samples, it also drives the framework-level download tests
# that exercise the real AssetDownloader on each platform (they are opt-in and
# skip on a plain `swift test` / `connectedAndroidTest`):
#   - Swift: MOONSHINE_DOWNLOAD_TESTS=1 swift test --filter AssetDownloaderNetworkTests
#     (macOS, when `swift` and swift/Moonshine.xcframework are present).
#   - Android: connectedAndroidTest for AssetDownloaderTest with
#     moonshineDownloadTests=1 (only when a device/emulator is already
#     connected; this script never boots one - use scripts/test-android.sh for
#     emulator lifecycle).
#
# This test hits the network, so it is intentionally NOT part of the hermetic
# scripts/test-core.sh. Run it directly (it assumes a fast connection) or from
# scripts/build-all-platforms.sh / CI. Pass --all to cover more configurations
# (extra languages and embedding variants) for nightly runs. Pass
# --skip-frameworks to run only the native samples.
#
# Usage:
#   scripts/test-model-downloads.sh [--all] [--skip-frameworks]

RUN_ALL=0
SKIP_FRAMEWORKS=0
for arg in "$@"; do
    case "${arg}" in
        --all) RUN_ALL=1 ;;
        --skip-frameworks) SKIP_FRAMEWORKS=1 ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            echo "Unknown argument: ${arg}" >&2
            exit 2
            ;;
    esac
done

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR="$(dirname "${SCRIPTS_DIR}")"
BUILD_DIR="${REPO_ROOT_DIR}/core/build"

# Build just the smoke tool (and its library dependency) into the shared core
# build directory. This is cheap when the tree is already built.
mkdir -p "${BUILD_DIR}"
cmake -S "${REPO_ROOT_DIR}/core" -B "${BUILD_DIR}" >/dev/null
cmake --build "${BUILD_DIR}" --target moonshine-download-smoke

# The binary lands either as a plain executable or inside a macOS .app bundle
# (Swift / iOS-style builds set MACOSX_BUNDLE).
SMOKE_BIN=""
for candidate in \
    "${BUILD_DIR}/moonshine-download-smoke" \
    "${BUILD_DIR}/moonshine-download-smoke.app/Contents/MacOS/moonshine-download-smoke"; do
    if [[ -x "${candidate}" ]]; then
        SMOKE_BIN="${candidate}"
        break
    fi
done
if [[ -z "${SMOKE_BIN}" ]]; then
    echo "error: could not find moonshine-download-smoke binary under ${BUILD_DIR}" >&2
    exit 1
fi

# Help the dynamic loader find libonnxruntime for the shared library build.
ORT_LIB_ROOT="${REPO_ROOT_DIR}/core/third-party/onnxruntime/lib"
UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"
if [[ "${UNAME_S}" == "Darwin" ]]; then
    ORT_ARCH_DIR="${ORT_LIB_ROOT}/macos/${UNAME_M}"
    export DYLD_LIBRARY_PATH="${BUILD_DIR}:${ORT_ARCH_DIR}:${DYLD_LIBRARY_PATH:-}"
else
    if [[ "${UNAME_M}" == "aarch64" || "${UNAME_M}" == "arm64" ]]; then
        ORT_ARCH_DIR="${ORT_LIB_ROOT}/linux/aarch64"
    else
        ORT_ARCH_DIR="${ORT_LIB_ROOT}/linux/x86_64"
    fi
    export LD_LIBRARY_PATH="${BUILD_DIR}:${ORT_ARCH_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Run the smoke tool from the repo root so it can find test-assets for STT.
cd "${REPO_ROOT_DIR}"

PASS_COUNT=0
FAIL_COUNT=0

# download_and_run <label> <modality> <spec...>
# The spec is passed both to the manifest and run subcommands. The model root
# for run mode is the temp directory the manifest files are downloaded into.
download_and_run() {
    local label="$1"
    local modality="$2"
    shift 2
    local spec=("$@")

    echo ""
    echo "=== ${label} (${modality} ${spec[*]}) ==="

    local tmp_dir
    tmp_dir="$(mktemp -d)"
    # Ensure cleanup even if we return early below.
    trap 'rm -rf "${tmp_dir}"' RETURN

    local manifest
    if ! manifest="$("${SMOKE_BIN}" manifest "${modality}" "${spec[@]}")"; then
        echo "FAIL: manifest resolution failed for ${label}" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 0
    fi
    if [[ -z "${manifest}" ]]; then
        echo "FAIL: empty manifest for ${label}" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 0
    fi

    local file_count=0
    local download_failed=0
    # Each manifest line is "<url>\t<relative_path>".
    while IFS=$'\t' read -r url rel_path; do
        [[ -z "${url}" ]] && continue
        local dest="${tmp_dir}/${rel_path}"
        mkdir -p "$(dirname "${dest}")"
        if ! curl -fsSL "${url}" -o "${dest}"; then
            echo "FAIL: could not download ${url}" >&2
            download_failed=1
            break
        fi
        file_count=$((file_count + 1))
    done <<< "${manifest}"

    if [[ "${download_failed}" -ne 0 ]]; then
        FAIL_COUNT=$((FAIL_COUNT + 1))
        return 0
    fi
    echo "downloaded ${file_count} file(s)"

    if "${SMOKE_BIN}" run "${modality}" "${tmp_dir}" "${spec[@]}"; then
        echo "PASS: ${label}"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "FAIL: ${label} did not load/run from the downloaded files" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# --- Default sample: one config per code path (smallest representative). ------

# STT non-streaming English (arch 0): exercises the English attention-decoder
# extra plus the encoder/decoder/tokenizer trio.
download_and_run "STT tiny-en (non-streaming)" stt en 0

# STT streaming English (arch 2): the distinct streaming file list plus the
# streaming attention decoder.
download_and_run "STT tiny-streaming-en" stt en 2

# STT non-English (arch 0): the no-attention-extra branch and a non-English
# model directory.
download_and_run "STT tiny-ja" stt ja 0

# G2P: a small rule-based lexicon.
download_and_run "G2P de" g2p de

# TTS: a small Piper English voice plus the en_us G2P assets.
download_and_run "TTS en_us (Piper lessac)" tts en_us piper_en_US-lessac-medium

# Intent / embedding: q4 is the smallest published embedding variant.
download_and_run "Intent embeddinggemma-300m q4" intent embeddinggemma-300m q4

# --- Extended sample (nightly): more languages and embedding variants. --------

if [[ "${RUN_ALL}" -eq 1 ]]; then
    download_and_run "STT base-es" stt es 1
    download_and_run "STT base-zh" stt zh 1
    download_and_run "Intent embeddinggemma-300m q8" intent embeddinggemma-300m q8
    download_and_run "Intent embeddinggemma-300m fp16" intent embeddinggemma-300m fp16
fi

# --- Framework download tests (real AssetDownloader on each platform). --------

# Swift: run the opt-in network tests when a Swift toolchain and the local
# XCFramework the package links against are both available.
run_swift_framework_tests() {
    echo ""
    echo "=== Swift framework download tests (AssetDownloaderNetworkTests) ==="
    if [[ "${UNAME_S}" != "Darwin" ]]; then
        echo "SKIP: Swift download tests only run on macOS"
        return 0
    fi
    if ! command -v swift >/dev/null 2>&1; then
        echo "SKIP: swift not found in PATH"
        return 0
    fi
    if [[ ! -d "${REPO_ROOT_DIR}/swift/Moonshine.xcframework" ]]; then
        echo "SKIP: swift/Moonshine.xcframework missing (build it with scripts/build-swift.sh)"
        return 0
    fi
    if MOONSHINE_DOWNLOAD_TESTS=1 swift test \
        --package-path "${REPO_ROOT_DIR}/swift" \
        --filter 'AssetDownloaderNetworkTests'; then
        echo "PASS: Swift framework download tests"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "FAIL: Swift framework download tests" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# Android: run the opt-in instrumentation test, but only if a device/emulator is
# already connected (this script does not manage emulator lifecycle).
run_android_framework_tests() {
    echo ""
    echo "=== Android framework download tests (AssetDownloaderTest) ==="
    local sdk="${ANDROID_HOME:-${ANDROID_SDK_ROOT:-}}"
    if [[ -z "${sdk}" ]]; then
        for cand in "${HOME}/Library/Android/sdk" "${HOME}/Android/Sdk"; do
            [[ -d "${cand}" ]] && sdk="${cand}" && break
        done
    fi
    local adb="${sdk}/platform-tools/adb"
    if [[ -z "${sdk}" || ! -x "${adb}" ]]; then
        echo "SKIP: Android SDK / adb not found (set ANDROID_HOME)"
        return 0
    fi
    local device_count
    device_count="$("${adb}" devices | awk 'NR>1 && $2=="device"{c++} END{print c+0}')"
    if [[ "${device_count}" -eq 0 ]]; then
        echo "SKIP: no connected device/emulator (use scripts/test-android.sh to boot one)"
        return 0
    fi
    if (cd "${REPO_ROOT_DIR}" && ANDROID_HOME="${sdk}" ./gradlew \
        -Pandroid.useAndroidX=true \
        -Pandroid.testInstrumentationRunnerArguments.class=ai.moonshine.voice.AssetDownloaderTest \
        -Pandroid.testInstrumentationRunnerArguments.moonshineDownloadTests=1 \
        connectedAndroidTest --no-daemon); then
        echo "PASS: Android framework download tests"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "FAIL: Android framework download tests" >&2
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

if [[ "${SKIP_FRAMEWORKS}" -eq 0 ]]; then
    run_swift_framework_tests
    run_android_framework_tests
fi

echo ""
echo "================================================================"
echo "Model download tests: ${PASS_COUNT} passed, ${FAIL_COUNT} failed"
echo "================================================================"

if [[ "${FAIL_COUNT}" -ne 0 ]]; then
    exit 1
fi
echo "All model download tests passed"
