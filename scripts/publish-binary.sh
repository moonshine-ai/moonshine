#! /bin/bash -ex

VERSION=0.0.68
REPO="moonshine-ai/moonshine"

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

CORE_DIR=${REPO_ROOT_DIR}/core
BUILD_DIR=${CORE_DIR}/build

# Arguments (order-independent):
#   upload      - after packaging, publish the tarball to the GitHub release.
#   skip-build  - reuse whatever is already in BUILD_DIR instead of doing a
#                 clean rebuild (for re-packaging/re-uploading an existing
#                 build without paying for a full compile).
DO_UPLOAD=""
SKIP_BUILD=""
for arg in "$@"; do
    case "$arg" in
        upload) DO_UPLOAD=1 ;;
        skip-build|--skip-build|--no-build) SKIP_BUILD=1 ;;
        *) echo "Unknown argument: '$arg'" >&2; exit 1 ;;
    esac
done

if [[ "$OSTYPE" == "darwin"* ]]; then
	ARCH=$(uname -m)
	if [[ "$ARCH" == "arm64" ]]; then
		PLATFORM=macos-arm64
	else
		PLATFORM=macos-x86_64
	fi
elif grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null || grep -q "BCM2" /proc/cpuinfo 2>/dev/null; then
	PLATFORM=rpi-arm64
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
        PLATFORM=linux-x86_64
    else
        PLATFORM=linux-arm64
    fi
elif [[ "$OSTYPE" == "msys"* ]]; then
    echo "Use publish-binary.bat for Windows"
	exit 1
else
	echo "Unsupported platform: $OSTYPE"
	exit 1
fi

if [[ "$PLATFORM" == macos-* ]]; then
    EXPECTED_ARTIFACT=${BUILD_DIR}/moonshine.framework/Versions/A/moonshine
else
    EXPECTED_ARTIFACT=${BUILD_DIR}/libmoonshine.so
fi

if [[ -n "${SKIP_BUILD}" ]]; then
    echo "skip-build: reusing existing artifacts in ${BUILD_DIR} (no rebuild)."
    if [[ ! -f "${EXPECTED_ARTIFACT}" ]]; then
        echo "Expected pre-built artifact not found: ${EXPECTED_ARTIFACT}" >&2
        echo "Re-run without skip-build to compile it." >&2
        exit 1
    fi
else
    rm -rf ${BUILD_DIR}
    mkdir -p ${BUILD_DIR}
    cd ${BUILD_DIR}
    if [[ "$PLATFORM" == macos-* ]]; then
      cmake .. -DMOONSHINE_BUILD_SWIFT=YES
    else
      cmake ..
    fi
    make clean
    cmake --build . -v
fi

TMP_DIR=$(mktemp -d)
FOLDER_NAME=moonshine-voice-${PLATFORM}
BINARY_DIR=${TMP_DIR}/${FOLDER_NAME}
mkdir -p ${BINARY_DIR}
INCLUDE_DIR=${BINARY_DIR}/include
mkdir -p ${INCLUDE_DIR}
cp ${CORE_DIR}/moonshine-c-api.h ${INCLUDE_DIR}
cp ${CORE_DIR}/moonshine-cpp.h ${INCLUDE_DIR}

LIB_DIR=${BINARY_DIR}/lib
mkdir -p ${LIB_DIR}
if [[ "$PLATFORM" == macos-* ]]; then
    cp ${BUILD_DIR}/moonshine.framework/Versions/A/moonshine ${LIB_DIR}/libmoonshine.a
elif [[ "$PLATFORM" == "linux-x86_64" || "$PLATFORM" == "linux-arm64" || "$PLATFORM" == "rpi-arm64" ]]; then
    cp ${BUILD_DIR}/libmoonshine.so ${LIB_DIR}/libmoonshine.so
fi

cd ${TMP_DIR}
TAR_NAME=${FOLDER_NAME}.tar.gz
tar -czvf ${TAR_NAME} ${FOLDER_NAME}
cp ${TAR_NAME} ${REPO_ROOT_DIR}

if [[ -n "${DO_UPLOAD}" ]]; then
    TAR_PATH=${TMP_DIR}/${TAR_NAME}

    # Check if the GitHub release exists; create it if missing.
    if ! gh release view "v${VERSION}" >/dev/null 2>&1; then
        gh release create "v${VERSION}" --title "v${VERSION}" --notes "Release v${VERSION}"
    fi

    # `gh release upload` has no client-side timeout, so a stalled TLS
    # connection to GitHub can leave it hanging indefinitely (this mirrors the
    # Windows path, which wraps the upload in gh-upload-retry.ps1). Bound each
    # attempt with `timeout` (prefer GNU `timeout`, fall back to `gtimeout` from
    # coreutils on macOS) and retry a few times before giving up.
    if command -v timeout >/dev/null 2>&1; then
        TIMEOUT_CMD=timeout
    elif command -v gtimeout >/dev/null 2>&1; then
        TIMEOUT_CMD=gtimeout
    else
        TIMEOUT_CMD=""
    fi

    UPLOAD_TIMEOUT_SEC=180
    UPLOAD_RETRIES=5
    upload_ok=0
    for attempt in $(seq 1 ${UPLOAD_RETRIES}); do
        echo "[publish-binary] Attempt ${attempt}/${UPLOAD_RETRIES}: uploading ${TAR_NAME} to release v${VERSION}..."
        # `set -e` would otherwise abort the script on a failed attempt before we
        # get a chance to retry, so guard the call and inspect its status.
        rc=0
        if [[ -n "${TIMEOUT_CMD}" ]]; then
            "${TIMEOUT_CMD}" "${UPLOAD_TIMEOUT_SEC}" gh release upload "v${VERSION}" "${TAR_PATH}" --clobber || rc=$?
        else
            gh release upload "v${VERSION}" "${TAR_PATH}" --clobber || rc=$?
        fi
        if [[ ${rc} -eq 0 ]]; then
            upload_ok=1
            break
        fi
        echo "[publish-binary] Attempt ${attempt} failed or timed out (exit ${rc})."
        if [[ ${attempt} -lt ${UPLOAD_RETRIES} ]]; then
            sleep 5
        fi
    done

    if [[ ${upload_ok} -ne 1 ]]; then
        echo "[publish-binary] ERROR: upload of ${TAR_NAME} to v${VERSION} failed after ${UPLOAD_RETRIES} attempts." >&2
        exit 1
    fi
    echo "[publish-binary] Uploaded ${TAR_NAME} to release v${VERSION}."
fi
