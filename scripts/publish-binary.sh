#! /bin/bash -ex

VERSION=0.0.70
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
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # A Raspberry Pi is just an aarch64 Linux box as far as this library is
    # concerned (same vendored ONNX Runtime, same build), so package it as the
    # generic linux-arm64 artifact rather than a Pi-specific one.
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
elif [[ "$PLATFORM" == "linux-x86_64" || "$PLATFORM" == "linux-arm64" ]]; then
    cp ${BUILD_DIR}/libmoonshine.so ${LIB_DIR}/libmoonshine.so
    # Ship the ONNX Runtime shared library that libmoonshine.so depends on
    # (its DT_NEEDED is libonnxruntime.so.1) right next to it. Combined with the
    # $ORIGIN rpath baked into libmoonshine.so, this makes the extracted lib/
    # folder self-contained: no LD_LIBRARY_PATH needed to run against it.
    if [[ "$PLATFORM" == "linux-arm64" ]]; then
        ORT_ARCH_DIR=aarch64
    else
        ORT_ARCH_DIR=x86_64
    fi
    cp ${CORE_DIR}/third-party/onnxruntime/lib/linux/${ORT_ARCH_DIR}/libonnxruntime.so.1 \
        ${LIB_DIR}/libonnxruntime.so.1
fi

cd ${TMP_DIR}
TAR_NAME=${FOLDER_NAME}.tar.gz
tar -czvf ${TAR_NAME} ${FOLDER_NAME}
cp ${TAR_NAME} ${REPO_ROOT_DIR}

if [[ -n "${DO_UPLOAD}" ]]; then
    TAR_PATH=${TMP_DIR}/${TAR_NAME}

    # Create the release if needed and upload with a bounded per-attempt timeout
    # and retries. The retry logic is shared with the Docker-based arm64 build
    # (scripts/build-pip-docker.sh) via this helper.
    REPO="${REPO}" "${SCRIPTS_DIR}/gh-upload-retry.sh" "${VERSION}" "${TAR_PATH}"
fi
