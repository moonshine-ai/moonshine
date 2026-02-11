#! /bin/bash -ex

VERSION=0.0.48
REPO="moonshine-ai/moonshine"

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)

CORE_DIR=${REPO_ROOT_DIR}/core
BUILD_DIR=${CORE_DIR}/build

if [[ "$OSTYPE" == "darwin"* ]]; then
	PLATFORM=macos
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

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
if [[ "$PLATFORM" == "macos" ]]; then
  cmake .. -DMOONSHINE_BUILD_SWIFT=YES
else
  cmake ..
fi
make clean
cmake --build . -v

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
if [[ "$PLATFORM" == "macos" ]]; then
    cp ${BUILD_DIR}/moonshine.framework/Versions/A/moonshine ${LIB_DIR}/libmoonshine.a
elif [[ "$PLATFORM" == "linux-x86_64" || "$PLATFORM" == "linux-arm64" || "$PLATFORM" == "rpi-arm64" ]]; then
    cp ${BUILD_DIR}/libmoonshine.so ${LIB_DIR}/libmoonshine.so
fi

cd ${TMP_DIR}
TAR_NAME=${FOLDER_NAME}.tar.gz
tar -czvf ${TAR_NAME} ${FOLDER_NAME}
cp ${TAR_NAME} ${REPO_ROOT_DIR}

cd ${REPO_ROOT_DIR}

# Check if the GitHub release exists; create it if missing
if ! gh release view v$VERSION >/dev/null 2>&1; then
	gh release create v$VERSION --title "v$VERSION" --notes "Release v$VERSION"
fi

# gh release upload v$VERSION $TAR_NAME --clobber

cd ${REPO_ROOT_DIR}/examples

for EXAMPLE_DIR in *; do
	if [ -d "$EXAMPLE_DIR" ]; then
	    TAR_NAME=${EXAMPLE_DIR}-examples.tar.gz
		cd ${EXAMPLE_DIR}
	    tar -czvf ${TAR_NAME} *
		gh release upload v$VERSION ${TAR_NAME} --clobber
		rm -rf ${TAR_NAME}
		cd ..
	fi
done