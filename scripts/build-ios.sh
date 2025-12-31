#!/bin/bash -ex

if [ -z $1 ]; then
	TARGET=phone
elif [ $1 == "simulator" ]; then
	TARGET=simulator
elif [ $1 == "phone" ]; then
	TARGET=phone
else
	echo "Usage: $0 <phone|simulator>"
	exit 1
fi

IOS_VERSION=15.1

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
CORE_DIR=${REPO_ROOT_DIR}/core
CORE_BUILD_DIR=${CORE_DIR}/build

cd ${CORE_DIR}
find . -type d -name build -exec rm -rf {} +

if [ $TARGET == "phone" ]; then
	cmake -B build \
		-G Xcode \
		-DCMAKE_SYSTEM_NAME=iOS \
		-DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_VERSION} \
		-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
		.
else
	cmake -B build \
		-G Xcode \
		-DCMAKE_SYSTEM_NAME=iOS \
		-DCMAKE_OSX_SYSROOT=iphonesimulator \
		-DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
		-DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_VERSION} \
		-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
		.
fi

cd ${CORE_BUILD_DIR}
cmake --build . --config Release
