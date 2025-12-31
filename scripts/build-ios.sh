#!/bin/bash -ex

IOS_VERSION=15.1

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
CORE_DIR=${REPO_ROOT_DIR}/core
CORE_BUILD_DIR=${CORE_DIR}/build

cd ${CORE_DIR}
find . -type d -name build -exec rm -rf {} +

mkdir -p ${CORE_BUILD_DIR}
cd ${CORE_BUILD_DIR}
cmake -B build-phone \
	-G Xcode \
	-DCMAKE_SYSTEM_NAME=iOS \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_VERSION} \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	..
cmake --build build-phone --config Release

cmake -B build-simulator \
	-G Xcode \
	-DCMAKE_SYSTEM_NAME=iOS \
	-DCMAKE_OSX_SYSROOT=iphonesimulator \
	-DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_VERSION} \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	..
cmake --build build-simulator --config Release
