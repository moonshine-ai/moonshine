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

# Build for macOS
cmake -B build-macos \
	-G Xcode \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	-DMOONSHINE_BUILD_SWIFT=YES \
	..
cmake --build build-macos --config Release

xcodebuild -create-xcframework \
	-framework build-phone/Release-iphoneos/moonshine.framework \
	-framework build-simulator/Release-iphonesimulator/moonshine.framework \
	-framework build-macos/Release/moonshine.framework \
	-output ${CORE_BUILD_DIR}/Moonshine.xcframework

ARCHS=("ios-arm64" "ios-arm64_x86_64-simulator" "macos-arm64")
for ARCH in ${ARCHS[@]}; do
	MODULES_PATH=${CORE_BUILD_DIR}/Moonshine.xcframework/${ARCH}/moonshine.framework/Modules/
	mkdir -p ${MODULES_PATH}
	cp ${CORE_DIR}/module.modulemap ${MODULES_PATH}/module.modulemap
	HEADERS_PATH=${CORE_BUILD_DIR}/Moonshine.xcframework/${ARCH}/moonshine.framework/Headers/
	mkdir -p ${HEADERS_PATH}
	cp ${CORE_DIR}/moonshine.h ${HEADERS_PATH}/moonshine.h
done

cp -r ${CORE_BUILD_DIR}/Moonshine.xcframework ${REPO_ROOT_DIR}/swift/
