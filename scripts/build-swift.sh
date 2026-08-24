#!/bin/bash -ex

IOS_VERSION=15.1

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
CORE_DIR=${REPO_ROOT_DIR}/core
CORE_BUILD_DIR=${CORE_DIR}/build

cd ${CORE_DIR}
# The three configures below each own a tree under here, so clearing this one
# directory is a full clean.
rm -rf ${CORE_BUILD_DIR}

mkdir -p ${CORE_BUILD_DIR}
cd ${CORE_BUILD_DIR}
cmake -B build-phone \
	-G Xcode \
	-DCMAKE_SYSTEM_NAME=iOS \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_VERSION} \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	..
# Only the moonshine framework is packaged. ALL_BUILD also compiles every
# unit-test binary (and lipos each one for macOS), which has filled the disk
# mid-release more than once.
cmake --build build-phone --config Release --target moonshine

cmake -B build-simulator \
	-G Xcode \
	-DCMAKE_SYSTEM_NAME=iOS \
	-DCMAKE_OSX_SYSROOT=iphonesimulator \
	-DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=${IOS_VERSION} \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	..
cmake --build build-simulator --config Release --target moonshine

# Build for macOS
cmake -B build-macos \
	-G Xcode \
	-DCMAKE_OSX_ARCHITECTURES="x86_64;arm64" \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	-DMOONSHINE_BUILD_SWIFT=YES \
	..
cmake --build build-macos --config Release --target moonshine

MOONSHINE_FRAMEWORK_PHONE=${CORE_BUILD_DIR}/build-phone/Release-iphoneos/moonshine.framework/
MOONSHINE_FRAMEWORK_SIMULATOR=${CORE_BUILD_DIR}/build-simulator/Release-iphonesimulator/moonshine.framework/
MOONSHINE_FRAMEWORK_MACOS=${CORE_BUILD_DIR}/build-macos/Release/moonshine.framework/Versions/A/

mv ${MOONSHINE_FRAMEWORK_PHONE}/moonshine ${MOONSHINE_FRAMEWORK_PHONE}/libmoonshine.a
mv ${MOONSHINE_FRAMEWORK_SIMULATOR}/moonshine ${MOONSHINE_FRAMEWORK_SIMULATOR}/libmoonshine.a
mv ${MOONSHINE_FRAMEWORK_MACOS}/moonshine ${MOONSHINE_FRAMEWORK_MACOS}/libmoonshine.a

xcodebuild -create-xcframework \
	-library ${MOONSHINE_FRAMEWORK_PHONE}/libmoonshine.a \
	-headers ${MOONSHINE_FRAMEWORK_PHONE}/Headers \
	-library ${MOONSHINE_FRAMEWORK_SIMULATOR}/libmoonshine.a \
	-headers ${MOONSHINE_FRAMEWORK_SIMULATOR}/Headers \
	-library ${MOONSHINE_FRAMEWORK_MACOS}/libmoonshine.a \
	-headers ${MOONSHINE_FRAMEWORK_MACOS}/Headers \
	-output ${CORE_BUILD_DIR}/Moonshine.xcframework

# The three platform trees are only needed to produce the .a files that
# create-xcframework already copied in. Dropping them before we stage
# test-assets keeps peak disk use within a laptop-sized volume.
rm -rf ${CORE_BUILD_DIR}/build-phone \
	${CORE_BUILD_DIR}/build-simulator \
	${CORE_BUILD_DIR}/build-macos

ARCHS=("ios-arm64" "ios-arm64_x86_64-simulator" "macos-arm64_x86_64")
for ARCH in ${ARCHS[@]}; do
	HEADERS_PATH=${CORE_BUILD_DIR}/Moonshine.xcframework/${ARCH}/Headers/
	mkdir -p ${HEADERS_PATH}
	cp ${CORE_DIR}/moonshine-c-api.h ${HEADERS_PATH}/moonshine-c-api.h
	cp ${CORE_DIR}/module.modulemap ${HEADERS_PATH}/module.modulemap
	RESOURCES_PATH=${CORE_BUILD_DIR}/Moonshine.xcframework/${ARCH}/Resources/
	mkdir -p ${RESOURCES_PATH}
	# -c is an APFS clone: same bytes, no extra disk until something writes.
	cp -cR ${REPO_ROOT_DIR}/test-assets ${RESOURCES_PATH}/test-assets
	rm -rf ${RESOURCES_PATH}/test-assets/.git
	rm -rf ${RESOURCES_PATH}/test-assets/.DS_Store
	rm -rf ${RESOURCES_PATH}/test-assets/output
done

rm -rf ${REPO_ROOT_DIR}/language-bindings/swift/Moonshine.xcframework
cp -cR -P ${CORE_BUILD_DIR}/Moonshine.xcframework ${REPO_ROOT_DIR}/language-bindings/swift/

rm -rf ${REPO_ROOT_DIR}/language-bindings/swift/Tests/MoonshineVoiceTests/test-assets
cp -cR ${REPO_ROOT_DIR}/test-assets ${REPO_ROOT_DIR}/language-bindings/swift/Tests/MoonshineVoiceTests/test-assets

cd ${REPO_ROOT_DIR}/language-bindings/swift
swift package clean
# First time test is run it fails? Maybe a build ordering issue?
swift test || true
swift test