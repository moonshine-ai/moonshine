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

# Build for macOS per-arch, then lipo. The static merge step in core/CMakeLists.txt
# pulls in libonnxruntime.a from the vendored third-party tree, which is
# arch-specific (see find-ort-library-path.cmake). Driving both arches from a
# single cmake invocation would silently drop one slice; instead we build each
# arch independently and lipo the merged archives together at the end.
cmake -B build-macos-arm64 \
	-G Xcode \
	-DCMAKE_OSX_ARCHITECTURES="arm64" \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	-DMOONSHINE_BUILD_SWIFT=YES \
	..
cmake --build build-macos-arm64 --config Release

cmake -B build-macos-x86_64 \
	-G Xcode \
	-DCMAKE_OSX_ARCHITECTURES="x86_64" \
	-DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
	-DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
	-DMOONSHINE_BUILD_SWIFT=YES \
	..
cmake --build build-macos-x86_64 --config Release

MOONSHINE_FRAMEWORK_PHONE=${CORE_BUILD_DIR}/build-phone/Release-iphoneos/moonshine.framework/
MOONSHINE_FRAMEWORK_SIMULATOR=${CORE_BUILD_DIR}/build-simulator/Release-iphonesimulator/moonshine.framework/
MOONSHINE_FRAMEWORK_MACOS_ARM64=${CORE_BUILD_DIR}/build-macos-arm64/Release/moonshine.framework/Versions/A/
MOONSHINE_FRAMEWORK_MACOS_X86_64=${CORE_BUILD_DIR}/build-macos-x86_64/Release/moonshine.framework/Versions/A/
MOONSHINE_FRAMEWORK_MACOS=${MOONSHINE_FRAMEWORK_MACOS_ARM64}

mv ${MOONSHINE_FRAMEWORK_PHONE}/moonshine ${MOONSHINE_FRAMEWORK_PHONE}/libmoonshine.a
mv ${MOONSHINE_FRAMEWORK_SIMULATOR}/moonshine ${MOONSHINE_FRAMEWORK_SIMULATOR}/libmoonshine.a
mv ${MOONSHINE_FRAMEWORK_MACOS_ARM64}/moonshine ${MOONSHINE_FRAMEWORK_MACOS_ARM64}/libmoonshine.a
mv ${MOONSHINE_FRAMEWORK_MACOS_X86_64}/moonshine ${MOONSHINE_FRAMEWORK_MACOS_X86_64}/libmoonshine.a

# Thin each per-arch archive to its own arch first. The libtool -static merge
# step inside core/CMakeLists.txt pulls in the vendored libonnxruntime.a, which
# is a fat universal archive; merging it into a per-arch libmoonshine.a causes
# the output to also become fat, but with only ONNX Runtime symbols in the
# "foreign" slice (no Moonshine code compiled for that arch). A naive
# lipo -create over the two per-arch archives would then silently keep the
# first input's foreign slice and discard the real one, leaving libmoonshine.a
# with a broken x86_64 slice (Moonshine symbols absent). Thinning first
# guarantees each input provides exactly one well-formed slice.
thin_to_arch() {
	local input=$1
	local arch=$2
	local output=$3
	if lipo -info "$input" | grep -q "^Non-fat file"; then
		cp "$input" "$output"
	else
		lipo "$input" -thin "$arch" -output "$output"
	fi
}

thin_to_arch ${MOONSHINE_FRAMEWORK_MACOS_ARM64}/libmoonshine.a arm64 \
	${MOONSHINE_FRAMEWORK_MACOS_ARM64}/libmoonshine-thin.a
thin_to_arch ${MOONSHINE_FRAMEWORK_MACOS_X86_64}/libmoonshine.a x86_64 \
	${MOONSHINE_FRAMEWORK_MACOS_X86_64}/libmoonshine-thin.a

lipo -create \
	${MOONSHINE_FRAMEWORK_MACOS_ARM64}/libmoonshine-thin.a \
	${MOONSHINE_FRAMEWORK_MACOS_X86_64}/libmoonshine-thin.a \
	-output ${MOONSHINE_FRAMEWORK_MACOS}/libmoonshine.a.fat
mv ${MOONSHINE_FRAMEWORK_MACOS}/libmoonshine.a.fat ${MOONSHINE_FRAMEWORK_MACOS}/libmoonshine.a

xcodebuild -create-xcframework \
	-library ${MOONSHINE_FRAMEWORK_PHONE}/libmoonshine.a \
	-headers ${MOONSHINE_FRAMEWORK_PHONE}/Headers \
	-library ${MOONSHINE_FRAMEWORK_SIMULATOR}/libmoonshine.a \
	-headers ${MOONSHINE_FRAMEWORK_SIMULATOR}/Headers \
	-library ${MOONSHINE_FRAMEWORK_MACOS}/libmoonshine.a \
	-headers ${MOONSHINE_FRAMEWORK_MACOS}/Headers \
	-output ${CORE_BUILD_DIR}/Moonshine.xcframework

ARCHS=("ios-arm64" "ios-arm64_x86_64-simulator" "macos-arm64_x86_64")
for ARCH in ${ARCHS[@]}; do
	HEADERS_PATH=${CORE_BUILD_DIR}/Moonshine.xcframework/${ARCH}/Headers/
	mkdir -p ${HEADERS_PATH}
	cp ${CORE_DIR}/moonshine-c-api.h ${HEADERS_PATH}/moonshine-c-api.h
	cp ${CORE_DIR}/module.modulemap ${HEADERS_PATH}/module.modulemap
	RESOURCES_PATH=${CORE_BUILD_DIR}/Moonshine.xcframework/${ARCH}/Resources/
	mkdir -p ${RESOURCES_PATH}
	cp -r ${REPO_ROOT_DIR}/test-assets ${RESOURCES_PATH}/test-assets
	rm -rf ${RESOURCES_PATH}/test-assets/.git
	rm -rf ${RESOURCES_PATH}/test-assets/.DS_Store
	rm -rf ${RESOURCES_PATH}/test-assets/output
done

rm -rf ${REPO_ROOT_DIR}/swift/Moonshine.xcframework
cp -R -P ${CORE_BUILD_DIR}/Moonshine.xcframework ${REPO_ROOT_DIR}/swift/

cp -r ${REPO_ROOT_DIR}/test-assets ${REPO_ROOT_DIR}/swift/Tests/MoonshineVoiceTests/test-assets

cd ${REPO_ROOT_DIR}/swift
swift package clean
# First time test is run it fails? Maybe a build ordering issue?
swift test || true
swift test