#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
PYTHON_DIR=${REPO_ROOT_DIR}/python

CORE_DIR=${REPO_ROOT_DIR}/core
CORE_BUILD_DIR=${CORE_DIR}/build
rm -rf ${CORE_BUILD_DIR}
mkdir -p ${CORE_BUILD_DIR}
cd ${CORE_BUILD_DIR}
cmake ..
cmake --build . --config Release

cp ${CORE_BUILD_DIR}/libmoonshine.* ${PYTHON_DIR}/src/moonshine_voice/

if [[ "$OSTYPE" == "darwin"* ]]; then
	cp ${CORE_DIR}/third-party/onnxruntime/lib/macos/arm64/libonnxruntime*.dylib ${PYTHON_DIR}/src/moonshine_voice/
elif grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null || grep -q "BCM2" /proc/cpuinfo 2>/dev/null; then
	cp ${CORE_DIR}/third-party/onnxruntime/lib/linux/aarch64/libonnxruntime*.so* ${PYTHON_DIR}/src/moonshine_voice/
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
	cp ${CORE_DIR}/third-party/onnxruntime/lib/linux/x86_64/libonnxruntime*.so* ${PYTHON_DIR}/src/moonshine_voice/
elif [[ "$OSTYPE" == "msys"* ]]; then
	cp ${CORE_DIR}/third-party/onnxruntime/lib/windows/x86_64/libonnxruntime*.dll ${PYTHON_DIR}/src/moonshine_voice/
else
	echo "Unsupported platform: $OSTYPE"
	echo "You'll need to manually copy the ONNX Runtime library to the python/src/moonshine_voice/ directory."
fi

cd ${PYTHON_DIR}

rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -r build-requirements.txt

# Determine platform tag for wheel name
if [[ "$OSTYPE" == "darwin"* ]]; then
	# macOS - detect architecture
	ARCH=$(uname -m)
	if [[ "$ARCH" == "arm64" ]]; then
		PLAT_NAME="macosx_11_0_arm64"
	else
		PLAT_NAME="macosx_10_9_x86_64"
	fi
elif grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null || grep -q "BCM2" /proc/cpuinfo 2>/dev/null; then
	PLAT_NAME="manylinux_2_17_aarch64"
	ARCH="aarch64"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
	ARCH=$(uname -m)
	if [[ "$ARCH" == "x86_64" ]]; then
		PLAT_NAME="manylinux_2_17_x86_64"
	else
		PLAT_NAME="manylinux_2_17_${ARCH}"
	fi
elif [[ "$OSTYPE" == "msys"* ]]; then
	PLAT_NAME="win_amd64"
else
	PLAT_NAME="any"
fi

# Build platform-specific wheel
# bdist_wheel should auto-detect platform from binary files, but we can also specify it
rm -rf dist/* wheelhouse/*
uv run setup.py bdist_wheel
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	# Target manylinux_2_34 for wider compatibility (default would be 2_39 on newer images)
	auditwheel repair dist/moonshine_voice-*.whl -w dist/ --plat "manylinux_2_34_${ARCH}"
	rm -rf dist/moonshine_voice-*-linux_*.whl
fi

if [[ "$1" == "upload" ]]; then
	twine upload dist/*
fi
