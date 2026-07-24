#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
PYTHON_DIR=${REPO_ROOT_DIR}/python

CORE_DIR=${REPO_ROOT_DIR}/core
CORE_BUILD_DIR=${CORE_DIR}/build
rm -rf ${CORE_BUILD_DIR}
mkdir -p ${CORE_BUILD_DIR}
# Align with bundled ONNX Runtime / dylibs so wheel metadata matches binary minimum macOS (silences
# delocate/wheel warnings about MACOSX_DEPLOYMENT_TARGET vs interpreter).
if [[ "$OSTYPE" == "darwin"* ]]; then
	export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
fi
# Configure/build out-of-source (cmake -S/-B) from a stable working directory
# (${CORE_DIR}, never removed) instead of cd-ing into the just-recreated
# ${CORE_BUILD_DIR}. On Docker Desktop for macOS the bind-mounted host dir is
# served over VirtioFS/gRPC-FUSE, and a freshly `rm -rf`'d + re-`mkdir`'d
# directory can briefly resolve to a stale inode, making getcwd() fail for a
# process whose cwd is inside it (cmake then aborts with "Current working
# directory cannot be established"). See scripts/publish-binary.sh for the same
# guard.
cd ${CORE_DIR}
cmake -S ${CORE_DIR} -B ${CORE_BUILD_DIR}
cmake --build ${CORE_BUILD_DIR} --config Release

# Drop stale native libs from other platforms (e.g. macOS dylibs left in the
# tree when this script runs inside Docker with the host repo bind-mounted).
rm -f ${PYTHON_DIR}/src/moonshine_voice/libmoonshine.* \
	${PYTHON_DIR}/src/moonshine_voice/libonnxruntime.*

cp ${CORE_BUILD_DIR}/libmoonshine.* ${PYTHON_DIR}/src/moonshine_voice/

if [[ "$OSTYPE" == "darwin"* ]]; then
	ARCH=$(uname -m)
	if [[ "$ARCH" == "arm64" ]]; then
		cp ${CORE_DIR}/third-party/onnxruntime/lib/macos/arm64/libonnxruntime*.dylib ${PYTHON_DIR}/src/moonshine_voice/
	else
		cp ${CORE_DIR}/third-party/onnxruntime/lib/macos/x86_64/libonnxruntime*.dylib ${PYTHON_DIR}/src/moonshine_voice/
	fi
	codesign --force --sign - ${PYTHON_DIR}/src/moonshine_voice/libmoonshine.dylib
elif grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null || grep -q "BCM2" /proc/cpuinfo 2>/dev/null; then
    LINUX_VERSION=2_39
	ORT_LINUX_LIB_DIR=${CORE_DIR}/third-party/onnxruntime/lib/linux/aarch64
	cp ${ORT_LINUX_LIB_DIR}/libonnxruntime*.so* ${PYTHON_DIR}/src/moonshine_voice/
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    LINUX_VERSION=2_34
	# Pick the ONNX Runtime build matching the machine we're running on. The Pi
	# branch above handles Raspberry Pi hardware specifically; this branch covers
	# generic Linux, including native arm64 (e.g. an aarch64 Docker container on
	# an Apple Silicon host), so select aarch64 vs x86_64 by uname rather than
	# assuming x86_64.
	if [[ "$(uname -m)" == "aarch64" || "$(uname -m)" == "arm64" ]]; then
		ORT_LINUX_LIB_DIR=${CORE_DIR}/third-party/onnxruntime/lib/linux/aarch64
	else
		ORT_LINUX_LIB_DIR=${CORE_DIR}/third-party/onnxruntime/lib/linux/x86_64
	fi
	cp ${ORT_LINUX_LIB_DIR}/libonnxruntime*.so* ${PYTHON_DIR}/src/moonshine_voice/
elif [[ "$OSTYPE" == "msys"* ]]; then
	cp ${CORE_DIR}/third-party/onnxruntime/lib/windows/x86_64/libonnxruntime*.dll ${PYTHON_DIR}/src/moonshine_voice/
else
	echo "Unsupported platform: $OSTYPE"
	echo "You'll need to manually copy the ONNX Runtime library to the python/src/moonshine_voice/ directory."
fi

cd ${PYTHON_DIR}

if [[ "$OSTYPE" == "darwin"* ]]; then
	export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-15.0}"
fi

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

# Build platform-specific wheel (PEP 517 avoids deprecated setup.py install paths)
rm -rf dist/* wheelhouse/*
uv build --wheel --out-dir dist
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
	# Target manylinux_2_34 for wider compatibility (default would be 2_39 on newer images).
	# libmoonshine.so now carries only a `$ORIGIN` RUNPATH (so the released C++ .so is
	# relocatable), which no longer points auditwheel at the vendored ONNX Runtime.
	# Add that directory to LD_LIBRARY_PATH so auditwheel can locate libonnxruntime.so.1
	# and graft it into the repaired wheel.
	LD_LIBRARY_PATH="${ORT_LINUX_LIB_DIR}:${LD_LIBRARY_PATH:-}" \
		auditwheel repair dist/moonshine_voice-*.whl -w dist/ --plat "manylinux_${LINUX_VERSION}_${ARCH}"
	rm -rf dist/moonshine_voice-*-linux_*.whl
fi

if [[ "$1" == "upload" ]]; then
	twine upload --verbose --skip-existing dist/*
fi
