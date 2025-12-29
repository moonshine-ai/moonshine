#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
BUILD_DIR=${REPO_ROOT_DIR}/core/build

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake ..
make clean
cmake --build .

cd ${REPO_ROOT_DIR}/test-assets

${REPO_ROOT_DIR}/core/bin-tokenizer/build/bin-tokenizer-test
${REPO_ROOT_DIR}/core/third-party/onnxruntime/build/onnxruntime-test
${REPO_ROOT_DIR}/core/third-party/ten_vad/build/ten_vad-test
${REPO_ROOT_DIR}/core/moonshine-utils/build/debug-utils-test
${REPO_ROOT_DIR}/core/moonshine-utils/build/string-utils-test
${REPO_ROOT_DIR}/core/build/resampler-test
${REPO_ROOT_DIR}/core/build/voice-activity-detector-test
${REPO_ROOT_DIR}/core/build/transcriber-test
${REPO_ROOT_DIR}/core/build/moonshine-test-v2

echo "All tests passed"