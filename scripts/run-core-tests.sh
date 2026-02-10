#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
BUILD_DIR=${REPO_ROOT_DIR}/core/build

rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}
cmake ..
make clean
cmake --build .

cd ${REPO_ROOT_DIR}/test-assets

export LD_LIBRARY_PATH=${REPO_ROOT_DIR}/core/third-party/onnxruntime/lib/linux/x86_64

${REPO_ROOT_DIR}/core/bin-tokenizer/build/bin-tokenizer-test
${REPO_ROOT_DIR}/core/third-party/onnxruntime/build/onnxruntime-test
${REPO_ROOT_DIR}/core/moonshine-utils/build/debug-utils-test
${REPO_ROOT_DIR}/core/moonshine-utils/build/string-utils-test
${REPO_ROOT_DIR}/core/build/resampler-test
${REPO_ROOT_DIR}/core/build/voice-activity-detector-test
${REPO_ROOT_DIR}/core/build/transcriber-test
${REPO_ROOT_DIR}/core/build/moonshine-c-api-test
${REPO_ROOT_DIR}/core/build/moonshine-cpp-test
${REPO_ROOT_DIR}/core/build/cosine-distance-test
${REPO_ROOT_DIR}/core/build/speaker-embedding-model-test
${REPO_ROOT_DIR}/core/build/online-clusterer-test

echo "All tests passed"
