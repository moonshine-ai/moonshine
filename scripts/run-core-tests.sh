#!/bin/bash -ex

SCRIPTS_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT_DIR=$(dirname $SCRIPTS_DIR)
BUILD_DIR=${REPO_ROOT_DIR}/core/build
MOONSHINE_TTS_BUILD_DIR=${REPO_ROOT_DIR}/core/moonshine-tts/build

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
${REPO_ROOT_DIR}/core/build/word-alignment-test

# moonshine-tts tests resolve bundled assets via ``core/moonshine-tts/data`` relative to the monorepo root.
cd "${REPO_ROOT_DIR}"

# moonshine-tts (core + ONNX); binaries live under core/moonshine-tts/build
${MOONSHINE_TTS_BUILD_DIR}/utf8_utils_test
${MOONSHINE_TTS_BUILD_DIR}/german_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/dutch_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/italian_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/portuguese_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/russian_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/chinese_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/korean_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/vietnamese_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/french_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/spanish_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/turkish_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/ukrainian_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/hindi_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/text_normalize_test
${MOONSHINE_TTS_BUILD_DIR}/heteronym_context_test
${MOONSHINE_TTS_BUILD_DIR}/ipa_postprocess_test
${MOONSHINE_TTS_BUILD_DIR}/cmudict_tsv_test
${MOONSHINE_TTS_BUILD_DIR}/json_config_test
${MOONSHINE_TTS_BUILD_DIR}/english_hand_oov_test
${MOONSHINE_TTS_BUILD_DIR}/onnx_g2p_smoke_test
${MOONSHINE_TTS_BUILD_DIR}/korean_tok_pos_onnx_test
${MOONSHINE_TTS_BUILD_DIR}/japanese_tok_pos_onnx_test
${MOONSHINE_TTS_BUILD_DIR}/chinese_tok_pos_onnx_test
${MOONSHINE_TTS_BUILD_DIR}/japanese_onnx_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/arabic_rule_g2p_test
${MOONSHINE_TTS_BUILD_DIR}/english_rule_g2p_test

echo "All tests passed"
