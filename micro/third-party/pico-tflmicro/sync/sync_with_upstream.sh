#!/bin/bash -xe

TFLM_REPO_URL=https://github.com/tensorflow/tflite-micro
ROOT_DIR=${PWD}
BUILD_DIR=${ROOT_DIR}/build
TFLM_REPO_DIR=${BUILD_DIR}/tflm_repo
ARDUINO_REPO_DIR=${BUILD_DIR}/arduino_repo
TFLM_TREE_DIR="${BUILD_DIR}/tflm_tree"

rm -rf ${TFLM_REPO_DIR}
mkdir -p ${TFLM_REPO_DIR}
cd "${TFLM_REPO_DIR}"
git clone --depth 1 --single-branch ${TFLM_REPO_URL}
cd tflite-micro

make -f tensorflow/lite/micro/tools/make/Makefile clean_downloads

TARGET=cortex_m_generic
OPTIMIZED_KERNEL_DIR=cmsis_nn
TARGET_ARCH=project_generation

# Create the TFLM base tree
rm -rf ${TFLM_TREE_DIR}
mkdir -p ${TFLM_TREE_DIR}
python3 tensorflow/lite/micro/tools/project_generation/create_tflm_tree.py \
  -e hello_world -e micro_speech -e person_detection \
  --makefile_options="TARGET=${TARGET} OPTIMIZED_KERNEL_DIR=${OPTIMIZED_KERNEL_DIR} TARGET_ARCH=${TARGET_ARCH}" \
  "${TFLM_TREE_DIR}"

# Create the final tree in ${ARDUINO_REPO_DIR} using the base tree in ${TFLM_TREE_DIR}
# The create_tflm_arduino.py script takes care of cleaning ${ARDUINO_REPO_DIR}
rm -rf ${ARDUINO_REPO_DIR}
mkdir -p ${ARDUINO_REPO_DIR}
cd "${ROOT_DIR}"
python3 sync/create_tflm_arduino.py \
  --output_dir="${ARDUINO_REPO_DIR}" \
  --base_dir="${TFLM_TREE_DIR}" \
  --manifest_file=sync/MANIFEST.ini

# Copy over the bulk of TFLM source files.
rm -rf src
mkdir src
cp -r ${ARDUINO_REPO_DIR}/src/tensorflow src/tensorflow
cp -r ${ARDUINO_REPO_DIR}/src/third_party src/third_party
cp -r ${ARDUINO_REPO_DIR}/signal src/signal

# Build data files needed for tests.
cd ${ROOT_DIR}/build/tflm_repo/tflite-micro
python3 tensorflow/lite/micro/tools/generate_cc_arrays.py \
  tensorflow/lite/micro/models/keyword_scrambled_model_data.cc \
  tensorflow/lite/micro/models/keyword_scrambled.tflite
python3 tensorflow/lite/micro/tools/generate_cc_arrays.py \
  tensorflow/lite/micro/models/keyword_scrambled_model_data.h \
  tensorflow/lite/micro/models/keyword_scrambled.tflite
cd ${ROOT_DIR}

# Make the tests.
rm -rf tests
sync/create_tests.py ${TFLM_REPO_DIR}/tflite-micro .

# Override specific files with special versions.
cp sync/micro_time.cpp src/tensorflow/lite/micro/micro_time.cpp
cp sync/system_setup.cpp src/tensorflow/lite/micro/system_setup.cpp
cp sync/arm_nn_mat_mult_nt_t_s8.c src/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c
mkdir -p src/tensorflow/lite/micro/benchmarks
cp sync/micro_benchmark.h src/tensorflow/lite/micro/benchmarks

mkdir -p src/tensorflow/lite/micro/testing
cp sync/micro_test.h src/tensorflow/lite/micro/testing

rm -rf CMakeLists.txt
SOURCE_LIST_FILE="$(mktemp /tmp/source_file_list.XXXXXXXXX)" || exit 1
find src \( -iname "*.cpp" -o -iname "*.c" -o -iname "*.h" \) | \
  sort | \
  sed -E 's#(.*)#\  ${CMAKE_CURRENT_LIST_DIR}/\1#g' \
  > ${SOURCE_LIST_FILE}

TEST_LIST_FILE="$(mktemp /tmp/test_file_list.XXXXXXXXX)" || exit 1
ls tests | \
  sort | \
  sed -E 's#(.*)#add_subdirectory("tests/\1")#g' \
  > ${TEST_LIST_FILE}

sync/replace_string_with_file_contents.py \
  CMakeLists_template.txt \
  ${SOURCE_LIST_FILE} \
  "{{LIBRARY_SOURCES}}" \
  ${TEST_LIST_FILE} \
  "{{TEST_FOLDERS}}" \
  > CMakeLists.txt