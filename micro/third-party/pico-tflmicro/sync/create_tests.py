#!/usr/bin/env python3
# Lint as: python2, python3
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Moves source files to match Arduino library conventions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import shutil
import sys

TESTS_TO_IGNORE = {
  "micro_interpreter_test",     # Too large for the RP2040.
  "compression_metadata_test",  # Uses experimental compression.
}

def create_test(test_name, source_cc_path, output_folder, cmake_template):
  if test_name in TESTS_TO_IGNORE:
    return

  if test_name == "kernels_conv_test":
    extra_cc_source_filenames = [
      "conv_test_common.cc",
      "testdata/conv_test_data.cc",
      "testdata/conv_test_data.h",
    ]
    extra_cpp_path = "tensorflow/lite/micro/kernels/"
  elif test_name == "kernels_circular_buffer_test":
    extra_cc_source_filenames = [
      "circular_buffer_flexbuffers_generated_data.cc",
      "circular_buffer_flexbuffers_generated_data.h",
    ]
    extra_cpp_path = "tensorflow/lite/micro/kernels"
  elif test_name == "kernels_detection_postprocess_test":
    extra_cc_source_filenames = [
      "detection_postprocess_flexbuffers_generated_data.cc",
      "detection_postprocess_flexbuffers_generated_data.h",
    ]
    extra_cpp_path = "tensorflow/lite/micro/kernels"
  elif test_name == "kernels_lstm_eval_test" or test_name == "kernels_unidirectional_sequence_lstm_test":
    extra_cc_source_filenames = [
      "testdata/lstm_test_data.cc",
      "testdata/lstm_test_data.h",
    ]
    extra_cpp_path = "tensorflow/lite/micro/kernels"
  elif test_name == "memory_arena_threshold_test":
    extra_cc_source_filenames = [
      "models/keyword_scrambled_model_data.cc",
      "models/keyword_scrambled_model_data.h",
      "testing/test_conv_model.cc",
      "testing/test_conv_model.h",
    ]
    extra_cpp_path = "tensorflow/lite/micro"
  elif test_name == "micro_allocator_test" or test_name == "recording_micro_allocator_test":
    extra_cc_source_filenames = [
      "testing/test_conv_model.cc",
      "testing/test_conv_model.h",
    ]
    extra_cpp_path = "tensorflow/lite/micro"
  else:
    extra_cc_source_filenames = []
    extra_cpp_path = None

  os.makedirs(output_folder, exist_ok=True)
  source_cc_dir = os.path.dirname(source_cc_path)
  cpp_base = os.path.basename(source_cc_path).replace(".cc", ".cpp")
  output_cc_path = os.path.join(output_folder, cpp_base)
  shutil.copyfile(source_cc_path, output_cc_path)
  extra_sources_text = ""
  for extra_cc_source_filename in extra_cc_source_filenames:
    extra_source_cc_path = os.path.join(os.path.dirname(source_cc_path), extra_cc_source_filename)
    extra_cpp_base = extra_cc_source_filename.replace(".cc", ".cpp")
    extra_output_cpp_path = os.path.join(output_folder, extra_cpp_path, extra_cpp_base)
    os.makedirs(os.path.dirname(extra_output_cpp_path), exist_ok=True)
    shutil.copyfile(extra_source_cc_path, extra_output_cpp_path)
    if extra_cpp_path:
      extra_sources_text += "  ${CMAKE_CURRENT_LIST_DIR}/../../tests/" + test_name + "/" + extra_cpp_path + "/" + extra_cpp_base + "\n"
    else:
      extra_sources_text += "  ${CMAKE_CURRENT_LIST_DIR}/../../tests/" + test_name + "/" + extra_cpp_base + "\n"

  cmake_contents = cmake_template.replace("{{TEST_NAME}}", test_name)
  cmake_contents = cmake_contents.replace("{{EXTRA_SOURCES}}", extra_sources_text)
  cmake_contents = cmake_contents.replace("{{CPP_BASE}}", cpp_base)
  cmake_path = os.path.join(output_folder, "CMakeLists.txt")
  with open(cmake_path, "w") as cmake_file:
      cmake_file.write(cmake_contents)


def create_tests(tflm_path, output_path, cmake_template):
  micro_path = os.path.join(tflm_path, "tensorflow/lite/micro/")
  search_paths = [
    os.path.join(micro_path, "*_test.cc"),
    os.path.join(micro_path, "**", "*_test.cc"),
  ]
  any_found = False
  for search_path in search_paths:
    for test_cc_path in glob.glob(search_path):
      any_found = True
      test_name = test_cc_path.replace(micro_path, "").replace("/", "_").replace(".cc", "")
      test_output_folder = os.path.join(output_path, "tests", test_name)
      create_test(test_name, test_cc_path, test_output_folder, cmake_template)
  return any_found

if __name__ == "__main__":
  if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} <TFL Micro Repo> <Output directory>")
    sys.exit(1)
  with open("sync/test_CMakeLists_template.txt", "r") as cmake_template_file:
    cmake_template = cmake_template_file.read()
  any_found = create_tests(sys.argv[1], sys.argv[2], cmake_template)
  if not any_found:
    print(f"No tests found at {sys.argv[1]}")
    sys.exit(1)
