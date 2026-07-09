// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/micro/hexdump.h"

#include <array>

#include "tensorflow/lite/micro/span.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

constexpr tflite::Span<const char> input{
    "This is an input string for testing."};

const tflite::Span<const std::byte> region{
    reinterpret_cast<const std::byte*>(input.data()), input.size()};

// clang-format off
constexpr tflite::Span<const char> expected{
    "00000000: 54 68 69 73 20 69 73 20  61 6E 20 69 6E 70 75 74  This is an input\n"
    "00000001: 20 73 74 72 69 6E 67 20  66 6F 72 20 74 65 73 74   string for test\n"
    "00000002: 69 6E 67 2E 00                                    ing..\n"};
// clang-format on

// String literals have null terminators, but don't expect a null terminator
// in the hexdump output.
constexpr tflite::Span<const char> expected_no_null{expected.data(),
                                                    expected.size() - 1};

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(TestOutputToBuffer) {
  // Allocate a buffer with an arbitrary amount of extra room so the test has
  // the possibility of failing if hexdump mishandles the extra space.
  std::array<char, expected.size() + 10> buffer;

  tflite::Span<char> output = tflite::hexdump(region, buffer);
  TF_LITE_MICRO_EXPECT(output == expected_no_null);
}

TF_LITE_MICRO_TEST(TestOutputToDebugLog) {
  // There's no easy way to verify DebugLog output; however, test it anyhow to
  // catch an outright crash, and so the output appears in the log should
  // someone wish to examine it.
  tflite::hexdump(region);
}

TF_LITE_MICRO_TESTS_END
