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

#ifndef TENSORFLOW_LITE_MICRO_HEXDUMP_H_
#define TENSORFLOW_LITE_MICRO_HEXDUMP_H_

#include <cstddef>

#include "tensorflow/lite/micro/span.h"

namespace tflite {

// Displays the contents of a memory region, formatted in hexadecimal and ASCII
// in a style matching Python's hexdump module, using DebugLog().
void hexdump(Span<const std::byte> region);

// Writes the contents of a memory region, formatted in hexadecimal and ASCII
// in a style matching Python's hexdump module, to a buffer. Returns the portion
// of the buffer written.
Span<char> hexdump(Span<const std::byte> region, Span<char> buffer);

}  // end namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_HEXDUMP_H_
