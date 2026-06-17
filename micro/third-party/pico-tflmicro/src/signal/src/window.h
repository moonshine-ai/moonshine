/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef SIGNAL_SRC_WINDOW_H_
#define SIGNAL_SRC_WINDOW_H_

#include <stdint.h>

namespace tflm_signal {

// Applies a window function to an input signal
//
// * `input` and `window` must be both of size `size` elements and are
//    multiplied element-by element.
// * `shift` is a right shift to apply before writing the result to `output`.
void ApplyWindow(const int16_t* input, const int16_t* window, int size,
                 int shift, int16_t* output);
}  // namespace tflm_signal
#endif  // SIGNAL_SRC_WINDOW_H_
