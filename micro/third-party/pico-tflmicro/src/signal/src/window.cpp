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

#include "signal/src/window.h"

#include <cstdint>

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

void ApplyWindow(const int16_t* input, const int16_t* window, int size,
                 int shift, int16_t* output) {
  for (int i = 0; i < size; ++i) {
    int32_t raw = (static_cast<int32_t>(input[i]) * window[i]) >> shift;
    if (raw < INT16_MIN) {
      output[i] = INT16_MIN;
    } else if (raw > INT16_MAX) {
      output[i] = INT16_MAX;
    } else {
      output[i] = static_cast<int16_t>(raw);
    }
  }
}
}  // namespace tflm_signal
