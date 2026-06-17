/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/src/filter_bank_square_root.h"

#include "signal/src/square_root.h"

namespace tflite {
namespace tflm_signal {

void FilterbankSqrt(const uint64_t* input, int num_channels,
                    int scale_down_bits, uint32_t* output) {
  for (int i = 0; i < num_channels; ++i) {
    output[i] = Sqrt64(input[i]) >> scale_down_bits;
  }
}

}  // namespace tflm_signal
}  // namespace tflite
