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

#include "signal/src/filter_bank_log.h"

#include "signal/src/log.h"

namespace tflite {
namespace tflm_signal {

void FilterbankLog(const uint32_t* input, int num_channels,
                   int32_t output_scale, uint32_t correction_bits,
                   int16_t* output) {
  for (int i = 0; i < num_channels; ++i) {
    const uint32_t scaled = input[i] << correction_bits;
    if (scaled > 1) {
      const uint32_t log_value = Log32(scaled, output_scale);
      output[i] = ((log_value < static_cast<uint32_t>(INT16_MAX))
                       ? log_value
                       : static_cast<uint32_t>(INT16_MAX));
    } else {
      output[i] = 0;
    }
  }
}

}  // namespace tflm_signal
}  // namespace tflite
