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

#ifndef SIGNAL_SRC_FILTER_BANK_LOG_H_
#define SIGNAL_SRC_FILTER_BANK_LOG_H_

#include <stdint.h>

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above

// Apply natural log to each element in array `input` of size `num_channels`
// with pre-shift and post scaling.
// The operation is roughly equivalent to:
// `output` = min(Log(`input` << `correction_bits`) * `output_scale`, INT16_MAX)
//  Where:
//    If (input << `correction_bits`) is 1 or 0, the function returns 0
void FilterbankLog(const uint32_t* input, int num_channels,
                   int32_t output_scale, uint32_t correction_bits,
                   int16_t* output);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_SRC_FILTER_BANK_LOG_H_
