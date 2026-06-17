/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SIGNAL_MICRO_KERNELS__SRC_PCAN_AGC_FIXED_H
#define SIGNAL_MICRO_KERNELS__SRC_PCAN_AGC_FIXED_H
#include <cstdint>

#include "msb.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace tflm_signal {

#define kPcanSnrBits 12
#define kPcanOutputBits 6

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut);

uint32_t PcanShrink(const uint32_t x);

void ApplyPcanAutoGainControlFixed(const int16_t* gain_lut, int32_t snr_shift,
                                   const uint32_t* noise_estimate,
                                   uint32_t* filterbank_output,
                                   int num_channels);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_MICRO_KERNELS__PCAN_AGC_FIXED_H
