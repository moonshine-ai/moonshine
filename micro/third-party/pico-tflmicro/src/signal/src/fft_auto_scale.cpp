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

#include "signal/src/fft_auto_scale.h"

#include <stddef.h>
#include <stdint.h>

#include "signal/src/max_abs.h"
#include "signal/src/msb.h"

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflite {
namespace tflm_signal {

int FftAutoScale(const int16_t* input, int size, int16_t* output) {
  const int16_t max = MaxAbs16(input, size);
  int scale_bits = (sizeof(int16_t) * 8) - MostSignificantBit32(max) - 1;
  if (scale_bits <= 0) {
    scale_bits = 0;
  }
  for (int i = 0; i < size; i++) {
    // (input[i] << scale_bits) is undefined if input[i] is negative.
    // Multiply explicitly to make the code portable.
    output[i] = input[i] * (1 << scale_bits);
  }
  return scale_bits;
}
}  // namespace tflm_signal
}  // namespace tflite
