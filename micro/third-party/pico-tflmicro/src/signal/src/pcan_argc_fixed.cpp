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

#include "pcan_argc_fixed.h"

namespace tflite {
namespace tflm_signal {

int16_t WideDynamicFunction(const uint32_t x, const int16_t* lut) {
  if (x <= 2) {
    return lut[x];
  }

  const int16_t interval = MostSignificantBit32(x);
  lut += 4 * interval - 6;

  const int16_t frac =
      ((interval < 11) ? (x << (11 - interval)) : (x >> (interval - 11))) &
      0x3FF;

  int32_t result = ((int32_t)lut[2] * frac) >> 5;
  result += (int32_t)((uint32_t)lut[1] << 5);
  result *= frac;
  result = (result + (1 << 14)) >> 15;
  result += lut[0];
  return (int16_t)result;
}

// Evaluate the piecewise polynomial "shrink" function defined by
//   shrink(x) = x^2 / 4  for x < 2,
//   shrink(x) = x - 1    for x >= 2.
// The input x has kPcanSnrBits fractional bits, and the output has
// kPcanOutputBits fractional bits.
uint32_t PcanShrink(const uint32_t x) {
  TFLITE_DCHECK(kPcanSnrBits >= kPcanOutputBits);
  if (x < (2 << kPcanSnrBits)) {
    // Compute x^2 / 4.
    return (x * x) >> (2 + 2 * kPcanSnrBits - kPcanOutputBits);
  } else {
    // Compute x - 1.
    return (x >> (kPcanSnrBits - kPcanOutputBits)) - (1 << kPcanOutputBits);
  }
}

void ApplyPcanAutoGainControlFixed(const int16_t* gain_lut, int32_t snr_shift,
                                   const uint32_t* noise_estimate,
                                   uint32_t* filterbank_output,
                                   int num_channels) {
  int i;
  for (i = 0; i < num_channels; ++i) {
    // The gain has gain_bits fractional bits, and filterbank_output[i] has
    // -input_correction_bits fractional bits. The product is shifted so that
    // the resulting snr has kPcanSnrBits fractional bits.
    const uint32_t gain = WideDynamicFunction(noise_estimate[i], gain_lut);
    const uint32_t snr = ((uint64_t)filterbank_output[i] * gain) >> snr_shift;
    // Result has kPcanOutputBits fractional bits.
    // NOTE: This assumes filterbank_output_scale = 1 << kPcanOutputBits.
    filterbank_output[i] = PcanShrink(snr);
  }
}

}  // namespace tflm_signal
}  // namespace tflite
