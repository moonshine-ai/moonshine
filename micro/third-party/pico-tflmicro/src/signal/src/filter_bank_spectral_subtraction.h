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

#ifndef SIGNAL_SRC_FILTER_BANK_SPECTRAL_SUBTRACTION_H_
#define SIGNAL_SRC_FILTER_BANK_SPECTRAL_SUBTRACTION_H_

#include <stdint.h>

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above

struct SpectralSubtractionConfig {
  // Number of filterbank channels in input and output
  int32_t num_channels;
  // The constant used for the lowpass filter for finding the noise.
  // Higher values correspond to more aggressively adapting estimates
  // of the noise.
  // Scale is 1 << spectral_subtraction_bits
  uint32_t smoothing;
  // One minus smoothing constant for low pass filter.
  // Scale is 1 << spectral_subtraction_bits
  uint32_t one_minus_smoothing;
  // The maximum cap to subtract away from the signal (ie, if this is
  // 0.2, then the result of spectral subtraction will not go below
  // 0.2 * signal).
  //  Scale is 1 << spectral_subtraction_bits
  uint32_t min_signal_remaining;
  // If positive, specifies the filter coefficient for odd-index
  // channels, while 'smoothing' is used as the coefficient for even-
  // index channels. Otherwise, the same filter coefficient is
  // used on all channels.
  //  Scale is 1 << spectral_subtraction_bits
  uint32_t alternate_smoothing;
  // Alternate One minus smoothing constant for low pass filter.
  // Scale is 1 << spectral_subtraction_bits
  uint32_t alternate_one_minus_smoothing;
  // Extra fractional bits for the noise_estimate smoothing filter.
  uint32_t smoothing_bits;
  // Scaling bits for some members of this struct
  uint32_t spectral_subtraction_bits;
  // If true, when the filterbank level drops below the output,
  // the noise estimate will be forced down to the new noise level.
  // If false, the noise estimate will remain above the current
  // filterbank output (but the subtraction will still keep the
  // output non negative).
  bool clamping;
};

// Apply spectral subtraction to each element in `input`, then write the result
// to `output` and `noise_estimate`. `input`, `output` and `noise estimate`
// must all be of size `config.num_channels`. `config` holds the
//  parameters of the spectral subtraction algorithm.
void FilterbankSpectralSubtraction(const SpectralSubtractionConfig* config,
                                   const uint32_t* input, uint32_t* output,
                                   uint32_t* noise_estimate);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_SRC_FILTER_BANK_SPECTRAL_SUBTRACTION_H_
