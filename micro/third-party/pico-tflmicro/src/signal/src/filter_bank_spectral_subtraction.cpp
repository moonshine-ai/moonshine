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

#include "signal/src/filter_bank_spectral_subtraction.h"

namespace tflite {
namespace tflm_signal {

void FilterbankSpectralSubtraction(const SpectralSubtractionConfig* config,
                                   const uint32_t* input, uint32_t* output,
                                   uint32_t* noise_estimate) {
  const bool data_clamping = config->clamping;
  const int smoothing_bits = config->smoothing_bits;
  const int num_channels = config->num_channels;

  for (int i = 0; i < num_channels; ++i) {
    uint32_t smoothing;
    uint32_t one_minus_smoothing;
    if ((i & 1) == 0) {
      smoothing = config->smoothing;
      one_minus_smoothing = config->one_minus_smoothing;
    } else {  // Use alternate smoothing coefficient on odd-index channels.
      smoothing = config->alternate_smoothing;
      one_minus_smoothing = config->alternate_one_minus_smoothing;
    }

    // Scale up signal[i] for smoothing filter computation.
    const uint32_t signal_scaled_up = input[i] << smoothing_bits;
    noise_estimate[i] =
        ((static_cast<uint64_t>(signal_scaled_up) * smoothing) +
         (static_cast<uint64_t>(noise_estimate[i]) * one_minus_smoothing)) >>
        config->spectral_subtraction_bits;

    uint32_t estimate_scaled_up = noise_estimate[i];
    // Make sure that we can't get a negative value for the signal - estimate.
    if (estimate_scaled_up > signal_scaled_up) {
      estimate_scaled_up = signal_scaled_up;
      if (data_clamping) {
        noise_estimate[i] = estimate_scaled_up;
      }
    }
    const uint32_t floor =
        (static_cast<uint64_t>(input[i]) * config->min_signal_remaining) >>
        config->spectral_subtraction_bits;
    const uint32_t subtracted =
        (signal_scaled_up - estimate_scaled_up) >> smoothing_bits;
    output[i] = subtracted > floor ? subtracted : floor;
  }
}

}  // namespace tflm_signal
}  // namespace tflite
