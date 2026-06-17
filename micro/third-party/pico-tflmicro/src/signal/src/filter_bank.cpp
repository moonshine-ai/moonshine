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

#include "signal/src/filter_bank.h"

namespace tflite {
namespace tflm_signal {

void FilterbankAccumulateChannels(const FilterbankConfig* config,
                                  const uint32_t* input, uint64_t* output) {
  // With a log mel filterbank, the energy at each frequency gets added to
  // two adjacent filterbank filters/channels.
  // For the first filter bank channel, its energy is first multiplied by
  // some weight 'w', then gets accumulated.
  // For the subsequent filter bank, its power is first multiplied by 1-'w'
  // (called unweight here), then gets accumulated.
  // For this reason, we need to calculate (config->num_channels + 1) output
  // where element 0 is only used as scratch storage for the unweights of
  // element 1 (channel 0). The caller should discard element 0.
  // Writing the code like this doesn't save multiplications, but it lends
  // itself better to optimization, because input[freq_start + j] only needs
  // to be loaded once.
  uint64_t weight_accumulator = 0;
  uint64_t unweight_accumulator = 0;
  for (int i = 0; i < config->num_channels + 1; i++) {
    const int16_t freq_start = config->channel_frequency_starts[i];
    const int16_t weight_start = config->channel_weight_starts[i];
    for (int j = 0; j < config->channel_widths[i]; ++j) {
      weight_accumulator += config->weights[weight_start + j] *
                            static_cast<uint64_t>(input[freq_start + j]);
      unweight_accumulator += config->unweights[weight_start + j] *
                              static_cast<uint64_t>(input[freq_start + j]);
    }
    output[i] = weight_accumulator;
    weight_accumulator = unweight_accumulator;
    unweight_accumulator = 0;
  }
}

}  // namespace tflm_signal
}  // namespace tflite