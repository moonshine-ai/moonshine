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

#ifndef SIGNAL_SRC_FILTER_BANK_H_
#define SIGNAL_SRC_FILTER_BANK_H_

#include <stdint.h>

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above

struct FilterbankConfig {
  // Number of filterbank channels
  int32_t num_channels;

  // Each of the following three arrays is of size num_channels + 1
  // An extra channel is needed for scratch. See implementation of
  // FilterbankAccumulateChannels() for more details

  // For each channel, the index in the input (spectrum) where its band starts
  const int16_t* channel_frequency_starts;
  // For each channel, the index in the weights/unweights arrays where
  // it filter weights start
  const int16_t* channel_weight_starts;
  // For each channel, the number of bins in the input (spectrum) that span
  // its band
  const int16_t* channel_widths;

  // The weights array holds the triangular filter weights of all the filters
  // in the bank. The output of each filter in the bank is caluclated by
  // multiplying the elements in the input spectrum that are in its band
  // (see above: channel_frequency_starts, channel_widths) by the filter weights
  // then accumulating. Each element in the unweights array holds the 1 minus
  // corresponding elements in the weights array and is used to make this
  // operation more efficient. For more details, see documnetation in
  // FilterbankAccumulateChannels()
  const int16_t* weights;
  const int16_t* unweights;
  int32_t output_scale;

  int32_t input_correction_bits;
};

// Accumulate the energy spectrum bins in `input` into filter bank channels
// contained in `output`.
// * `input` - Spectral energy array
// * `output` - of size `config.num_channels` + 1.
//              Elements [1:num_channels] contain the filter bank channels.
//              Element 0 is used as scratch and should be ignored
void FilterbankAccumulateChannels(const FilterbankConfig* config,
                                  const uint32_t* input, uint64_t* output);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_SRC_FILTER_BANK_H_
