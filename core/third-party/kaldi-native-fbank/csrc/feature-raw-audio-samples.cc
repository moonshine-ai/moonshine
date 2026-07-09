/**
 * Copyright (c)  2025  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "kaldi-native-fbank/csrc/feature-raw-audio-samples.h"

#include <algorithm>
#include <vector>

namespace knf {

std::ostream &operator<<(std::ostream &os, const RawAudioSamplesOptions &opts) {
  os << opts.ToString();
  return os;
}

void RawAudioSamplesComputer::Compute(float /*signal_raw_log_energy*/,
                                      float /*vtln_warp*/,
                                      std::vector<float> *signal_frame,
                                      float *feature) {
  std::copy(signal_frame->begin(), signal_frame->end(), feature);
}

}  // namespace knf
