/**
 * Copyright (c)  2023  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_
#define KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"

namespace knf {

struct WhisperFeatureOptions {
  WhisperFeatureOptions(const FrameExtractionOptions &frame_opts = {},
                        int32_t dim = 80)
      : frame_opts(frame_opts), dim(dim) {}

  FrameExtractionOptions frame_opts;
  int32_t dim = 80;

  std::string ToString() const;
};

class WhisperFeatureComputer {
 public:
  // note: opts.frame_opts is ignored and we reset it inside
  explicit WhisperFeatureComputer(const WhisperFeatureOptions &opts = {});

  int32_t Dim() const { return opts_.dim; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  void Compute(float /*signal_raw_log_energy*/, float /*vtln_warp*/,
               std::vector<float> *signal_frame, float *feature);

  // if true, compute log_energy_pre_window but after dithering and dc removal
  bool NeedRawLogEnergy() const { return false; }

  using Options = WhisperFeatureOptions;

 private:
  std::unique_ptr<MelBanks> mel_banks_;
  WhisperFeatureOptions opts_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_WHISPER_FEATURE_H_
