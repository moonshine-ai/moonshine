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

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_

#include <cstdint>
#include <sstream>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"

namespace knf {

struct RawAudioSamplesOptions {
  FrameExtractionOptions frame_opts;

  RawAudioSamplesOptions() {
    frame_opts.window_type = "rectangular";
    frame_opts.dither = 0;
    frame_opts.preemph_coeff = 0;
    frame_opts.remove_dc_offset = false;
    frame_opts.round_to_power_of_two = false;
    frame_opts.snip_edges = true;
  }

  std::string ToString() const {
    std::ostringstream os;
    os << "frame_opts: \n";
    os << frame_opts << "\n";
    os << "\n";
    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const RawAudioSamplesOptions &opts);

class RawAudioSamplesComputer {
 public:
  using Options = RawAudioSamplesOptions;

  explicit RawAudioSamplesComputer(const RawAudioSamplesOptions &opts)
      : opts_(opts){};

  int32_t Dim() const {
    return opts_.frame_opts.frame_length_ms * opts_.frame_opts.samp_freq / 1000;
  }

  bool NeedRawLogEnergy() const { return false; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  const RawAudioSamplesOptions &GetOptions() const { return opts_; }

  void Compute(float unused_signal_raw_log_energy, float unused_vtln_warp,
               std::vector<float> *signal_frame, float *feature);

 private:
  RawAudioSamplesOptions opts_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_RAW_AUDIO_SAMPLES_H_
