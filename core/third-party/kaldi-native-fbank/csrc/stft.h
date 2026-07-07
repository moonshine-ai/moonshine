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

#ifndef KALDI_NATIVE_FBANK_CSRC_STFT_H_
#define KALDI_NATIVE_FBANK_CSRC_STFT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace knf {

struct StftConfig {
  int32_t n_fft;  // should be a power of two
  int32_t hop_length;
  int32_t win_length;
  std::string window_type;
  bool center = true;
  std::string pad_mode = "reflect";
  bool normalized = false;

  // if it is specified, then window_type is ignored
  std::vector<float> window;

  std::string ToString() const;
};

struct StftResult {
  // [num_frames, n_fft/2+1], flattened in row major
  std::vector<float> real;
  std::vector<float> imag;
  int32_t num_frames;
};

class Stft {
 public:
  explicit Stft(const StftConfig &config);
  ~Stft();
  StftResult Compute(const float *data, int32_t n) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_STFT_H_
        //
