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

#ifndef KALDI_NATIVE_FBANK_CSRC_ISTFT_H_
#define KALDI_NATIVE_FBANK_CSRC_ISTFT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/stft.h"

namespace knf {

class IStft {
 public:
  explicit IStft(const StftConfig &config);
  ~IStft();
  std::vector<float> Compute(const StftResult &stft_result) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_ISTFT_H_
        //
