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
#include "kaldi-native-fbank/csrc/stft.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {
std::string StftConfig::ToString() const {
  std::ostringstream os;
  os << "StftConfig(";
  os << "n_fft=" << n_fft << ", ";
  os << "hop_length=" << hop_length << ", ";
  os << "win_length=" << win_length << ", ";
  os << "window_type=\"" << window_type << "\", ";
  os << "center=" << (center ? "True" : "False") << ", ";
  os << "pad_mode=\"" << pad_mode << "\", ";
  os << "normalized=" << (normalized ? "True" : "False") << ")";
  return os.str();
}

class Stft::Impl {
 public:
  explicit Impl(const StftConfig &config) : config_(config) {
    if (!config.window.empty()) {
      window_ = std::make_unique<FeatureWindowFunction>(config.window);
    } else if (!config.window_type.empty()) {
      window_ = std::make_unique<FeatureWindowFunction>(config.window_type,
                                                        config.win_length);
    }
  }

  StftResult Compute(const float *data, int32_t n) const {
    int32_t n_fft = config_.n_fft;
    int32_t hop_length = config_.hop_length;

    std::vector<float> samples;
    const float *p = data;

    if (config_.center) {
      samples = Pad(data, n);
      p = samples.data();
      n = samples.size();
    }

    int64_t num_frames = 1 + (n - n_fft) / hop_length;

    Rfft rfft(n_fft);

    StftResult ans;
    ans.num_frames = num_frames;
    ans.real.resize(num_frames * (n_fft / 2 + 1));
    ans.imag.resize(num_frames * (n_fft / 2 + 1));

    std::vector<float> tmp(config_.n_fft);

    for (int32_t i = 0; i < num_frames; ++i) {
      tmp = {p + i * hop_length, p + i * hop_length + n_fft};
      if (window_) {
        window_->Apply(tmp.data());
      }

      rfft.Compute(tmp.data());

      for (int32_t k = 0; k < n_fft / 2; ++k) {
        if (k == 0) {
          ans.real[i * (n_fft / 2 + 1)] = tmp[0];
          ans.real[i * (n_fft / 2 + 1) + n_fft / 2] = tmp[1];
        } else {
          ans.real[i * (n_fft / 2 + 1) + k] = tmp[2 * k];

          ans.imag[i * (n_fft / 2 + 1) + k] = tmp[2 * k + 1];
        }
      }
    }

    if (config_.normalized) {
      float scale = 1 / std::sqrt(n_fft);
      for (int32_t i = 0; i < ans.real.size(); ++i) {
        ans.real[i] *= scale;
        ans.imag[i] *= scale;
      }
    }

    return ans;
  }

  std::vector<float> Pad(const float *data, int32_t n) const {
    int32_t pad_amount = config_.n_fft / 2;
    std::vector<float> ans(n + config_.n_fft);
    std::copy(data, data + n, ans.begin() + pad_amount);
    if (config_.pad_mode == "constant") {
      // do nothing
    } else if (config_.pad_mode == "reflect") {
      // left
      std::copy(data + 1, data + 1 + pad_amount, ans.rend() - pad_amount);
      std::copy(data + n - pad_amount - 1, data + n - 1, ans.rbegin());
    } else if (config_.pad_mode == "replicate") {
      std::fill(ans.begin(), ans.begin() + pad_amount, data[0]);
      std::fill(ans.end() - pad_amount, ans.end(), data[n - 1]);
    } else {
      fprintf(stderr, "Unsupported pad_mode: '%s'. Use 0 padding\n",
              config_.pad_mode.c_str());
    }

    return ans;
  }

 private:
  StftConfig config_;
  std::unique_ptr<FeatureWindowFunction> window_;
};

Stft::Stft(const StftConfig &config) : impl_(std::make_unique<Impl>(config)) {}

Stft::~Stft() = default;

StftResult Stft::Compute(const float *data, int32_t n) const {
  return impl_->Compute(data, n);
}

}  // namespace knf
