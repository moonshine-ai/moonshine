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
#include "kaldi-native-fbank/csrc/istft.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <utility>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

class IStft::Impl {
 public:
  explicit Impl(const StftConfig &config) : config_(config) {
    if (!config.window.empty()) {
      window_ = std::make_unique<FeatureWindowFunction>(config_.window);
    } else if (!config.window_type.empty()) {
      window_ = std::make_unique<FeatureWindowFunction>(config.window_type,
                                                        config.win_length);
    }
  }

  std::vector<float> Compute(const StftResult &stft_result) const {
    Rfft rfft(config_.n_fft, true);

    int32_t num_samples =
        config_.n_fft + (stft_result.num_frames - 1) * config_.hop_length;

    std::vector<float> samples(num_samples);
    for (int32_t i = 0; i < stft_result.num_frames; ++i) {
      auto x = InverseFFT(stft_result, i, &rfft);
      OverlapAdd(std::move(x), i, &samples);
    }

    auto denominator = GetDenominator(stft_result.num_frames);

    for (int32_t i = 0; i < num_samples; ++i) {
      if (denominator[i]) {
        samples[i] /= denominator[i];
      }
    }

    if (config_.center) {
      samples = {samples.begin() + config_.n_fft / 2,
                 samples.end() - config_.n_fft / 2};
    }

    return samples;
  }

  std::vector<float> InverseFFT(const StftResult &r, int32_t frame_index,
                                Rfft *rfft) const {
    int32_t n_fft = config_.n_fft;
    int32_t hop_length = config_.hop_length;

    const float *p_real = r.real.data() + frame_index * (n_fft / 2 + 1);
    const float *p_imag = r.imag.data() + frame_index * (n_fft / 2 + 1);
    std::vector<float> tmp(n_fft);

    float scale = 1;
    if (config_.normalized) {
      scale = std::sqrt(n_fft);
    }

    for (int32_t i = 0; i < n_fft / 2; ++i) {
      if (i == 0) {
        tmp[0] = p_real[0] * scale;
        tmp[1] = p_real[n_fft / 2] * scale;
      } else {
        tmp[2 * i] = p_real[i] * scale;
        tmp[2 * i + 1] = p_imag[i] * scale;
      }
    }

    rfft->Compute(tmp.data());

    scale = 1.0f / n_fft;
    for (auto &f : tmp) {
      f *= scale;
    }

    return tmp;
  }

  void OverlapAdd(std::vector<float> current_frame, int32_t frame_index,
                  std::vector<float> *samples) const {
    if (window_) {
      window_->Apply(current_frame.data());
    }

    float *p = samples->data() + frame_index * config_.hop_length;
    for (int32_t i = 0; i < config_.n_fft; ++i) {
      p[i] += current_frame[i];
    }
  }

  std::vector<float> GetDenominator(int32_t num_frames) const {
    int32_t num_samples = config_.n_fft + (num_frames - 1) * config_.hop_length;
    std::vector<float> ans(num_samples);
    if (!window_) {
      for (int32_t i = 0; i < num_frames; ++i) {
        int32_t start = i * config_.hop_length;
        int32_t end = start + config_.n_fft;

        for (int32_t k = start; k < end; ++k) {
          ans[k] += 1;
        }
      }
    } else {
      const auto &w = window_->GetWindow();
      for (int32_t i = 0; i < num_frames; ++i) {
        int32_t start = i * config_.hop_length;
        int32_t end = start + config_.n_fft;

        auto pw = w.data();
        for (int32_t k = start; k < end; ++k, ++pw) {
          ans[k] += (*pw) * (*pw);
        }
      }
    }

    return ans;
  }

 private:
  StftConfig config_;
  std::unique_ptr<FeatureWindowFunction> window_;
};

IStft::IStft(const StftConfig &config)
    : impl_(std::make_unique<Impl>(config)) {}

IStft::~IStft() = default;

std::vector<float> IStft::Compute(const StftResult &stft_result) const {
  return impl_->Compute(stft_result);
}

}  // namespace knf
