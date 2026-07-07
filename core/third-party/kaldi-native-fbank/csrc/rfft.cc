/**
 * Copyright (c)  2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include "kaldi-native-fbank/csrc/rfft.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "kaldi-native-fbank/csrc/log.h"
#include "kiss_fftr.h"

namespace knf {

class Rfft::RfftImpl {
 public:
  RfftImpl(int32_t n, bool inverse) : n_(n), inverse_(inverse) {
    if ((n & 1) != 0) {
      fprintf(stderr, "n should be even. Given: %d \n", n);
      exit(-1);
    }

    if (n < 0) {
      fprintf(stderr, "n should be positive. Given: %d \n", n);
      exit(-1);
    }
  }

  void Compute(float *in_out) {
    if (!inverse_) {
      Forward(in_out);
    } else {
      Reverse(in_out);
    }
  }

  void Compute(double *in_out) {
    std::vector<float> f(in_out, in_out + n_);

    Compute(f.data());

    std::copy(f.begin(), f.end(), in_out);
  }

 private:
  void Forward(float *in_out) const {
    kiss_fftr_cfg cfg = kiss_fftr_alloc(n_, 0, nullptr, nullptr);

    std::vector<kiss_fft_cpx> out(n_ / 2 + 1);

    kiss_fftr(cfg, in_out, out.data());

    kiss_fftr_free(cfg);

    in_out[0] = out[0].r;
    in_out[1] = out[n_ / 2].r;

    for (int32_t i = 1; i < n_ / 2; ++i) {
      in_out[2 * i] = out[i].r;
      in_out[2 * i + 1] = out[i].i;
    }
  }

  void Reverse(float *in_out) const {
    std::vector<kiss_fft_cpx> out(n_ / 2 + 1);
    out[0].r = in_out[0];
    out[0].i = 0;

    out[n_ / 2].r = in_out[1];
    out[n_ / 2].i = 0;

    for (int32_t i = 1; i < n_ / 2; ++i) {
      out[i].r = in_out[2 * i];
      out[i].i = in_out[2 * i + 1];
    }

    kiss_fftr_cfg cfg = kiss_fftr_alloc(n_, 1, nullptr, nullptr);

    kiss_fftri(cfg, out.data(), in_out);

    kiss_fftr_free(cfg);
  }

 private:
  int32_t n_;
  bool inverse_ = false;
};

Rfft::Rfft(int32_t n, bool inverse /*=false*/)
    : impl_(std::make_unique<RfftImpl>(n, inverse)) {}

Rfft::~Rfft() = default;

void Rfft::Compute(float *in_out) { impl_->Compute(in_out); }
void Rfft::Compute(double *in_out) { impl_->Compute(in_out); }

}  // namespace knf
