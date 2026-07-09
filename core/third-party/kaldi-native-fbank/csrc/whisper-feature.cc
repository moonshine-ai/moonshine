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

#include "kaldi-native-fbank/csrc/whisper-feature.h"

#include <cmath>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/log.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"

#ifndef M_2PI
#define M_2PI 6.283185307179586476925286766559005
#endif

namespace knf {

std::string WhisperFeatureOptions::ToString() const {
  std::ostringstream os;
  os << "WhisperFeatureOptions(";
  os << "frame_opts=" << frame_opts.ToString() << ", ";
  os << "dim=" << dim << ")";
  return os.str();
}

static void dft(const std::vector<float> &in, std::vector<float> *out) {
  // this function is modified from
  // https://github.com/ggerganov/whisper.cpp/blob/master/whisper.cpp#L2353
  int32_t N = in.size();

  out->resize(N * 2);

  auto M_2PI_over_N = M_2PI / N;
  for (int32_t k = 0; k < N; ++k) {
    float re = 0;
    float im = 0;

    for (int32_t n = 0; n < N; ++n) {
      float angle = M_2PI_over_N * k * n;
      re += in[n] * cos(angle);
      im -= in[n] * sin(angle);
    }

    (*out)[k * 2 + 0] = re;
    (*out)[k * 2 + 1] = im;
  }
}

// Cooley-Tukey FFT
// poor man's implementation - use something better
// input is real-valued
// output is complex-valued
static void fft(const std::vector<float> &in, std::vector<float> *out) {
  // this function is copied from
  // https://github.com/ggerganov/whisper.cpp/blob/master/whisper.cpp#L2373C1-L2429C1

  int32_t N = in.size();
  out->resize(N * 2);

  if (N == 1) {
    (*out)[0] = in[0];
    (*out)[1] = 0;
    return;
  }

  if (N % 2 == 1) {
    dft(in, out);
    return;
  }

  std::vector<float> even;
  std::vector<float> odd;

  even.reserve(N / 2);
  odd.reserve(N / 2);

  for (int32_t i = 0; i != N; ++i) {
    if (i % 2 == 0) {
      even.push_back(in[i]);
    } else {
      odd.push_back(in[i]);
    }
  }

  std::vector<float> even_fft;
  std::vector<float> odd_fft;

  fft(even, &even_fft);
  fft(odd, &odd_fft);

  for (int32_t k = 0; k < N / 2; ++k) {
    float theta = M_2PI * k / N;

    float re = cos(theta);
    float im = -sin(theta);

    float re_odd = odd_fft[2 * k + 0];
    float im_odd = odd_fft[2 * k + 1];

    (*out)[2 * k + 0] = even_fft[2 * k + 0] + re * re_odd - im * im_odd;
    (*out)[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

    (*out)[2 * (k + N / 2) + 0] =
        even_fft[2 * k + 0] - re * re_odd + im * im_odd;
    (*out)[2 * (k + N / 2) + 1] =
        even_fft[2 * k + 1] - re * im_odd - im * re_odd;
  }
}

WhisperFeatureComputer::WhisperFeatureComputer(
    const WhisperFeatureOptions &opts /*= {}*/)
    : opts_(opts) {
  opts_.frame_opts.samp_freq = 16000;
  opts_.frame_opts.frame_shift_ms = 10;
  opts_.frame_opts.frame_length_ms = 25;
  opts_.frame_opts.dither = 0;
  opts_.frame_opts.preemph_coeff = 0;
  opts_.frame_opts.remove_dc_offset = false;
  opts_.frame_opts.window_type = "hann";
  opts_.frame_opts.round_to_power_of_two = false;
  opts_.frame_opts.snip_edges = false;

  MelBanksOptions mel_opts;
  mel_opts.num_bins = opts_.dim;
  mel_opts.low_freq = 0;
  mel_opts.is_librosa = true;

  mel_banks_ = std::make_unique<MelBanks>(mel_opts, opts_.frame_opts, 1.0f);
}

void WhisperFeatureComputer::Compute(float /*signal_raw_log_energy*/,
                                     float /*vtln_warp*/,
                                     std::vector<float> *signal_frame,
                                     float *feature) {
  KNF_CHECK_EQ(signal_frame->size(), frame_opts_.PaddedWindowSize());
  // we have already applied window function to signal_frame before
  // calling this method
  std::vector<float> fft_out;
  fft(*signal_frame, &fft_out);

  int32_t num_fft = signal_frame->size();
  std::vector<float> power(num_fft / 2 + 1);
  for (int32_t i = 0; i <= num_fft / 2; ++i) {
    float re = fft_out[2 * i + 0];
    float im = fft_out[2 * i + 1];
    power[i] = re * re + im * im;
  }

  // feature is pre-allocated by the user
  mel_banks_->Compute(power.data(), feature);
}

}  // namespace knf
