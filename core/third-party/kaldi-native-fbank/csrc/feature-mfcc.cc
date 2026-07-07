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

// This file is copied/modified from kaldi/src/feat/feature-mfcc.cc
//
#include "kaldi-native-fbank/csrc/feature-mfcc.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-functions.h"
#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/kaldi-math.h"
#include "kaldi-native-fbank/csrc/log.h"

namespace knf {

static std::vector<float> ComputeDctMatrix(int32_t num_rows, int32_t num_cols) {
  // this function is copied from
  // https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/matrix-functions.cc#L592

  std::vector<float> ans(num_rows * num_cols);
  float *p = ans.data();

  float normalizer = std::sqrt(1.0 / num_cols);  // normalizer for X_0

  for (int32_t i = 0; i != num_cols; ++i) {
    p[i] = normalizer;
  }

  normalizer = std::sqrt(2.0 / num_cols);  // normalizer for other elements

  for (int32_t k = 1; k != num_rows; ++k) {
    for (int32_t n = 0; n != num_cols; ++n) {
      *(p + k * num_cols + n) =
          normalizer *
          std::cos(static_cast<double>(M_PI) / num_cols * (n + 0.5) * k);
    }
  }

  return ans;
}

std::ostream &operator<<(std::ostream &os, const MfccOptions &opts) {
  os << opts.ToString();
  return os;
}

MfccComputer::MfccComputer(const MfccOptions &opts)
    : opts_(opts),
      rfft_(opts.frame_opts.PaddedWindowSize()),
      mel_energies_(opts.mel_opts.num_bins) {
  if (opts.energy_floor > 0.0f) {
    log_energy_floor_ = logf(opts.energy_floor);
  }

  // We'll definitely need the filterbanks info for VTLN warping factor 1.0.
  // [note: this call caches it.]
  GetMelBanks(1.0f);

  int32_t num_bins = opts.mel_opts.num_bins;

  KNF_CHECK_LE(opts.num_ceps, num_bins)
      << "num-ceps cannot be larger than num-mel-bins."
      << " It should be smaller or equal. You provided num-ceps: "
      << opts.num_ceps << "  and num-mel-bins: " << num_bins;

  dct_matrix_ = ComputeDctMatrix(opts.num_ceps, num_bins);

  if (opts.cepstral_lifter != 0.0) {
    lifter_coeffs_ = std::vector<float>(opts.num_ceps);
    // torch::empty({1, opts.num_ceps}, torch::kFloat32);
    ComputeLifterCoeffs(opts.cepstral_lifter, &lifter_coeffs_);
  }
}

MfccComputer::~MfccComputer() {
  for (auto iter = mel_banks_.begin(); iter != mel_banks_.end(); ++iter)
    delete iter->second;
}

const MelBanks *MfccComputer::GetMelBanks(float vtln_warp) {
  MelBanks *this_mel_banks = nullptr;

  // std::map<float, MelBanks *>::iterator iter = mel_banks_.find(vtln_warp);
  auto iter = mel_banks_.find(vtln_warp);
  if (iter == mel_banks_.end()) {
    this_mel_banks = new MelBanks(opts_.mel_opts, opts_.frame_opts, vtln_warp);
    mel_banks_[vtln_warp] = this_mel_banks;
  } else {
    this_mel_banks = iter->second;
  }
  return this_mel_banks;
}

void MfccComputer::Compute(float signal_raw_log_energy, float vtln_warp,
                           std::vector<float> *signal_frame, float *feature) {
  const MelBanks &mel_banks = *(GetMelBanks(vtln_warp));

  KNF_CHECK_EQ(signal_frame->size(), opts_.frame_opts.PaddedWindowSize());

  // Compute energy after window function (not the raw one).
  if (opts_.use_energy && !opts_.raw_energy) {
    signal_raw_log_energy = std::log(
        std::max<float>(InnerProduct(signal_frame->data(), signal_frame->data(),
                                     signal_frame->size()),
                        std::numeric_limits<float>::epsilon()));
  }
  rfft_.Compute(signal_frame->data());  // signal_frame is modified in-place
  ComputePowerSpectrum(signal_frame);

  // Sum with mel filter banks over the power spectrum
  mel_banks.Compute(signal_frame->data(), mel_energies_.data());

  // Avoid log of zero (which should be prevented anyway by dithering).
  for (int32_t i = 0; i != opts_.mel_opts.num_bins; ++i) {
    auto t = std::max(mel_energies_[i], std::numeric_limits<float>::epsilon());
    mel_energies_[i] = std::log(t);
  }

  // feature = dct_matrix_ * mel_energies [which now have log]
  for (int32_t i = 0; i != opts_.num_ceps; ++i) {
    feature[i] = InnerProduct(dct_matrix_.data() + i * opts_.mel_opts.num_bins,
                              mel_energies_.data(), opts_.mel_opts.num_bins);
  }

  if (opts_.cepstral_lifter != 0.0) {
    for (int32_t i = 0; i != opts_.num_ceps; ++i) {
      feature[i] *= lifter_coeffs_[i];
    }
  }

  if (opts_.use_energy) {
    if (opts_.energy_floor > 0.0 && signal_raw_log_energy < log_energy_floor_) {
      signal_raw_log_energy = log_energy_floor_;
    }
    feature[0] = signal_raw_log_energy;
  }

  if (opts_.htk_compat) {
    double energy = feature[0];

    for (int32_t i = 0; i < opts_.num_ceps - 1; ++i) {
      feature[i] = feature[i + 1];
    }

    if (!opts_.use_energy) {
      energy *= M_SQRT2;  // scale on C0 (actually removing a scale
    }
    // we previously added that's part of one common definition of
    // the cosine transform.)
    feature[opts_.num_ceps - 1] = static_cast<float>(energy);
  }
}

}  // namespace knf
