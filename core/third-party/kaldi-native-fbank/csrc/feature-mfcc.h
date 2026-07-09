/**
 * Copyright 2009-2011  Karel Vesely;  Petr Motlicek;  Saarland University
 *           2014-2016  Johns Hopkins University (author: Daniel Povey)
 * Copyright 2024       Xiaomi Corporation (authors: Fangjun Kuang)
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

// This file is copied/modified from kaldi/src/feat/feature-mfcc.h

#ifndef KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_
#define KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_

#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "kaldi-native-fbank/csrc/feature-window.h"
#include "kaldi-native-fbank/csrc/mel-computations.h"
#include "kaldi-native-fbank/csrc/rfft.h"

namespace knf {

/// MfccOptions contains basic options for computing MFCC features.
// (this class is copied from kaldi)
struct MfccOptions {
  FrameExtractionOptions frame_opts;
  MelBanksOptions mel_opts;

  // Number of cepstra in MFCC computation (including C0)
  int32_t num_ceps = 13;

  // Use energy (not C0) in MFCC computation
  bool use_energy = true;

  // Floor on energy (absolute, not relative) in MFCC
  // computation. Only makes a difference if use_energy=true;
  // only necessary if dither=0.0.
  // Suggested values: 0.1 or 1.0
  float energy_floor = 0.0;

  // If true, compute energy before preemphasis and windowing
  bool raw_energy = true;

  // Constant that controls scaling of MFCCs
  float cepstral_lifter = 22.0;

  // If true, put energy or C0 last and use a factor of
  // sqrt(2) on C0.
  // Warning: not sufficient to get HTK compatible features
  // (need to change other parameters)
  bool htk_compat = false;

  MfccOptions() { mel_opts.num_bins = 23; }

  std::string ToString() const {
    std::ostringstream os;
    os << "MfccOptions(";
    os << "frame_opts=" << frame_opts.ToString() << ", ";
    os << "mel_opts=" << mel_opts.ToString() << ", ";

    os << "num_ceps=" << num_ceps << ", ";
    os << "use_energy=" << (use_energy ? "True" : "False") << ", ";
    os << "energy_floor=" << energy_floor << ", ";
    os << "raw_energy=" << (raw_energy ? "True" : "False") << ", ";
    os << "cepstral_lifter=" << cepstral_lifter << ", ";
    os << "htk_compat=" << (htk_compat ? "True" : "False") << ")";

    return os.str();
  }
};

std::ostream &operator<<(std::ostream &os, const MfccOptions &opts);

class MfccComputer {
 public:
  using Options = MfccOptions;

  explicit MfccComputer(const MfccOptions &opts);
  ~MfccComputer();

  int32_t Dim() const { return opts_.num_ceps; }

  // if true, compute log_energy_pre_window but after dithering and dc removal
  bool NeedRawLogEnergy() const { return opts_.use_energy && opts_.raw_energy; }

  const FrameExtractionOptions &GetFrameOptions() const {
    return opts_.frame_opts;
  }

  const MfccOptions &GetOptions() const { return opts_; }

  /**
     Function that computes one frame of features from
     one frame of signal.

     @param [in] signal_raw_log_energy The log-energy of the frame of the signal
         prior to windowing and pre-emphasis, or
         log(numeric_limits<float>::min()), whichever is greater.  Must be
         ignored by this function if this class returns false from
         this->NeedsRawLogEnergy().
     @param [in] vtln_warp  The VTLN warping factor that the user wants
         to be applied when computing features for this utterance.  Will
         normally be 1.0, meaning no warping is to be done.  The value will
         be ignored for feature types that don't support VLTN, such as
         spectrogram features.
     @param [in] signal_frame  One frame of the signal,
       as extracted using the function ExtractWindow() using the options
       returned by this->GetFrameOptions().  The function will use the
       vector as a workspace, which is why it's a non-const pointer.
     @param [out] feature  Pointer to a vector of size this->Dim(), to which
         the computed feature will be written. It should be pre-allocated.
  */
  void Compute(float signal_raw_log_energy, float vtln_warp,
               std::vector<float> *signal_frame, float *feature);

 private:
  const MelBanks *GetMelBanks(float vtln_warp);

  MfccOptions opts_;
  float log_energy_floor_;
  std::map<float, MelBanks *> mel_banks_;  // float is VTLN coefficient.
  Rfft rfft_;

  // temp buffer of size num_mel_bins = opts.mel_opts.num_bins
  std::vector<float> mel_energies_;

  // opts_.num_ceps
  std::vector<float> lifter_coeffs_;

  // [num_ceps][num_mel_bins]
  std::vector<float> dct_matrix_;
};

}  // namespace knf

#endif  // KALDI_NATIVE_FBANK_CSRC_FEATURE_MFCC_H_
