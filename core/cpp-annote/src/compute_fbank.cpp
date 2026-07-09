// SPDX-License-Identifier: MIT

#include "compute_fbank.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "kaldi-native-fbank/csrc/online-feature.h"

namespace cppannote::fbank {

void wespeaker_like_fbank(float sample_hz, int num_mel_bins,
                          float frame_length_ms, float frame_shift_ms,
                          const float* mono, int num_samples,
                          std::vector<float>& out_rowmajor, int& num_frames,
                          int& mel_dim_out) {
  knf::FbankOptions opts;
  opts.frame_opts.samp_freq = sample_hz;
  opts.frame_opts.frame_length_ms = frame_length_ms;
  opts.frame_opts.frame_shift_ms = frame_shift_ms;
  opts.frame_opts.dither = 0.f;
  opts.frame_opts.window_type = "hamming";
  opts.frame_opts.snip_edges = true;
  opts.mel_opts.num_bins = num_mel_bins;
  opts.use_energy = false;
  opts.use_log_fbank = true;

  knf::OnlineFbank fbank(opts);
  const float scale = static_cast<float>(1 << 15);
  std::vector<float> scaled(static_cast<std::size_t>(std::max(0, num_samples)));
  for (int i = 0; i < num_samples; ++i) {
    scaled[static_cast<std::size_t>(i)] = mono[i] * scale;
  }
  if (num_samples > 0) {
    fbank.AcceptWaveform(sample_hz, scaled.data(),
                         static_cast<int32_t>(num_samples));
  }
  fbank.InputFinished();
  num_frames = fbank.NumFramesReady();
  mel_dim_out = fbank.Dim();
  if (num_frames <= 0 || mel_dim_out <= 0) {
    out_rowmajor.clear();
    num_frames = 0;
    mel_dim_out = 0;
    return;
  }
  out_rowmajor.assign(static_cast<std::size_t>(num_frames) *
                          static_cast<std::size_t>(mel_dim_out),
                      0.f);
  for (int t = 0; t < num_frames; ++t) {
    const float* row = fbank.GetFrame(t);
    std::memcpy(&out_rowmajor[static_cast<std::size_t>(t) *
                              static_cast<std::size_t>(mel_dim_out)],
                row, static_cast<std::size_t>(mel_dim_out) * sizeof(float));
  }
  for (int m = 0; m < mel_dim_out; ++m) {
    double sum = 0.0;
    for (int t = 0; t < num_frames; ++t) {
      sum += static_cast<double>(
          out_rowmajor[static_cast<std::size_t>(t) *
                           static_cast<std::size_t>(mel_dim_out) +
                       static_cast<std::size_t>(m)]);
    }
    const float mean =
        static_cast<float>(sum / static_cast<double>(std::max(1, num_frames)));
    for (int t = 0; t < num_frames; ++t) {
      out_rowmajor[static_cast<std::size_t>(t) *
                       static_cast<std::size_t>(mel_dim_out) +
                   static_cast<std::size_t>(m)] -= mean;
    }
  }
}

}  // namespace cppannote::fbank
