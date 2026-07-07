// SPDX-License-Identifier: MIT
// Kaldi log-fbank aligned with cppannote
// ``ONNXWeSpeakerPretrainedSpeakerEmbedding.compute_fbank``.

#ifndef COMPUTE_FBANK_H_
#define COMPUTE_FBANK_H_

#include <cstdint>
#include <vector>

namespace cppannote::fbank {

/// Mono waveform ``[-1,1]`` → log-fbank, shape ``(T * num_mel_bins)``
/// row-major. Applies ``wave * (1<<15)`` before analysis and per-mel mean
/// subtraction over time (matches Torch path).
void wespeaker_like_fbank(float sample_hz, int num_mel_bins,
                          float frame_length_ms, float frame_shift_ms,
                          const float* mono, int num_samples,
                          std::vector<float>& out_rowmajor, int& num_frames,
                          int& mel_dim_out);

}  // namespace cppannote::fbank

#endif  // COMPUTE_FBANK_H_
