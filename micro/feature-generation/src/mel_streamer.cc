#include <cmath>

#include "feature_generation/feature_generation.h"
#include "fft_scratch.h"
#include "kiss_fftr.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace spelling {

MelStreamer::MelStreamer(int n_mels, int window_frames, int n_fft,
                         const float* window, const int* nz_off,
                         const int* nz_idx, const float* nz_val,
                         kiss_fftr_state* fft, float eps)
    : n_mels_(n_mels),
      window_frames_(window_frames),
      n_fft_(n_fft),
      n_freq_(n_fft / 2 + 1),
      window_(window),
      nz_off_(nz_off),
      nz_idx_(nz_idx),
      nz_val_(nz_val),
      fft_(fft),
      eps_(eps) {
  if (window_frames_ > kMaxCols || n_mels_ > kMaxMels) {
    MicroPrintf("MelStreamer: window_frames %d / n_mels %d exceed max %d/%d",
                window_frames_, n_mels_, kMaxCols, kMaxMels);
    while (true) { /* halt */
    }
  }
  if (n_fft_ > kFftScratchNFft) {
    MicroPrintf("MelStreamer: n_fft %d exceeds shared scratch pool %d", n_fft_,
                kFftScratchNFft);
    while (true) { /* halt */
    }
  }
  Reset();
}

void MelStreamer::Reset() {
  head_ = 0;
  count_ = 0;
  // Clear the ring so any slot read before it is written (e.g. an indexing
  // edge case during warm-up) yields log(eps)-like silence rather than
  // uninitialised stack garbage, which can be NaN and poison the whole
  // per-window standardisation.
  const float log_eps = std::log(eps_);
  for (std::size_t i = 0; i < static_cast<std::size_t>(kMaxCols) * kMaxMels;
       ++i) {
    ring_[i] = log_eps;
  }
}

void MelStreamer::PushHop(const float* hop_samples) {
  // FFT scratch comes from a single shared .bss pool (fft_scratch.cc), NOT the
  // stack: on the RP2350 each core's stack is a single 4 KB scratch bank
  // (core 0 -> SCRATCH_Y, core 1 -> SCRATCH_X right below). A ~10 KB stack
  // frame here overflows core 0's bank into core 1's stack, and CMSIS-NN runs
  // a GEMM worker on core 1 during TFLM Invoke(), so the colliding stacks
  // silently corrupt data (see petewarden.com/2024/01/16). The pool is shared
  // with LogMelSpectrogram::Compute -- safe because both run only on core 0
  // and never concurrently (VAD-listen vs STT phase) or re-entrantly.
  float* const frame_buf = g_fft_scratch_frame;
  kiss_fft_cpx* const spectrum = g_fft_scratch_spec;
  float* const power_row = g_fft_scratch_pow;

  // Window the block (no reflect padding: center=False, hop == n_fft).
  for (int i = 0; i < n_fft_; ++i) {
    frame_buf[i] = hop_samples[i] * window_[i];
  }
  kiss_fftr(fft_, frame_buf, spectrum);
  for (int k = 0; k < n_freq_; ++k) {
    const float re = spectrum[k].r;
    const float im = spectrum[k].i;
    power_row[k] = re * re + im * im;
  }

  // Fused CSR mel multiply + log into the newest ring column.
  float* col = &ring_[static_cast<std::size_t>(head_) * n_mels_];
  for (int m = 0; m < n_mels_; ++m) {
    const int lo = nz_off_[m];
    const int hi = nz_off_[m + 1];
    float sum = 0.0f;
    for (int p = lo; p < hi; ++p) {
      sum += nz_val_[p] * power_row[nz_idx_[p]];
    }
    col[m] = std::log(sum + eps_);
  }

  head_ = (head_ + 1) % window_frames_;
  if (count_ < window_frames_) ++count_;
}

void MelStreamer::BuildModelInput(float* out) const {
  const int T = window_frames_;
  const float log_eps = std::log(eps_);

  // Lay out columns oldest (t=0) .. newest (t=T-1). The frame pushed `i` steps
  // ago lives in slot (head_-1-i) mod T and maps to t = T-1-i. Positions older
  // than the number of real frames are silence-padded with log(eps).
  for (int t = 0; t < T; ++t) {
    const int i = (T - 1) - t;  // steps-ago for this temporal position
    if (i < count_) {
      const int slot = ((head_ - 1 - i) % T + T) % T;
      const float* col = &ring_[static_cast<std::size_t>(slot) * n_mels_];
      for (int m = 0; m < n_mels_; ++m) {
        out[static_cast<std::size_t>(m) * T + t] = col[m];
      }
    } else {
      for (int m = 0; m < n_mels_; ++m) {
        out[static_cast<std::size_t>(m) * T + t] = log_eps;
      }
    }
  }

  // Per-window mean/std standardisation -- identical to
  // LogMelSpectrogram::Compute(): Bessel-corrected std, clamped to 1e-3.
  const std::size_t total = static_cast<std::size_t>(n_mels_) * T;
  double sum = 0.0;
  for (std::size_t i = 0; i < total; ++i) sum += out[i];
  const double mean = sum / static_cast<double>(total > 0 ? total : 1);
  double ssq = 0.0;
  for (std::size_t i = 0; i < total; ++i) {
    const double d = static_cast<double>(out[i]) - mean;
    ssq += d * d;
  }
  const std::size_t denom = total > 1 ? total - 1 : 1;
  double std_dev = std::sqrt(ssq / static_cast<double>(denom));
  if (std_dev < 1e-3) std_dev = 1e-3;
  const float inv = static_cast<float>(1.0 / std_dev);
  const float meanf = static_cast<float>(mean);
  for (std::size_t i = 0; i < total; ++i) {
    out[i] = (out[i] - meanf) * inv;
  }
}

}  // namespace spelling
