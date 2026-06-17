// Heap-free log-mel spectrogram for the on-device build.
//
// Reflect-pad (center=True), windowed STFT, squared-magnitude, Slaney mel
// filterbank, crop to target_frames, log(value + eps), per-clip mean/std
// normalisation with std clamped >= 1e-3.
//
// The DIFFERENCES from the desktop port are all about memory:
//
//   1. Compute() takes a caller-supplied output buffer instead of
//      returning std::vector. Saves 64 KB of heap traffic per call.
//
//   2. Reflect padding is done on-the-fly per sample inside the
//      framing loop. The desktop port materialised a 16512-sample
//      `padded` vector (~66 KB).
//
//   3. STFT + power-spectrum + filterbank multiply are FUSED into a
//      single pass: for each output frame we compute one 257-bin
//      power row on the stack, then immediately scatter it through
//      the mel filterbank into one column of the output buffer. The
//      desktop port allocated a full (n_freq, n_frames) power matrix
//      (~200 KB) and a (n_mels, n_frames) mel matrix (~64 KB).
//
//   4. log() and normalisation run in-place on the output buffer.
//      Saves another 64 KB.
//
// With our defaults this drops peak per-call heap usage from
// ~450 KB to 0 (just ~5 KB on the stack for frame_buf, spectrum,
// and one power row). The window + filterbank still live as heap
// std::vector members of LogMelSpectrogram itself (84 KB total),
// allocated ONCE in the constructor.

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include "feature_generation/feature_generation.h"
#include "fft_scratch.h"
#include "kiss_fftr.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace {
[[noreturn]] void FatalAndHalt(const char* msg) {
  MicroPrintf("log_mel fatal: %s", msg);
  while (true) { /* nothing useful to do without the front-end */
  }
}
}  // namespace

namespace spelling {

namespace {
// Slaney mel-scale constants (match torchaudio _hz_to_mel("slaney")).
constexpr float kFSp = 200.0f / 3.0f;
constexpr float kMinLogHz = 1000.0f;
constexpr float kMinLogMel = kMinLogHz / kFSp;  // == 15.0
const float kLogStep = std::log(6.4f) / 27.0f;

// Reflect index without duplicating the boundary. Matches
// torch.nn.functional.pad(x, (pad, pad), mode="reflect"): index -1 -> 1,
// -2 -> 2, ..., and on the right side, n -> n-2, n+1 -> n-3, ...
//
// `n` is the original waveform length, `i` is the padded-coords
// index in [-pad, n + pad). The implementation handles only the
// edge regions: callers in the hot loop pass already-validated
// indices in [0, n) directly past this helper. We use it from the
// framing loop where we sometimes step a sample or two past either
// edge due to center padding.
inline int ReflectIndex(int i, int n) {
  if (n <= 1) return 0;
  // Fast path: vast majority of samples are inside [0, n).
  if (i >= 0 && i < n) return i;
  // The mathematically clean form uses (2n - 2) as the period of
  // the reflected sequence. We loop in case the offset is very
  // large; in practice we only reflect by at most n_fft/2 = 256.
  const int period = 2 * (n - 1);
  // Map i to [0, period) by adding multiples of period.
  int m = i % period;
  if (m < 0) m += period;
  return (m < n) ? m : (period - m);
}
}  // namespace

float HzToMelSlaney(float hz) {
  if (hz >= kMinLogHz) {
    return kMinLogMel + std::log(hz / kMinLogHz) / kLogStep;
  }
  return hz / kFSp;
}

float MelToHzSlaney(float mel) {
  if (mel >= kMinLogMel) {
    return kMinLogHz * std::exp(kLogStep * (mel - kMinLogMel));
  }
  return kFSp * mel;
}

std::vector<float> HannWindowPeriodic(int length) {
  if (length <= 0) return {};
  // Periodic Hann: symmetric window of length L+1 with the last
  // sample dropped so adjacent windows tile cleanly.
  const int n = length + 1;
  std::vector<float> w(length);
  const double denom = static_cast<double>(n - 1);
  for (int i = 0; i < length; ++i) {
    w[i] = static_cast<float>(
        0.5 - 0.5 * std::cos(2.0 * M_PI * static_cast<double>(i) / denom));
  }
  return w;
}

std::vector<float> MakeMelFilterbank(int n_freq, int n_mels, int sample_rate,
                                     float f_min, float f_max) {
  if (n_freq <= 1 || n_mels <= 0) return {};
  const float mel_min = HzToMelSlaney(f_min);
  const float mel_max = HzToMelSlaney(f_max);
  std::vector<float> mel_pts(static_cast<std::size_t>(n_mels + 2));
  for (int i = 0; i < n_mels + 2; ++i) {
    mel_pts[static_cast<std::size_t>(i)] =
        mel_min + (mel_max - mel_min) * static_cast<float>(i) /
                      static_cast<float>(n_mels + 1);
  }
  std::vector<float> hz_pts(mel_pts.size());
  for (std::size_t i = 0; i < mel_pts.size(); ++i) {
    hz_pts[i] = MelToHzSlaney(mel_pts[i]);
  }
  std::vector<float> bin_hz(static_cast<std::size_t>(n_freq));
  for (int k = 0; k < n_freq; ++k) {
    bin_hz[static_cast<std::size_t>(k)] = static_cast<float>(sample_rate) *
                                          0.5f * static_cast<float>(k) /
                                          static_cast<float>(n_freq - 1);
  }
  std::vector<float> fb(
      static_cast<std::size_t>(n_mels) * static_cast<std::size_t>(n_freq),
      0.0f);
  for (int m = 0; m < n_mels; ++m) {
    const float f_left = hz_pts[static_cast<std::size_t>(m)];
    const float f_center = hz_pts[static_cast<std::size_t>(m + 1)];
    const float f_right = hz_pts[static_cast<std::size_t>(m + 2)];
    const float enorm = 2.0f / (f_right - f_left);  // Slaney area norm
    float* row =
        &fb[static_cast<std::size_t>(m) * static_cast<std::size_t>(n_freq)];
    for (int k = 0; k < n_freq; ++k) {
      const float f = bin_hz[static_cast<std::size_t>(k)];
      if (f <= f_left || f >= f_right) continue;
      const float w = (f <= f_center) ? (f - f_left) / (f_center - f_left)
                                      : (f_right - f) / (f_right - f_center);
      row[k] = w * enorm;
    }
  }
  return fb;
}

LogMelSpectrogram::LogMelSpectrogram(const LogMelParams& params)
    : params_(params), n_freq_(params.n_fft / 2 + 1) {
  if (params_.n_fft <= 0 || (params_.n_fft & (params_.n_fft - 1)) != 0) {
    FatalAndHalt("LogMelSpectrogram: n_fft must be a power of two");
  }
  if (params_.win_length > params_.n_fft) {
    FatalAndHalt("LogMelSpectrogram: win_length must be <= n_fft");
  }
  if (params_.f_max <= 0.0f) {
    params_.f_max = static_cast<float>(params_.sample_rate) * 0.5f;
  }

  const int n_mels = params_.n_mels;
  const int n_freq = n_freq_;
  if (n_mels < 0 || n_mels > 256) {
    FatalAndHalt("LogMelSpectrogram: n_mels out of supported range [0,256]");
  }

  const bool have_precomputed = params_.precomputed_window != nullptr &&
                                params_.precomputed_nz_off != nullptr &&
                                params_.precomputed_nz_idx != nullptr &&
                                params_.precomputed_nz_val != nullptr;

  if (have_precomputed) {
    // FLASH-TABLE PATH (the default for the Pico app): the window and
    // CSR mel filterbank were precomputed at build time into const
    // arrays in flash (generated/mel_tables.{h,cc}). We do ZERO heap
    // allocation and ZERO trig/log here -- just validate the dimensions
    // the caller asserts and adopt the pointers. This is what lets the
    // front-end fit comfortably in a small heap.
    if (params_.precomputed_window_len != params_.n_fft) {
      FatalAndHalt("precomputed_window_len != n_fft");
    }
    if (params_.precomputed_nz_total < 0 ||
        params_.precomputed_nz_off[0] != 0 ||
        params_.precomputed_nz_off[n_mels] != params_.precomputed_nz_total) {
      // Catches a table generated for a different n_mels / config than
      // the LogMelParams we were handed.
      FatalAndHalt("precomputed CSR offsets inconsistent with n_mels/nz_total");
    }
    window_ = params_.precomputed_window;
    mel_nz_off_ = params_.precomputed_nz_off;
    mel_nz_idx_ = params_.precomputed_nz_idx;
    mel_nz_val_ = params_.precomputed_nz_val;
    mel_nz_total_ = params_.precomputed_nz_total;
  } else {
    // RUNTIME-BUILD FALLBACK (host tests / when no tables are supplied):
    // compute the window + CSR filterbank into owned std::vector storage.
    //
    // Center the raw Hann window inside a zero-padded length-n_fft
    // buffer. With our default win_length == n_fft the pad is zero and
    // the window is just the raw Hann.
    std::vector<float> raw = HannWindowPeriodic(params_.win_length);
    if (params_.win_length == params_.n_fft) {
      window_storage_ = std::move(raw);
    } else {
      const int pad_total = params_.n_fft - params_.win_length;
      const int left = pad_total / 2;
      window_storage_.assign(static_cast<std::size_t>(params_.n_fft), 0.0f);
      std::copy(raw.begin(), raw.end(),
                window_storage_.begin() + static_cast<std::ptrdiff_t>(left));
    }

    // Build the CSR mel filterbank DIRECTLY, without ever materialising
    // the dense (n_mels x n_freq) matrix. At 64 mels x 257 bins that
    // dense buffer is ~64 KB -- a single allocation that on its own
    // overflows the whole Pico heap. Each triangular row is recomputed
    // on the fly; the per-row weight formula below is byte-for-byte
    // identical to MakeMelFilterbank() (kept for desktop tests).
    const float mel_min = HzToMelSlaney(params_.f_min);
    const float mel_max = HzToMelSlaney(params_.f_max);

    // Hz edges for the n_mels+2 mel points. n_mels is bounded above, so
    // a fixed stack array (<= 258 floats, ~1 KB) avoids any heap churn.
    float hz_pts[258];
    for (int i = 0; i < n_mels + 2; ++i) {
      const float mel = mel_min + (mel_max - mel_min) * static_cast<float>(i) /
                                      static_cast<float>(n_mels + 1);
      hz_pts[i] = MelToHzSlaney(mel);
    }

    // Triangular Slaney weight for mel bin m at frequency bin k. Mirrors
    // MakeMelFilterbank()'s inner loop exactly: same bin_hz, same enorm
    // (Slaney area normalisation), same operand order, same zero region.
    auto row_weight = [&](int m, int k) -> float {
      const float f_left = hz_pts[m];
      const float f_center = hz_pts[m + 1];
      const float f_right = hz_pts[m + 2];
      const float f = static_cast<float>(params_.sample_rate) * 0.5f *
                      static_cast<float>(k) / static_cast<float>(n_freq - 1);
      if (f <= f_left || f >= f_right) return 0.0f;
      const float enorm = 2.0f / (f_right - f_left);
      const float w = (f <= f_center) ? (f - f_left) / (f_center - f_left)
                                      : (f_right - f) / (f_right - f_center);
      return w * enorm;
    };

    // Pass 1: count nonzeros per row to lay out the CSR offsets.
    nz_off_storage_.assign(static_cast<std::size_t>(n_mels + 1), 0);
    for (int m = 0; m < n_mels; ++m) {
      int cnt = 0;
      for (int k = 0; k < n_freq; ++k) {
        if (row_weight(m, k) != 0.0f) ++cnt;
      }
      nz_off_storage_[static_cast<std::size_t>(m + 1)] =
          nz_off_storage_[static_cast<std::size_t>(m)] + cnt;
    }
    // Pass 2: scatter the nonzero (index, value) pairs.
    mel_nz_total_ = nz_off_storage_[static_cast<std::size_t>(n_mels)];
    nz_idx_storage_.assign(static_cast<std::size_t>(mel_nz_total_), 0);
    nz_val_storage_.assign(static_cast<std::size_t>(mel_nz_total_), 0.0f);
    for (int m = 0; m < n_mels; ++m) {
      int w = nz_off_storage_[static_cast<std::size_t>(m)];
      for (int k = 0; k < n_freq; ++k) {
        const float val = row_weight(m, k);
        if (val != 0.0f) {
          nz_idx_storage_[static_cast<std::size_t>(w)] = k;
          nz_val_storage_[static_cast<std::size_t>(w)] = val;
          ++w;
        }
      }
    }

    window_ = window_storage_.data();
    mel_nz_off_ = nz_off_storage_.data();
    mel_nz_idx_ = nz_idx_storage_.data();
    mel_nz_val_ = nz_val_storage_.data();
  }

  if (params_.external_fft != nullptr) {
    // Adopt a caller-owned, shared FFT state (same n_fft). We must not free it.
    fft_ = params_.external_fft;
    owns_fft_ = false;
  } else {
    fft_ = kiss_fftr_alloc(params_.n_fft, /*inverse_fft=*/0, nullptr, nullptr);
    owns_fft_ = true;
    if (fft_ == nullptr) {
      FatalAndHalt("kiss_fftr_alloc failed");
    }
  }
}

LogMelSpectrogram::~LogMelSpectrogram() {
  if (fft_ != nullptr && owns_fft_) {
    kiss_fftr_free(fft_);
  }
  fft_ = nullptr;
}

void LogMelSpectrogram::Compute(const float* waveform, std::size_t n_samples,
                                float* out) const {
  const int n_fft = params_.n_fft;
  const int hop = params_.hop_length;
  const int n_mels = params_.n_mels;
  const int n_freq = n_freq_;
  const int target = params_.target_frames;
  const int pad = params_.center ? n_fft / 2 : 0;
  const int n = static_cast<int>(n_samples);

  // Per-frame FFT scratch lives in the shared .bss pool (fft_scratch.h), NOT
  // on the stack. The RP2350 gives each core only a 4 KB scratch-bank stack; a
  // ~10 KB stack frame here overflows core 0's bank into core 1's stack, and
  // CMSIS-NN runs a dual-core GEMM worker on core 1 during Invoke(), so the
  // colliding stacks silently corrupt data (see petewarden.com/2024/01/16).
  // Safe to share with the VAD streamer because both run only on core 0 and
  // never concurrently / re-entrantly.
  float* const frame_buf = g_fft_scratch_frame;
  kiss_fft_cpx* const spectrum = g_fft_scratch_spec;
  float* const power_row = g_fft_scratch_pow;
  if (n_fft > kFftScratchNFft) {
    FatalAndHalt("Compute: n_fft exceeds shared FFT scratch pool");
  }

  // Padded waveform length = n + 2*pad. n_frames is the count of
  // full-length windows that fit; we cap at target_frames.
  const int padded_n = n + 2 * pad;
  int n_frames_full = 0;
  if (padded_n >= n_fft) {
    n_frames_full = (padded_n - n_fft) / hop + 1;
  }
  const int n_frames_out = std::min(n_frames_full, target);

  // Pre-fill the output with log(eps): this becomes the value of any
  // trailing frames we don't actually compute (matches the desktop
  // implementation's right-zero-pad-then-log-then-normalize order,
  // because log(0 + eps) is exactly log(eps)).
  const float log_eps = std::log(params_.eps);
  for (int m = 0; m < n_mels; ++m) {
    float* row = out + static_cast<std::ptrdiff_t>(m) * target;
    for (int t = 0; t < target; ++t) row[t] = log_eps;
  }

  for (int fi = 0; fi < n_frames_out; ++fi) {
    // Padded-coordinate start of this frame. The actual waveform
    // index for `frame_buf[i]` is `s + i - pad`, with reflect
    // padding applied on the edges.
    const int s = fi * hop;

    // Build the windowed frame, reflecting across the waveform
    // boundary as needed. We don't materialise a `padded` buffer --
    // the reflection happens per sample. With our default config
    // each frame is touched 512 times, and only at most 256 of
    // those samples are in the reflect region (the very first and
    // very last frames).
    for (int i = 0; i < n_fft; ++i) {
      const int orig_idx = s + i - pad;
      float s_val;
      if (orig_idx >= 0 && orig_idx < n) {
        s_val = waveform[orig_idx];
      } else if (params_.center) {
        // ReflectIndex maps any index to a valid [0, n) waveform index.
        s_val = waveform[ReflectIndex(orig_idx, n)];
      } else {
        s_val = 0.0f;
      }
      frame_buf[i] = s_val * window_[static_cast<std::size_t>(i)];
    }

    kiss_fftr(fft_, frame_buf, spectrum);

    // Power spectrum for this frame (one 257-float row).
    for (int k = 0; k < n_freq; ++k) {
      const float re = spectrum[k].r;
      const float im = spectrum[k].i;
      power_row[k] = re * re + im * im;
    }

    // Fused mel filterbank multiply: for each mel bin, dot-product
    // the precomputed nonzero entries against power_row and write
    // the result directly into the (m, fi) cell of `out`. Output is
    // row-major (n_mels rows, target columns), so out[m][fi] is at
    // out[m * target + fi].
    for (int m = 0; m < n_mels; ++m) {
      const int off_lo = mel_nz_off_[static_cast<std::size_t>(m)];
      const int off_hi = mel_nz_off_[static_cast<std::size_t>(m + 1)];
      float sum = 0.0f;
      for (int p = off_lo; p < off_hi; ++p) {
        sum += mel_nz_val_[static_cast<std::size_t>(p)] *
               power_row[mel_nz_idx_[static_cast<std::size_t>(p)]];
      }
      // log(value + eps), written in-place. We do log here (rather
      // than a separate pass after the framing loop) because the
      // value is in cache right now.
      out[static_cast<std::ptrdiff_t>(m) * target + fi] =
          std::log(sum + params_.eps);
    }
  }
  // Frames in [n_frames_out, target) keep their pre-filled log(eps).

  // Per-clip standardisation -- exactly as in the desktop port, on
  // the same (n_mels * target) buffer. Bessel-corrected std, clamped
  // to 1e-3 so silent inputs don't divide by zero.
  const std::size_t total =
      static_cast<std::size_t>(n_mels) * static_cast<std::size_t>(target);
  double sum = 0.0;
  for (std::size_t i = 0; i < total; ++i) sum += out[i];
  const double mean =
      sum / static_cast<double>(std::max<std::size_t>(total, 1));
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
