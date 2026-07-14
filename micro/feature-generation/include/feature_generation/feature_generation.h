// feature-generation -- portable, heap-free log-mel spectrogram front-end.
//
// This is the single public header for the module. It exposes two front-ends
// Slaney mel / periodic-Hann / kissfft real-FFT maths shared by both front-ends:
//
//   * LogMelSpectrogram -- a one-shot ("batch") front-end. Given a whole
//     waveform it produces a normalised (n_mels x target_frames) log-mel
//     feature plane. Used by the STT classifier.
//
//   * MelStreamer -- a streaming front-end for always-on use. It keeps the
//     last `window_frames` log-mel columns in a ring and computes exactly ONE
//     FFT per pushed hop, then rebuilds the normalised (n_mels x window_frames)
//     model input. Used by the VAD, where the window slides one
//     non-overlapping n_fft-sample block per inference.
//
// Reflect-pad (center=true), windowed STFT, squared magnitude, Slaney mel
// filterbank, crop to frames, log(value + eps), per-clip mean/std
// standardisation (Bessel-corrected std clamped >= 1e-3).
//
// Dependencies: kissfft (real FFT) and TFLM's micro_log (MicroPrintf, used only
// on fatal misconfiguration). No app, VAD, or STT dependencies -- so the module
// can be reused by any front-end on any platform that provides those two.
//
// Memory: both front-ends are heap-free when handed precomputed flash tables.
// Per-call FFT scratch lives in a shared .bss pool (see src/fft_scratch.h),
// NOT on the stack, because the RP2350 gives each core only a 4 KB scratch-bank
// stack and a ~5 KB FFT frame would overflow it into the other core's stack.

#ifndef FEATURE_GENERATION_FEATURE_GENERATION_H_
#define FEATURE_GENERATION_FEATURE_GENERATION_H_

#include <cstddef>
#include <cstdint>
#include <vector>

// Forward-declare kissfft's opaque real-FFT config so this header doesn't pull
// in kiss_fft.h. The .cc files own the actual lifecycle.
struct kiss_fftr_state;

namespace spelling {

// ---------------------------------------------------------------------------
// Shared building blocks (exposed for table generation / tests).
// ---------------------------------------------------------------------------

// Periodic Hann window of length L (matches torch.hann_window(L,
// periodic=True)). Returns an empty vector when L <= 0.
std::vector<float> HannWindowPeriodic(int length);

// Slaney's auditory-toolbox mel scale (matches torchaudio
// _hz_to_mel("slaney")).
float HzToMelSlaney(float hz);
float MelToHzSlaney(float mel);

// Dense (n_mels, n_freq) Slaney-normalised triangular mel filterbank,
// row-major.
std::vector<float> MakeMelFilterbank(int n_freq, int n_mels, int sample_rate,
                                     float f_min, float f_max);

// ---------------------------------------------------------------------------
// Batch front-end: LogMelSpectrogram.
// ---------------------------------------------------------------------------

struct LogMelParams {
  int sample_rate = 16000;
  int n_fft = 512;  // power of two for kiss_fftr
  int hop_length = 80;
  int win_length = 512;  // <= n_fft; defaults to n_fft
  int n_mels = 80;
  float f_min = 20.0f;
  float f_max = 8000.0f;  // defaults to sample_rate / 2 at construction
  int target_frames = 200;
  float eps = 1e-6f;
  bool center = true;

  // OPTIONAL precomputed front-end tables (typically the const flash arrays a
  // generator emits). When all four CSR/window pointers are non-null the
  // constructor adopts them DIRECTLY -- no heap allocation, no trig/log work.
  // When null the window + CSR filterbank are computed into owned std::vector
  // storage (used by host tests / when no tables are supplied).
  //
  // The tables MUST match (sample_rate, n_fft, win_length, n_mels, f_min,
  // f_max); the constructor sanity-checks the sizes it can.
  const float* precomputed_window = nullptr;  // length n_fft
  int precomputed_window_len = 0;
  const int* precomputed_nz_off = nullptr;    // length n_mels + 1
  const int* precomputed_nz_idx = nullptr;    // length nz_off[n_mels]
  const float* precomputed_nz_val = nullptr;  // length nz_off[n_mels]
  int precomputed_nz_total = 0;               // == nz_off[n_mels]

  // OPTIONAL shared real-FFT state. When non-null the constructor adopts it
  // instead of calling kiss_fftr_alloc, and the destructor does NOT free it
  // (the caller owns it). Lets several front-ends with the same n_fft share one
  // twiddle table (~4 KB). They must never Compute()/PushHop() concurrently.
  kiss_fftr_state* external_fft = nullptr;
};

class LogMelSpectrogram {
 public:
  explicit LogMelSpectrogram(const LogMelParams& params);

  LogMelSpectrogram(const LogMelSpectrogram&) = delete;
  LogMelSpectrogram& operator=(const LogMelSpectrogram&) = delete;
  LogMelSpectrogram(LogMelSpectrogram&&) noexcept = default;
  LogMelSpectrogram& operator=(LogMelSpectrogram&&) noexcept = default;
  ~LogMelSpectrogram();

  // Heap-free streaming compute. `waveform` is read-only; `out` must hold at
  // least n_mels * target_frames floats. On return out[m * target_frames + t]
  // is the normalised log-mel value for mel bin m, frame t. Frames that would
  // require samples past the waveform are left at log(eps) (right-zero-pad).
  //
  // Two input formats: fp32 in [-1, 1], or raw int16 mic samples converted to
  // [-1, 1] at the read site (lets the caller store the 1 s clip as int16 and
  // halve its SRAM -- the samples are int16-precision to begin with). Both
  // produce identical features for the same underlying signal.
  void Compute(const float* waveform, std::size_t n_samples, float* out) const;
  void Compute(const int16_t* waveform, std::size_t n_samples, float* out) const;

  int n_mels() const { return params_.n_mels; }
  int target_frames() const { return params_.target_frames; }
  int n_freq() const { return n_freq_; }
  const LogMelParams& params() const { return params_; }

 private:
  // Shared body for both Compute() overloads; SampleT is float or int16_t.
  template <typename SampleT>
  void ComputeImpl(const SampleT* waveform, std::size_t n_samples,
                   float* out) const;

  LogMelParams params_;
  int n_freq_;            // n_fft / 2 + 1
  int mel_nz_total_ = 0;  // == mel_nz_off_[n_mels]

  // Owned backing storage, used ONLY when no precomputed tables were supplied.
  std::vector<float> window_storage_;
  std::vector<int> nz_off_storage_;
  std::vector<int> nz_idx_storage_;
  std::vector<float> nz_val_storage_;

  // Active views: either the precomputed flash arrays or the *_storage_
  // vectors. CSR layout: row m spans [mel_nz_off_[m], mel_nz_off_[m + 1]) of
  // mel_nz_idx_ (FFT bin column) / mel_nz_val_ (triangular weight).
  const float* window_ = nullptr;
  const int* mel_nz_off_ = nullptr;
  const int* mel_nz_idx_ = nullptr;
  const float* mel_nz_val_ = nullptr;

  kiss_fftr_state* fft_ = nullptr;
  bool owns_fft_ = true;  // false when adopting external_fft
};

// ---------------------------------------------------------------------------
// Streaming front-end: MelStreamer.
// ---------------------------------------------------------------------------
//
// Exactness: with hop == n_fft (non-overlapping blocks, center=false), after
// pushing N full blocks BuildModelInput() produces the SAME tensor
// LogMelSpectrogram::Compute(center=false) would over those N*n_fft samples
// (verified by tests/stream_parity_test).
//
// Memory: one ring of n_mels*window_frames floats (<= 4 KB at the VAD's
// 32x32 bound) + the shared .bss FFT scratch. No heap.
class MelStreamer {
 public:
  // All table pointers are borrowed (typically flash tables plus a shared Hann
  // window + FFT state). `window` has n_fft taps; the CSR mel filterbank is row
  // m -> [nz_off[m], nz_off[m+1]) into nz_idx (FFT bin) / nz_val (weight).
  MelStreamer(int n_mels, int window_frames, int n_fft, const float* window,
              const int* nz_off, const int* nz_idx, const float* nz_val,
              kiss_fftr_state* fft, float eps = 1e-6f);

  // Reset the ring (call at stream start / between independent clips).
  void Reset();

  // Push one hop == n_fft samples (a non-overlapping block). Computes one
  // windowed FFT -> power -> mel -> log column, stored as the newest column.
  void PushHop(const float* hop_samples);

  // Write the current window as a row-major (n_mels x window_frames) fp32
  // tensor into `out` (t=0 oldest .. window_frames-1 newest), per-window
  // mean/std normalised exactly like LogMelSpectrogram::Compute(). Warm-up
  // columns are filled with log(eps) on the OLD side.
  void BuildModelInput(float* out) const;

  int filled() const { return count_; }  // real hops pushed (saturates)
  int n_mels() const { return n_mels_; }
  int window_frames() const { return window_frames_; }

 private:
  const int n_mels_;
  const int window_frames_;
  const int n_fft_;
  const int n_freq_;
  const float* window_;
  const int* nz_off_;
  const int* nz_idx_;
  const float* nz_val_;
  kiss_fftr_state* fft_;
  const float eps_;

  // Ring of un-normalised log-mel columns: ring_[slot*n_mels + m]. head_ is the
  // next write slot; count_ the number of real columns present. The 32x32 cap
  // keeps this a ~4 KB member (a larger bound overflowed the small core stack
  // when the streamer is a local alongside the FFT scratch).
  static constexpr int kMaxCols = 32;  // window_frames upper bound
  static constexpr int kMaxMels = 32;  // n_mels upper bound
  float ring_[kMaxCols * kMaxMels];
  int head_;
  int count_;
};

}  // namespace spelling

#endif  // FEATURE_GENERATION_FEATURE_GENERATION_H_
