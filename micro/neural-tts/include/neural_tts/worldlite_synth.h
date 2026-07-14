// WORLD-lite vocoder synthesis (float32, kissfft) for the RP2350.
//
// Renders 61-control WORLD-lite frames -- f0 (Hz, 0 = unvoiced), benv[48]
// (sqrt-power spectral envelope at mel knots), bap[12] (aperiodicity at mel
// band centers) at 5 ms / 16 kHz -- to PCM. This is a single-precision port
// of WORLD's Synthesis() (mmorise/World, BSD-3-Clause): pitch-synchronous
// minimum-phase impulse responses plus shaped noise, overlap-added at
// phase-derived pulse locations. Differences from upstream WORLD:
//
//   * float32 + kiss_fftr everywhere (the M33 FPU is single-precision;
//     double is softfloat and ~10x slower).
//   * the band envelope / aperiodicity expansion (mel piecewise-linear to
//     the 513-bin FFT grid, models/neural_tts/worldlite.py expand()) is
//     fused in, so callers pass band controls, not full spectra.
//   * unvoiced pulse rate is 200 Hz (one per 5 ms frame) instead of
//     WORLD's 500 Hz default -- 2.5x fewer noise FFTs, no audible change
//     (validated against pyworld on the host).
//   * streaming output: overlap-add happens in a 2048-sample ring and
//     finished samples are emitted as int16 chunks, so a full-utterance
//     float PCM buffer is never held in RAM.

#ifndef NEURAL_TTS_WORLDLITE_SYNTH_H_
#define NEURAL_TTS_WORLDLITE_SYNTH_H_

#include <cstdint>

#include "kiss_fftr.h"

namespace neural_tts {

constexpr int kWorldSampleRate = 16000;
constexpr int kWorldFftSize = 1024;
constexpr int kWorldFrameSamples = 80;  // 5 ms at 16 kHz
constexpr int kWorldNumBenv = 48;
constexpr int kWorldNumBap = 12;
constexpr int kWorldSpecBins = kWorldFftSize / 2 + 1;  // 513

// Bytes kiss_fftr_alloc needs for one real-FFT plan (query with mem=nullptr).
inline size_t KissFftrPlanBytes(int nfft, int inverse_fft) {
  size_t n = 0;
  kiss_fftr_alloc(nfft, inverse_fft, nullptr, &n);
  return n;
}

// Forward + inverse plan storage for Synthesize() (typically ~21 KiB at nfft=1024).
inline size_t KissFftrPairBytes(int nfft) {
  return KissFftrPlanBytes(nfft, 0) + KissFftrPlanBytes(nfft, 1);
}

// Per-frame controls in the deploy parameterization.
struct WorldFrame {
  float f0;                       // Hz; 0 => unvoiced
  float benv[kWorldNumBenv];      // sqrt-power envelope at mel knots
  float bap[kWorldNumBap];        // aperiodicity 0..1 at mel band centers
};

class WorldLiteSynth {
 public:
  // Host / bring-up: allocates kissfft plans from the malloc heap.
  WorldLiteSynth();

  // On-device: build plans in caller-owned memory (e.g. the shared TFLM
  // tensor-arena bump). `plan_mem` must hold KissFftrPairBytes(nfft) bytes;
  // the destructor does not free it.
  WorldLiteSynth(void* plan_mem, size_t plan_mem_bytes);

  ~WorldLiteSynth();

  // Frames are read through the callback so the caller can stream-decode
  // them from flash; get_frame(user, t, &frame) fills frame index t.
  typedef void (*GetFrameFn)(void* user, int t, WorldFrame* frame);
  // emit(user, samples, n) receives finished int16 PCM in order.
  typedef void (*EmitFn)(void* user, const int16_t* samples, int n);

  // Renders num_frames control frames (num_frames * 80 output samples).
  // `gain` scales float PCM before the int16 conversion.
  void Synthesize(GetFrameFn get_frame, void* frame_user, int num_frames,
                  float gain, EmitFn emit, void* emit_user);

  // False if kissfft plan setup failed; using the synth in that state chases
  // garbage plan state forever.
  bool ok() const { return fwd_ != nullptr && inv_ != nullptr; }

  // Bring-up: plan pointers + a self-test that runs one forward+inverse
  // FFT pair through the plans (the on-device pipeline locks the core in
  // its first kiss_fftri; this isolates FFT-on-plans from the pipeline).
  const void* fwd_plan() const { return fwd_; }
  const void* inv_plan() const { return inv_; }
  float FftSelfTest();

 private:
  struct BinMap {
    int16_t idx;   // lower knot index
    float frac;    // interpolation fraction toward idx+1
  };

  void InitTables();
  void ExpandFrame(const WorldFrame& f, float* spec_pow, float* ap) const;
  void MinimumPhase(const float* log_amp_half, kiss_fft_cpx* min_phase);
  void RenderPulse(const float* spec_pow, const float* ap, bool voiced,
                   int noise_size, float frac_shift_s, float* response);
  void FlushTo(int abs_pos, float gain, EmitFn emit, void* emit_user);
  float Randn();

  // mel-knot -> FFT-bin expansion tables
  BinMap benv_map_[kWorldSpecBins];
  BinMap bap_map_[kWorldSpecBins];
  float dc_remover_[kWorldFftSize];

  // kissfft plans: heap-owned (default ctor) or arena-backed (2nd ctor).
  kiss_fftr_cfg fwd_;
  kiss_fftr_cfg inv_;
  bool owns_fft_plans_ = true;

  // scratch (~45 KiB total; instantiate statically on-device)
  float time_buf_[kWorldFftSize];
  kiss_fft_cpx spec_a_[kWorldSpecBins];
  kiss_fft_cpx spec_b_[kWorldSpecBins];
  float periodic_[kWorldFftSize];
  float aperiodic_[kWorldFftSize];
  float response_[kWorldFftSize];
  float pow_i_[kWorldSpecBins];
  float ap_i_[kWorldSpecBins];
  float cache_pow_[2][kWorldSpecBins];  // expanded-frame cache
  float cache_ap_[2][kWorldSpecBins];

  // streaming overlap-add ring (2 * fft samples)
  static constexpr int kRingSize = 2 * kWorldFftSize;
  float ring_[kRingSize];
  int flushed_;            // absolute sample index of first un-emitted sample
  int16_t emit_buf_[256];
  WorldFrame tmp_frame_;   // Synthesize scratch (stack is a 4 KB bank)

  uint32_t rng_state_;
};

}  // namespace neural_tts

#endif  // NEURAL_TTS_WORLDLITE_SYNTH_H_
