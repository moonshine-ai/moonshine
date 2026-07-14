// Float32 kissfft port of WORLD Synthesis() specialized to the WORLD-lite
// band parameterization. Reference: mmorise/World src/synthesis.cpp and
// src/common.cpp (BSD-3-Clause), models/neural_tts/worldlite.py expand().

#include "neural_tts/worldlite_synth.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(PICO_BUILD)
#include "pico/time.h"
// Progress hooks (defined by the app): record pipeline position for
// post-reboot hang reports and feed the watchdog (see pb_decoder.cc).
// tts_checkpoint2 is a secondary channel (pulse index). tts_trace
// appends (tag, value) to a RAM ring that survives watchdog reboots --
// USB stdio TX goes dark during synthesis, so this is the only reliable
// forensics channel.
extern "C" void tts_checkpoint(uint32_t v);
extern "C" void tts_checkpoint2(uint32_t v);
extern "C" void tts_trace(uint32_t tag, uint32_t val);
#define WL_CHECKPOINT(v) tts_checkpoint(v)
#define WL_CHECKPOINT2(v) tts_checkpoint2(v)
#define WL_TRACE_RING(tag, val) tts_trace((tag), (val))
#else
#define WL_CHECKPOINT(v) \
  do {                   \
  } while (0)
#define WL_CHECKPOINT2(v) \
  do {                    \
  } while (0)
#define WL_TRACE_RING(tag, val) \
  do {                          \
  } while (0)
#endif

namespace neural_tts {
namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kSafeGuardMin = 1e-12f;
// WORLD uses 500 Hz pulses in unvoiced spans; one per 5 ms frame is
// indistinguishable here and 2.5x cheaper.
constexpr float kUnvoicedPulseHz = 200.0f;
constexpr float kF0Floor = 32.0f;  // ~ fs / fft_size, as in WORLD Synthesis

float HzToMel(float hz) { return 2595.0f * log10f(1.0f + hz / 700.0f); }

}  // namespace

void WorldLiteSynth::InitTables() {
  flushed_ = 0;

  // mel piecewise-linear expansion tables (worldlite.py: knots are linearly
  // spaced in mel between 0 and fs/2)
  auto build_map = [](BinMap* map, int n_knots) {
    const float mel_max = HzToMel(kWorldSampleRate / 2.0f);
    for (int i = 0; i < kWorldSpecBins; ++i) {
      const float hz = static_cast<float>(i) * (kWorldSampleRate / 2.0f) /
                       (kWorldSpecBins - 1);
      const float pos = HzToMel(hz) / mel_max * (n_knots - 1);
      int idx = static_cast<int>(pos);
      if (idx >= n_knots - 1) idx = n_knots - 2;
      map[i].idx = static_cast<int16_t>(idx);
      float frac = pos - idx;
      if (frac < 0.0f) frac = 0.0f;
      if (frac > 1.0f) frac = 1.0f;
      map[i].frac = frac;
    }
  };
  build_map(benv_map_, kWorldNumBenv);
  build_map(bap_map_, kWorldNumBap);

  // WORLD GetDCRemover()
  float dc = 0.0f;
  for (int i = 0; i < kWorldFftSize / 2; ++i) {
    dc_remover_[i] =
        0.5f - 0.5f * cosf(2.0f * kPi * (i + 1.0f) / (1.0f + kWorldFftSize));
    dc_remover_[kWorldFftSize - i - 1] = dc_remover_[i];
    dc += dc_remover_[i] * 2.0f;
  }
  for (int i = 0; i < kWorldFftSize / 2; ++i) {
    dc_remover_[i] /= dc;
    dc_remover_[kWorldFftSize - i - 1] = dc_remover_[i];
  }
}

WorldLiteSynth::WorldLiteSynth() : rng_state_(0x8f1bbcdcu), owns_fft_plans_(true) {
  fwd_ = kiss_fftr_alloc(kWorldFftSize, 0, nullptr, nullptr);
  inv_ = kiss_fftr_alloc(kWorldFftSize, 1, nullptr, nullptr);
  InitTables();
}

WorldLiteSynth::WorldLiteSynth(void* plan_mem, size_t plan_mem_bytes)
    : rng_state_(0x8f1bbcdcu), owns_fft_plans_(false) {
  fwd_ = nullptr;
  inv_ = nullptr;
  if (plan_mem != nullptr && plan_mem_bytes >= KissFftrPairBytes(kWorldFftSize)) {
    uint8_t* p = static_cast<uint8_t*>(plan_mem);
    size_t left = plan_mem_bytes;
    size_t need = 0;
    kiss_fftr_alloc(kWorldFftSize, 0, nullptr, &need);
    if (need <= left) {
      fwd_ = kiss_fftr_alloc(kWorldFftSize, 0, p, &need);
      if (fwd_ != nullptr) {
        p += need;
        left -= need;
        size_t inv_need = 0;
        kiss_fftr_alloc(kWorldFftSize, 1, nullptr, &inv_need);
        if (inv_need <= left) {
          inv_ = kiss_fftr_alloc(kWorldFftSize, 1, p, &inv_need);
        }
      }
    }
  }
  InitTables();
}

WorldLiteSynth::~WorldLiteSynth() {
  if (owns_fft_plans_) {
    kiss_fftr_free(fwd_);
    kiss_fftr_free(inv_);
  }
}

float WorldLiteSynth::FftSelfTest() {
  // impulse -> forward -> inverse should give N * impulse; return the
  // max abs error against that (0.0 within float noise when healthy).
  for (int i = 0; i < kWorldFftSize; ++i) time_buf_[i] = 0.0f;
  time_buf_[3] = 1.0f;
  kiss_fftr(fwd_, time_buf_, spec_a_);
  kiss_fftri(inv_, spec_a_, time_buf_);
  float max_err = 0.0f;
  for (int i = 0; i < kWorldFftSize; ++i) {
    const float want = (i == 3) ? static_cast<float>(kWorldFftSize) : 0.0f;
    const float e = fabsf(time_buf_[i] - want);
    if (e > max_err) max_err = e;
  }
  return max_err;
}

float WorldLiteSynth::Randn() {
  // 12-uniform-sum gaussian (WORLD matlabfunctions randn()), xorshift32.
  float s = 0.0f;
  for (int i = 0; i < 12; ++i) {
    rng_state_ ^= rng_state_ << 13;
    rng_state_ ^= rng_state_ >> 17;
    rng_state_ ^= rng_state_ << 5;
    s += static_cast<float>(rng_state_) * (1.0f / 4294967296.0f);
  }
  return s - 6.0f;
}

void WorldLiteSynth::ExpandFrame(const WorldFrame& f, float* spec_pow,
                                 float* ap) const {
  for (int i = 0; i < kWorldSpecBins; ++i) {
    const BinMap& mb = benv_map_[i];
    const float amp =
        f.benv[mb.idx] + mb.frac * (f.benv[mb.idx + 1] - f.benv[mb.idx]);
    const float p = amp * amp;
    spec_pow[i] = p > 1e-12f ? p : 1e-12f;
    const BinMap& ma = bap_map_[i];
    float a = f.bap[ma.idx] + ma.frac * (f.bap[ma.idx + 1] - f.bap[ma.idx]);
    if (a < 0.001f) a = 0.001f;
    if (a > 0.999f) a = 0.999f;
    ap[i] = a;
  }
}

void WorldLiteSynth::MinimumPhase(const float* log_amp_half,
                                  kiss_fft_cpx* min_phase) {
  // cepstrum = N * IDFT(log amplitude, hermitian-even)
  WL_CHECKPOINT(80);
  for (int i = 0; i < kWorldSpecBins; ++i) {
    spec_a_[i].r = log_amp_half[i];
    spec_a_[i].i = 0.0f;
  }
  WL_CHECKPOINT(81);
  kiss_fftri(inv_, spec_a_, time_buf_);
  WL_CHECKPOINT(82);
  // fold to causal cepstrum, zero anticausal half
  for (int i = 1; i < kWorldFftSize / 2; ++i) time_buf_[i] *= 2.0f;
  for (int i = kWorldFftSize / 2 + 1; i < kWorldFftSize; ++i)
    time_buf_[i] = 0.0f;
  WL_CHECKPOINT(83);
  kiss_fftr(fwd_, time_buf_, min_phase);
  WL_CHECKPOINT(84);
  // complex exp; the 1/N compensates the unnormalized inverse above
  const float inv_n = 1.0f / kWorldFftSize;
  for (int i = 0; i < kWorldSpecBins; ++i) {
    const float mag = expf(min_phase[i].r * inv_n);
    const float ph = min_phase[i].i * inv_n;
    min_phase[i].r = mag * cosf(ph);
    min_phase[i].i = mag * sinf(ph);
  }
  WL_CHECKPOINT(85);
}

void WorldLiteSynth::RenderPulse(const float* spec_pow, const float* ap,
                                 bool voiced, int noise_size,
                                 float frac_shift_s, float* response) {
  // --- periodic component (voiced only) --------------------------------
  if (voiced) {
    WL_CHECKPOINT(60);
    // reuse periodic_ as the log-amplitude scratch
    for (int i = 0; i < kWorldSpecBins; ++i) {
      const float ap2 = ap[i] * ap[i];
      periodic_[i] = 0.5f * logf(spec_pow[i] * (1.0f - ap2) + kSafeGuardMin);
    }
    WL_CHECKPOINT(61);
    MinimumPhase(periodic_, spec_b_);
    WL_CHECKPOINT(62);
    // fractional time shift as a linear phase ramp (WORLD
    // GetSpectrumWithFractionalTimeShift)
    const float coef =
        2.0f * kPi * frac_shift_s * kWorldSampleRate / kWorldFftSize;
    for (int i = 0; i < kWorldSpecBins; ++i) {
      const float re = spec_b_[i].r, im = spec_b_[i].i;
      const float re2 = cosf(coef * i);
      float s2 = 1.0f - re2 * re2;
      if (s2 < 0.0f) s2 = 0.0f;
      const float im2 = sqrtf(s2);  // sin(theta), theta in [0, pi]
      spec_b_[i].r = re * re2 + im * im2;
      spec_b_[i].i = im * re2 - re * im2;
    }
    WL_CHECKPOINT(63);
    kiss_fftri(inv_, spec_b_, time_buf_);
    WL_CHECKPOINT(64);
    // fftshift
    for (int i = 0; i < kWorldFftSize / 2; ++i) {
      periodic_[i] = time_buf_[i + kWorldFftSize / 2];
      periodic_[i + kWorldFftSize / 2] = time_buf_[i];
    }
    // WORLD RemoveDCComponent: cancel the DC the impulse tail introduces
    float dc = 0.0f;
    for (int i = kWorldFftSize / 2; i < kWorldFftSize; ++i)
      dc += periodic_[i];
    for (int i = 0; i < kWorldFftSize; ++i)
      periodic_[i] -= dc * dc_remover_[i];
  } else {
    memset(periodic_, 0, sizeof(float) * kWorldFftSize);
  }

  // --- aperiodic component ----------------------------------------------
  // NB: MinimumPhase() uses spec_a_ and time_buf_ as scratch, so it must
  // run BEFORE the noise spectrum is produced into spec_a_.
  WL_CHECKPOINT(65);
  for (int i = 0; i < kWorldSpecBins; ++i) {
    const float ap2 = ap[i] * ap[i];
    aperiodic_[i] = voiced
                        ? 0.5f * logf(spec_pow[i] * ap2 + kSafeGuardMin)
                        : 0.5f * logf(spec_pow[i] + kSafeGuardMin);
  }
  WL_CHECKPOINT(66);
  MinimumPhase(aperiodic_, spec_b_);
  WL_CHECKPOINT(67);

  if (noise_size < 1) noise_size = 1;
  float mean = 0.0f;
  for (int i = 0; i < noise_size; ++i) {
    time_buf_[i] = Randn();
    mean += time_buf_[i];
  }
  mean /= noise_size;
  for (int i = 0; i < noise_size; ++i) time_buf_[i] -= mean;
  memset(time_buf_ + noise_size, 0,
         sizeof(float) * (kWorldFftSize - noise_size));
  WL_CHECKPOINT(68);
  kiss_fftr(fwd_, time_buf_, spec_a_);
  WL_CHECKPOINT(69);

  for (int i = 0; i < kWorldSpecBins; ++i) {
    const float re =
        spec_b_[i].r * spec_a_[i].r - spec_b_[i].i * spec_a_[i].i;
    const float im =
        spec_b_[i].r * spec_a_[i].i + spec_b_[i].i * spec_a_[i].r;
    spec_b_[i].r = re;
    spec_b_[i].i = im;
  }
  WL_CHECKPOINT(70);
  kiss_fftri(inv_, spec_b_, time_buf_);
  WL_CHECKPOINT(71);
  for (int i = 0; i < kWorldFftSize / 2; ++i) {
    aperiodic_[i] = time_buf_[i + kWorldFftSize / 2];
    aperiodic_[i + kWorldFftSize / 2] = time_buf_[i];
  }

  // --- mix (WORLD GetOneFrameSegment scaling) ----------------------------
  const float sqrt_noise = sqrtf(static_cast<float>(noise_size));
  const float inv_n = 1.0f / kWorldFftSize;
  for (int i = 0; i < kWorldFftSize; ++i)
    response[i] = (periodic_[i] * sqrt_noise + aperiodic_[i]) * inv_n;
}

void WorldLiteSynth::FlushTo(int abs_pos, float gain, EmitFn emit,
                             void* emit_user) {
  while (flushed_ < abs_pos) {
    int n = abs_pos - flushed_;
    if (n > static_cast<int>(sizeof(emit_buf_) / sizeof(emit_buf_[0])))
      n = sizeof(emit_buf_) / sizeof(emit_buf_[0]);
    for (int i = 0; i < n; ++i) {
      const int ri = (flushed_ + i) % kRingSize;
      float v = ring_[ri] * gain * 32767.0f;
      if (v > 32767.0f) v = 32767.0f;
      if (v < -32768.0f) v = -32768.0f;
      emit_buf_[i] = static_cast<int16_t>(v);
      ring_[ri] = 0.0f;  // ready for the next lap
    }
    emit(emit_user, emit_buf_, n);
    flushed_ += n;
  }
}

void WorldLiteSynth::Synthesize(GetFrameFn get_frame, void* frame_user,
                                int num_frames, float gain, EmitFn emit,
                                void* emit_user) {
  WL_CHECKPOINT(2);
  const int y_length = num_frames * kWorldFrameSamples;
  memset(ring_, 0, sizeof(float) * kRingSize);
  flushed_ = 0;
  WL_CHECKPOINT(3);

  // tmp_frame_ is a member: this callsite now sits above the lazy decode
  // path (get_frame -> TFLM Invoke), where every stack byte counts against
  // the 4 KB scratch-bank stack.
  int cached_t[2] = {-1, -1};
  WorldFrame& tmp_frame = tmp_frame_;
  auto expanded = [&](int t) -> int {
    if (t == cached_t[0]) return 0;
    if (t == cached_t[1]) return 1;
    const int slot = (cached_t[0] <= cached_t[1]) ? 0 : 1;  // evict older
    get_frame(frame_user, t, &tmp_frame);
    ExpandFrame(tmp_frame, cache_pow_[slot], cache_ap_[slot]);
    cached_t[slot] = t;
    return slot;
  };

  int cur_t = -1;
  float f0_a = 0.0f, f0_b = 0.0f;
  float phase = 0.0f;
  float prev_wrapped = 0.0f;

  for (int s = 0; s < y_length; ++s) {
    const float ft = static_cast<float>(s) / kWorldFrameSamples;
    int t0 = static_cast<int>(ft);
    if (t0 > num_frames - 1) t0 = num_frames - 1;
    if (t0 != cur_t) {
      cur_t = t0;
      get_frame(frame_user, t0, &tmp_frame);
      f0_a = tmp_frame.f0;
      if (t0 + 1 < num_frames) {
        get_frame(frame_user, t0 + 1, &tmp_frame);
        f0_b = tmp_frame.f0;
      } else {
        f0_b = f0_a;
      }
      if (f0_a < kF0Floor) f0_a = 0.0f;
      if (f0_b < kF0Floor) f0_b = 0.0f;
    }
    const float w = ft - t0;
    // voicing: WORLD thresholds linearly interpolated 0/1 vuv at 0.5
    const bool va = f0_a > 0.0f, vb = f0_b > 0.0f;
    const bool voiced = (w < 0.5f) ? va : vb;
    float f0;
    if (va && vb) {
      f0 = f0_a + w * (f0_b - f0_a);
    } else if (voiced) {
      f0 = va ? f0_a : f0_b;
    } else {
      f0 = kUnvoicedPulseHz;
    }

    phase += 2.0f * kPi * f0 / kWorldSampleRate;
    const float wrapped = fmodf(phase, 2.0f * kPi);
    const bool pulse = (s > 0) && (fabsf(wrapped - prev_wrapped) > kPi);
    if (!pulse) {
      prev_wrapped = wrapped;
      continue;
    }
    // WORLD GetPulseLocationsForTimeBase: the pulse sits at the sample
    // before the 2pi wrap; fractional offset from the linear crossing
    // y1 + x (y2 - y1) = 0, y1 = wrap[i] - 2pi (< 0), y2 = wrap[i+1].
    const int pulse_index = s - 1;
    const float py1 = prev_wrapped - 2.0f * kPi;
    const float py2 = wrapped;
    const float frac = (py2 > py1) ? (-py1 / (py2 - py1)) : 0.0f;
    const float frac_shift_s = frac / kWorldSampleRate;
    prev_wrapped = wrapped;

    // spectrum at the pulse time: interpolate the two neighboring frames
    // (power domain for the envelope, like WORLD GetSpectralEnvelope)
    const float pt = static_cast<float>(pulse_index) / kWorldFrameSamples;
    int pt0 = static_cast<int>(pt);
    if (pt0 > num_frames - 1) pt0 = num_frames - 1;
    const int pt1 = (pt0 + 1 < num_frames) ? pt0 + 1 : pt0;
    const float pw = pt - pt0;
    const int sa = expanded(pt0);
    const int sb = expanded(pt1);
    for (int i = 0; i < kWorldSpecBins; ++i) {
      pow_i_[i] =
          cache_pow_[sa][i] + pw * (cache_pow_[sb][i] - cache_pow_[sa][i]);
      ap_i_[i] =
          cache_ap_[sa][i] + pw * (cache_ap_[sb][i] - cache_ap_[sa][i]);
    }

    // noise segment: samples to the next pulse ~ one period
    int noise_size = static_cast<int>(kWorldSampleRate / f0 + 0.5f);
    if (noise_size > kWorldFftSize / 2) noise_size = kWorldFftSize / 2;

    // everything before the response's left edge is final: emit it
    const int offset = pulse_index - kWorldFftSize / 2 + 1;
    WL_CHECKPOINT(30);
    if (offset > flushed_) FlushTo(offset, gain, emit, emit_user);

    WL_CHECKPOINT2(static_cast<uint32_t>(pulse_index));
#if defined(PICO_BUILD)
    // Bring-up forensics via the crash-surviving RAM ring (USB stdio is
    // dark during synthesis): stack pointer + kissfft plan checksums per
    // pulse. A changed checksum = heap stomp; a plunging MSP = runaway.
    {
      uint32_t msp;
      __asm volatile("mrs %0, msp" : "=r"(msp));
      uint32_t ck = 0;
      const uint32_t* pf = reinterpret_cast<const uint32_t*>(fwd_);
      const uint32_t* pi = reinterpret_cast<const uint32_t*>(inv_);
      for (int k = 0; k < 32; ++k) {
        ck = ck * 31u + pf[k];
        ck = ck * 31u + pi[k];
      }
      WL_TRACE_RING(1, msp);
      WL_TRACE_RING(2, ck);
      WL_TRACE_RING(3, static_cast<uint32_t>(noise_size) |
                           (voiced ? 0x80000000u : 0u));
    }
#endif
    WL_CHECKPOINT(31);
    RenderPulse(pow_i_, ap_i_, voiced, noise_size, frac_shift_s, response_);
    WL_CHECKPOINT(32);

    int lo = offset < 0 ? -offset : 0;
    int hi = kWorldFftSize;
    if (offset + hi > y_length) hi = y_length - offset;
    for (int j = lo; j < hi; ++j)
      ring_[(j + offset) % kRingSize] += response_[j];
  }

  WL_CHECKPOINT(40);
  FlushTo(y_length, gain, emit, emit_user);
  WL_CHECKPOINT(41);
}

}  // namespace neural_tts
