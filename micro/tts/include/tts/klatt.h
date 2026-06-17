// Klatt-style cascade formant synthesizer (simplified).
//
// This is the "coral" stage from the design notes: it turns a stream of
// per-frame acoustic parameters into PCM samples. It is intentionally a small
// subset of the full Klatt 1980/1990 model -- enough for "robotic but
// understandable" -- and is written so the same arithmetic ports to the
// RP2350: float-only, fixed small state, no allocation in the inner loop.
//
// Architecture per sample:
//
//   voiced source (glottal flow derivative) ┐
//                                           ├─► [nasal zero]►[nasal pole]
//   aspiration noise ─────────────────────┘        │
//                                                   ▼
//                                   R1 ► R2 ► R3 ► R4 ► R5  (cascade)
//                                                   │
//   frication noise ► [highpass] ► R_fric ──────────┴──►  output
//
// All resonators are the standard Klatt 2-pole form with unity DC gain.

#ifndef TTS_KLATT_H_
#define TTS_KLATT_H_

#include <cstdint>
#include <vector>

namespace tts {

// One control frame of acoustic parameters. Frames are spaced kFrameMs apart
// and the synth linearly interpolates the continuous parameters across each
// frame to avoid zipper noise.
struct SynthFrame {
  float f0 = 0.0f;  // fundamental frequency, Hz (0 => no voicing this frame)
  float f1 = 500.0f, f2 = 1500.0f, f3 = 2500.0f;  // formant freqs, Hz
  float b1 = 60.0f, b2 = 90.0f, b3 = 150.0f;      // formant bandwidths, Hz
  float av = 0.0f;                                // voicing amplitude, 0..1
  float af = 0.0f;                                // frication amplitude, 0..1
  float ah = 0.0f;                                // aspiration amplitude, 0..1
  float nasal = 0.0f;                             // nasal coupling, 0..1
  float fnp = 270.0f;                             // nasal pole freq, Hz
  float fnz = 450.0f;                             // nasal zero freq, Hz
  float fric_cf = 4000.0f;  // frication resonator centre, Hz
};

// A single two-pole resonator (band-pass with unity gain at DC).
struct Resonator {
  float a = 1.0f, b = 0.0f, c = 0.0f;
  float y1 = 0.0f, y2 = 0.0f;

  void SetParams(float freq_hz, float bw_hz, float sample_rate);
  inline float Step(float x) {
    const float y = a * x + b * y1 + c * y2;
    y2 = y1;
    y1 = y;
    return y;
  }
  void Reset() { y1 = y2 = 0.0f; }
};

// Tunable knobs the synth core needs. Populated by the caller from VoiceParams
// (kept as a small standalone struct so klatt.* doesn't depend on config.*).
struct KlattParams {
  float voice_gain = 18.0f;
  float fric_gain = 0.8f;
  float asp_gain = 0.30f;
  float fric_q = 2.2f;
  // Voice source. lf_rd selects the LF (Liljencrants-Fant) glottal-flow
  // derivative shape via Fant's single Rd parameter (~0.3 tense/pressed ..
  // ~2.7 lax/breathy); this replaces the old cosine pulse and is the main
  // naturalness lever. If lf_rd <= 0 the legacy Rosenberg pulse below is used.
  float lf_rd = 1.3f;
  float glottal_open = 0.40f;   // legacy Rosenberg rise (only if lf_rd <= 0)
  float glottal_close = 0.16f;  // legacy Rosenberg fall  (only if lf_rd <= 0)
  // Source spectral tilt: a one-pole low-pass on the glottal source, specified
  // as dB of attenuation at 3 kHz. Tames the harsh/buzzy high harmonics
  // (Klatt & Klatt 1990 "TL"). 0 = no tilt.
  float tilt_db = 0.0f;
  // Breathiness: open-phase-gated aspiration noise mixed into the voiced source
  // (Klatt & Klatt 1990 "Aturb" -- the most important breathiness cue). 0..1.
  float breath = 0.0f;
  // Cycle-to-cycle micro-perturbations applied once per glottal period: jitter
  // perturbs the period length, shimmer the cycle amplitude. Tiny amounts
  // (a few %) defuzz the mechanical periodicity that reads as "robotic".
  float jitter = 0.0f;
  float shimmer = 0.0f;
  float f4 = 3500.0f, b4 = 250.0f;
  float f5 = 4500.0f, b5 = 300.0f;
  // Higher-pole correction: a sixth cascade formant approximating the spectral
  // contribution of the (infinite) formants above F5, flattening the upper
  // spectrum (Klatt 1980 higher-pole correction). 0 disables it.
  float f6 = 5500.0f, b6 = 500.0f;
  // F0-dependent bandwidth widening: at higher pitch the formant bandwidths
  // effectively widen; coefficient scales the widening per 100 Hz above 100 Hz.
  float bw_f0_coef = 0.0f;
};

// A general biquad, used as a band-pass for the frication source. Unlike the
// Klatt Resonator (unity gain at DC, which lets low frequencies leak through),
// a true band-pass puts a zero at DC and Nyquist so frication noise sits in a
// band around its centre frequency instead of becoming broadband hiss.
struct Biquad {
  float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a1 = 0.0f, a2 = 0.0f;
  float x1 = 0.0f, x2 = 0.0f, y1 = 0.0f, y2 = 0.0f;

  // RBJ band-pass with constant 0 dB peak gain (level is independent of Q,
  // so it stays constant as the centre frequency moves between phones).
  void SetBandpass(float freq_hz, float q, float sample_rate);
  inline float Step(float x) {
    const float y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
    x2 = x1;
    x1 = x;
    y2 = y1;
    y1 = y;
    return y;
  }
  void Reset() { x1 = x2 = y1 = y2 = 0.0f; }
};

// A two-zero anti-resonator (notch), used for the nasal zero.
struct Antiresonator {
  float a = 1.0f, b = 0.0f, c = 0.0f;
  float x1 = 0.0f, x2 = 0.0f;

  void SetParams(float freq_hz, float bw_hz, float sample_rate);
  inline float Step(float x) {
    const float y = a * x + b * x1 + c * x2;
    x2 = x1;
    x1 = x;
    return y;
  }
  void Reset() { x1 = x2 = 0.0f; }
};

class KlattSynth {
 public:
  KlattSynth(float sample_rate, const KlattParams& params);

  // Render an entire parameter track. `samples_per_frame` controls how many
  // output samples each SynthFrame spans (= sample_rate * kFrameMs / 1000).
  // Output is mono float, roughly in [-1, 1] (the caller normalizes).
  std::vector<float> Render(const std::vector<SynthFrame>& frames,
                            int samples_per_frame);

  // Streaming primitive: render exactly `samples_per_frame` samples for a
  // single control frame into `out` (caller-owned, no allocation). Parameters
  // are linearly interpolated from `cur` toward `nxt` across the frame, and the
  // resonator/source state persists between calls -- so a caller can drive the
  // synth one frame at a time without ever holding the whole utterance. Pass
  // `nxt == cur` for the final frame. `Render` is a thin loop over this.
  void RenderFrame(const SynthFrame& cur, const SynthFrame& nxt,
                   int samples_per_frame, float* out);

 private:
  float NextNoise();
  // Recompute the normalized LF shape if the Rd parameter changed. Cheap
  // (one numeric solve per Rd value, i.e. usually once per utterance).
  void EnsureLfShape(float rd);
  // Normalized LF flow-derivative at glottal phase in [0,1); negative peak ==
  // -1.
  inline float LfDeriv(float phase) const;

  float sample_rate_;
  KlattParams p_;

  // Cascade vocal-tract resonators (F1..F6). F4..F6 are fixed high formants;
  // F6 is the higher-pole correction.
  Resonator r1_, r2_, r3_, r4_, r5_, r6_;
  // Nasal branch.
  Resonator nasal_pole_;
  Antiresonator nasal_zero_;
  // Frication branch.
  Biquad fric_bp_;

  // Voiced source state.
  float glottal_phase_ = 0.0f;
  float prev_glottal_ = 0.0f;
  // Per-cycle jitter/shimmer factors, redrawn at each period start.
  float cycle_jitter_ = 1.0f;
  float cycle_shimmer_ = 1.0f;

  // Cached LF shape (normalized to one pitch period). Recomputed by
  // EnsureLfShape when lf_rd changes.
  float lf_rd_cached_ = -1.0f;
  float lf_a_ = 0.0f;           // exponential growth of the open phase
  float lf_wg_ = 0.0f;          // pi / Tp (sinusoid rate of the open phase)
  float lf_te_ = 0.5f;          // normalized instant of peak excitation (GCI)
  float lf_eps_ = 1.0f;         // return-phase decay rate
  float lf_ta_ = 0.01f;         // return-phase time constant (== Ra)
  float lf_tb_ = 0.5f;          // 1 - te (return-phase duration)
  float lf_ee_ = 1.0f;          // |open-phase value at te|, used to normalize
  float lf_exp_eps_tb_ = 0.0f;  // exp(-eps * tb), precomputed

  // Source spectral-tilt one-pole state + coefficient (0 = bypass).
  float tilt_c_ = 0.0f;
  float tilt_y1_ = 0.0f;

  // Noise state.
  uint32_t rng_ = 0x1234567u;
};

}  // namespace tts

#endif  // TTS_KLATT_H_
