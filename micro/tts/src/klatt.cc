#include "klatt.h"

#include <algorithm>
#include <cmath>

namespace tts {

namespace {

constexpr float kPi = 3.14159265358979323846f;

// Rosenberg-style glottal flow pulse as a function of phase in [0, 1).
// Open phase rises (cos), then a shorter closing phase falls to zero, then a
// closed phase. Its sample-to-sample difference (taken by the caller) is the
// glottal flow derivative that actually excites the cascade.
inline float GlottalPulse(float phase, float open, float close) {
  if (phase < open) {
    return 0.5f * (1.0f - std::cos(kPi * phase / open));
  }
  if (phase < open + close) {
    return std::cos(kPi * (phase - open) / (2.0f * close));
  }
  return 0.0f;
}

// One-pole low-pass coefficient `c` (y = (1-c)x + c*y1) such that the response
// is `tilt_db` down at 3 kHz. Returns 0 (bypass) for tilt_db <= 0. Used for the
// glottal source spectral tilt (Klatt & Klatt 1990 "TL").
float TiltCoef(float tilt_db, float sample_rate) {
  if (tilt_db <= 0.01f) return 0.0f;
  const float g = std::pow(10.0f, -tilt_db / 20.0f);  // target gain at 3 kHz
  const float w = 2.0f * kPi * 3000.0f / sample_rate;
  const float cw = std::cos(w);
  // |H| = (1-c)/sqrt(1 - 2c cos w + c^2) = g  ->  quadratic in c.
  const float A = 1.0f - g * g;
  const float B = -2.0f + 2.0f * g * g * cw;
  const float C = 1.0f - g * g;
  if (std::fabs(A) < 1e-9f) return 0.0f;
  const float disc = B * B - 4.0f * A * C;
  if (disc < 0.0f) return 0.0f;
  const float root = (-B - std::sqrt(disc)) / (2.0f * A);  // the root in [0,1)
  if (root <= 0.0f || root >= 1.0f) return 0.0f;
  return root;
}

}  // namespace

void Resonator::SetParams(float freq_hz, float bw_hz, float sample_rate) {
  const float t = 1.0f / sample_rate;
  c = -std::exp(-2.0f * kPi * bw_hz * t);
  b = 2.0f * std::exp(-kPi * bw_hz * t) * std::cos(2.0f * kPi * freq_hz * t);
  a = 1.0f - b - c;  // unity gain at DC
}

void Biquad::SetBandpass(float freq_hz, float q, float sample_rate) {
  if (q < 0.1f) q = 0.1f;
  const float w0 = 2.0f * kPi * freq_hz / sample_rate;
  const float cw = std::cos(w0);
  const float sw = std::sin(w0);
  const float alpha = sw / (2.0f * q);
  const float a0 = 1.0f + alpha;
  // Constant 0 dB peak-gain band-pass (RBJ cookbook).
  b0 = alpha / a0;
  b1 = 0.0f;
  b2 = -alpha / a0;
  a1 = (-2.0f * cw) / a0;
  a2 = (1.0f - alpha) / a0;
}

void Antiresonator::SetParams(float freq_hz, float bw_hz, float sample_rate) {
  // Build the matching resonator coefficients, then invert to get the zero.
  const float t = 1.0f / sample_rate;
  const float rc = -std::exp(-2.0f * kPi * bw_hz * t);
  const float rb =
      2.0f * std::exp(-kPi * bw_hz * t) * std::cos(2.0f * kPi * freq_hz * t);
  const float ra = 1.0f - rb - rc;
  a = 1.0f / ra;
  b = -rb / ra;
  c = -rc / ra;
}

KlattSynth::KlattSynth(float sample_rate, const KlattParams& params)
    : sample_rate_(sample_rate), p_(params) {
  tilt_c_ = TiltCoef(p_.tilt_db, sample_rate_);
  if (p_.lf_rd > 0.0f) EnsureLfShape(p_.lf_rd);
}

// Map Fant's single Rd parameter to the LF glottal-flow-derivative shape, in
// time normalized to one pitch period (T0 = 1). Rd grows from ~0.3 (tense,
// strong high harmonics) to ~2.7 (lax/breathy, steep spectral roll-off). The
// timing relations are Fant (1995) "The LF-model revisited"; the open-phase
// growth `a` is then solved so the flow derivative integrates to zero over the
// period (the flow returns to baseline). This runs once per Rd value.
void KlattSynth::EnsureLfShape(float rd) {
  if (rd == lf_rd_cached_) return;
  lf_rd_cached_ = rd;
  rd = std::min(2.7f, std::max(0.3f, rd));

  const float ra = (-1.0f + 4.8f * rd) / 100.0f;
  const float rk = (22.4f + 11.8f * rd) / 100.0f;
  const float rg = 0.25f * rk / ((0.11f * rd) / (0.5f + 1.2f * rk) - ra);

  const float tp = 1.0f / (2.0f * rg);  // instant of peak glottal flow
  float te = tp * (1.0f + rk);          // instant of peak excitation (GCI)
  te = std::min(0.95f, std::max(tp + 1e-3f, te));
  const float ta = std::max(1e-4f, ra);  // return-phase time constant
  const float tb = 1.0f - te;            // return-phase duration
  const float wg = kPi / tp;

  // Return-phase decay eps from  eps*ta = 1 - exp(-eps*tb).
  float eps = 1.0f / ta;
  for (int i = 0; i < 24; ++i) {
    eps = (1.0f - std::exp(-eps * tb)) / ta;
  }

  const float sin_te = std::sin(wg * te);
  const float cos_te = std::cos(wg * te);

  // Solve `a` (open-phase growth) for zero net flow change. With E0 = 1:
  //   Aopen(a) = [exp(a te)(a*sin - wg*cos) + wg] / (a^2 + wg^2)
  //   Ee(a)    = -exp(a te) * sin(wg te)            (> 0)
  //   Aret(a)  = -(Ee/(eps*ta)) * (ta - tb*exp(-eps*tb))
  //   f(a)     = Aopen + Aret  -> root via bisection.
  const float exp_eps_tb = std::exp(-eps * tb);
  auto f = [&](float a) {
    const float ea = std::exp(a * te);
    const float aopen =
        (ea * (a * sin_te - wg * cos_te) + wg) / (a * a + wg * wg);
    const float ee = -ea * sin_te;
    const float aret = -(ee / (eps * ta)) * (ta - tb * exp_eps_tb);
    return aopen + aret;
  };
  float lo = -300.0f, hi = 300.0f;
  float flo = f(lo), fhi = f(hi);
  float a = 0.0f;
  if (flo * fhi <= 0.0f) {
    for (int i = 0; i < 80; ++i) {
      const float mid = 0.5f * (lo + hi);
      const float fm = f(mid);
      if (flo * fm <= 0.0f) {
        hi = mid;
        fhi = fm;
      } else {
        lo = mid;
        flo = fm;
      }
    }
    a = 0.5f * (lo + hi);
  }

  float ee = -std::exp(a * te) * sin_te;
  if (!(ee > 1e-6f)) ee = 1.0f;  // guard against a degenerate solve

  lf_a_ = a;
  lf_wg_ = wg;
  lf_te_ = te;
  lf_eps_ = eps;
  lf_ta_ = ta;
  lf_tb_ = tb;
  lf_ee_ = ee;
  lf_exp_eps_tb_ = exp_eps_tb;
}

// Normalized LF flow derivative; the negative excitation peak (at te) == -1.
inline float KlattSynth::LfDeriv(float phase) const {
  if (phase < lf_te_) {
    return std::exp(lf_a_ * phase) * std::sin(lf_wg_ * phase) / lf_ee_;
  }
  return -(1.0f / (lf_eps_ * lf_ta_)) *
         (std::exp(-lf_eps_ * (phase - lf_te_)) - lf_exp_eps_tb_);
}

float KlattSynth::NextNoise() {
  // xorshift32 -> uniform white noise in [-1, 1].
  rng_ ^= rng_ << 13;
  rng_ ^= rng_ >> 17;
  rng_ ^= rng_ << 5;
  return (static_cast<float>(rng_) / 2147483648.0f) - 1.0f;
}

void KlattSynth::RenderFrame(const SynthFrame& cur, const SynthFrame& nxt,
                             int samples_per_frame, float* out) {
  if (samples_per_frame <= 0 || out == nullptr) return;
  const float inv = 1.0f / static_cast<float>(samples_per_frame);

  for (int s = 0; s < samples_per_frame; ++s) {
    const float t = static_cast<float>(s) * inv;  // 0..1 across frame

    // Linearly interpolate the continuous parameters across the frame.
    const float f0 = cur.f0 + (nxt.f0 - cur.f0) * t;
    const float f1 = cur.f1 + (nxt.f1 - cur.f1) * t;
    const float f2 = cur.f2 + (nxt.f2 - cur.f2) * t;
    const float f3 = cur.f3 + (nxt.f3 - cur.f3) * t;
    const float b1 = cur.b1 + (nxt.b1 - cur.b1) * t;
    const float b2 = cur.b2 + (nxt.b2 - cur.b2) * t;
    const float b3 = cur.b3 + (nxt.b3 - cur.b3) * t;
    const float av = cur.av + (nxt.av - cur.av) * t;
    const float af = cur.af + (nxt.af - cur.af) * t;
    const float ah = cur.ah + (nxt.ah - cur.ah) * t;
    const float nasal = cur.nasal + (nxt.nasal - cur.nasal) * t;
    const float fnp = cur.fnp + (nxt.fnp - cur.fnp) * t;
    const float fnz = cur.fnz + (nxt.fnz - cur.fnz) * t;
    const float fric_cf = cur.fric_cf + (nxt.fric_cf - cur.fric_cf) * t;

    // F0-dependent bandwidth widening: bandwidths grow with pitch, which
    // softens the high-pitch buzz and tracks real vocal-tract behavior.
    float bw_scale = 1.0f;
    if (p_.bw_f0_coef > 0.0f && f0 > 100.0f) {
      bw_scale = 1.0f + p_.bw_f0_coef * (f0 - 100.0f) / 100.0f;
    }

    // Recompute resonator coefficients each sample. On an M33 with an FPU
    // this is cheap relative to the rest of the synth; on the desktop it's
    // free. Keeping it per-sample avoids stair-stepping artifacts.
    r1_.SetParams(f1, b1 * bw_scale, sample_rate_);
    r2_.SetParams(f2, b2 * bw_scale, sample_rate_);
    r3_.SetParams(f3, b3 * bw_scale, sample_rate_);
    r4_.SetParams(p_.f4, p_.b4, sample_rate_);
    r5_.SetParams(p_.f5, p_.b5, sample_rate_);
    if (p_.f6 > 0.0f) r6_.SetParams(p_.f6, p_.b6, sample_rate_);

    // --- Voiced source: glottal flow derivative ---
    float voiced = 0.0f;
    float breath = 0.0f;
    if (f0 > 1.0f && av > 0.0f) {
      glottal_phase_ += (f0 / sample_rate_) * cycle_jitter_;
      if (glottal_phase_ >= 1.0f) {
        glottal_phase_ -= 1.0f;
        // New glottal period: redraw jitter (period) and shimmer (amplitude).
        cycle_jitter_ =
            (p_.jitter > 0.0f) ? 1.0f + p_.jitter * NextNoise() : 1.0f;
        cycle_shimmer_ =
            (p_.shimmer > 0.0f) ? 1.0f + p_.shimmer * NextNoise() : 1.0f;
      }
      float exc;
      float open_frac;
      if (p_.lf_rd > 0.0f) {
        exc = LfDeriv(glottal_phase_);  // LF derivative (no differencing)
        open_frac = lf_te_;
      } else {
        const float g =
            GlottalPulse(glottal_phase_, p_.glottal_open, p_.glottal_close);
        exc = g - prev_glottal_;
        prev_glottal_ = g;
        open_frac = p_.glottal_open;
      }
      // Source spectral tilt: one-pole low-pass to soften high harmonics.
      if (tilt_c_ > 0.0f) {
        tilt_y1_ = (1.0f - tilt_c_) * exc + tilt_c_ * tilt_y1_;
        exc = tilt_y1_;
      }
      voiced = exc * av * p_.voice_gain * cycle_shimmer_;
      // Breathiness: glottal-synchronous aspiration, stronger in the open
      // phase, mixed into the source so the cascade shapes it under the
      // formants (Klatt & Klatt's dominant breathiness cue).
      if (p_.breath > 0.0f) {
        const float gate = (glottal_phase_ < open_frac) ? 1.0f : 0.25f;
        breath = NextNoise() * p_.breath * av * gate * p_.voice_gain * 0.5f;
      }
    } else {
      prev_glottal_ = 0.0f;
      tilt_y1_ = 0.0f;
    }

    // --- Aspiration noise (into the cascade) ---
    const float asp = NextNoise() * ah * p_.asp_gain;

    // --- Cascade input, optionally through the nasal branch ---
    float casc = voiced + breath + asp;
    if (nasal > 0.0f) {
      nasal_zero_.SetParams(fnz, 100.0f, sample_rate_);
      nasal_pole_.SetParams(fnp, 100.0f, sample_rate_);
      const float n = nasal_pole_.Step(nasal_zero_.Step(casc));
      casc = casc + nasal * (n - casc);  // blend in nasal coloring
    }

    float x = r1_.Step(casc);
    x = r2_.Step(x);
    x = r3_.Step(x);
    x = r4_.Step(x);
    x = r5_.Step(x);
    if (p_.f6 > 0.0f) x = r6_.Step(x);
    const float cascade_out = x;

    // --- Frication branch: white noise through a true band-pass ---
    float fric_out = 0.0f;
    if (af > 0.0f) {
      fric_bp_.SetBandpass(fric_cf, p_.fric_q, sample_rate_);
      fric_out = fric_bp_.Step(NextNoise()) * af * p_.fric_gain;
    }

    out[s] = cascade_out + fric_out;
  }
}

std::vector<float> KlattSynth::Render(const std::vector<SynthFrame>& frames,
                                      int samples_per_frame) {
  std::vector<float> out;
  if (frames.empty() || samples_per_frame <= 0) {
    return out;
  }
  out.resize(frames.size() * samples_per_frame);

  for (size_t fi = 0; fi < frames.size(); ++fi) {
    const SynthFrame& cur = frames[fi];
    const SynthFrame& nxt = frames[std::min(fi + 1, frames.size() - 1)];
    RenderFrame(cur, nxt, samples_per_frame,
                out.data() + fi * samples_per_frame);
  }

  return out;
}

}  // namespace tts
