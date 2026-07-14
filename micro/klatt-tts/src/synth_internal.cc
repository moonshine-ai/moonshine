#include "synth_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>

#include "phonemes.h"

namespace tts {
namespace synth_detail {

namespace {

Segment SegFromPhone(const Phone& p) {
  Segment s;
  s.dur_ms = p.dur_ms;
  s.f1 = p.f1;
  s.f2 = p.f2;
  s.f3 = p.f3;
  s.b1 = p.b1;
  s.b2 = p.b2;
  s.b3 = p.b3;
  s.av = p.av;
  s.af = p.af;
  s.ah = p.ah;
  s.nasal = (p.cls == PhoneClass::kNasal) ? 1.0f : 0.0f;
  s.fnp = p.fnp;
  s.fnz = p.fnz;
  s.fric_cf = p.fric_cf;
  s.is_vowel = (p.cls == PhoneClass::kVowel);
  s.is_silence = (p.cls == PhoneClass::kSilence);
  return s;
}

// Expand a stop into closure -> burst -> (aspiration) sub-segments so that
// voice-onset-time distinguishes /p t k/ from /b d g/.
void AppendStop(const Phone& p, const VoiceParams& vp, int src_token,
                Segment* out, int* n, int max) {
  const bool voiced = (p.src == Source::kVoiced);

  if (*n >= max) return;
  Segment closure = SegFromPhone(p);
  closure.dur_ms =
      voiced ? vp.stop_closure_voiced_ms : vp.stop_closure_voiceless_ms;
  closure.af = 0.0f;
  closure.ah = 0.0f;
  closure.av = voiced ? vp.stop_closure_av : 0.0f;
  closure.f1 = vp.stop_closure_f1;
  closure.src_token = src_token;
  out[(*n)++] = closure;

  if (*n >= max) return;
  Segment burst = SegFromPhone(p);
  burst.dur_ms = vp.stop_burst_ms;
  burst.av = voiced ? vp.stop_burst_av : 0.0f;
  burst.ah = 0.0f;
  burst.src_token = src_token;
  out[(*n)++] = burst;

  if (!voiced) {
    if (*n >= max) return;
    Segment asp = SegFromPhone(p);
    asp.dur_ms = vp.stop_asp_ms;
    asp.av = 0.0f;
    asp.af = 0.0f;
    asp.ah = p.ah;
    asp.src_token = src_token;
    out[(*n)++] = asp;
  }
}

}  // namespace

int BuildSegments(const char* const* phones, int n_phones, const VoiceParams& vp,
                  Segment* out, int max_out) {
  if (phones == nullptr || out == nullptr || max_out <= 0 || n_phones < 0) {
    return -1;
  }
  const Phone* sil = vp.Lookup(" ");
  if (sil == nullptr) return -1;

  int n = 0;
  if (n >= max_out) return -1;
  Segment lead = SegFromPhone(*sil);
  lead.dur_ms = vp.lead_ms;
  out[n++] = lead;

  const char* kPrimary = "\u02C8";
  const char* kSecondary = "\u02CC";
  bool has_explicit_stress = false;
  for (int ti = 0; ti < n_phones; ++ti) {
    const char* tok = phones[ti];
    if (std::strcmp(tok, kPrimary) == 0 || std::strcmp(tok, kSecondary) == 0) {
      has_explicit_stress = true;
      break;
    }
  }

  float pending_accent = 0.0f;
  bool word_needs_accent = true;
  int accents_in_phrase = 0;
  for (int ti = 0; ti < n_phones; ++ti) {
    const char* tok = phones[ti];
    if (std::strcmp(tok, kPrimary) == 0) {
      pending_accent = 1.0f;
      continue;
    }
    if (std::strcmp(tok, kSecondary) == 0) {
      pending_accent = 0.5f;
      continue;
    }

    const Phone* p = vp.Lookup(tok);
    if (p == nullptr) continue;

    if (p->cls == PhoneClass::kSilence) {
      word_needs_accent = true;
      if (n >= max_out) return -1;
      Segment s = SegFromPhone(*p);
      s.major_pause = (p->dur_ms >= kPhraseBreakMs);
      s.src_token = ti;
      if (s.major_pause) accents_in_phrase = 0;
      out[n++] = s;
      continue;
    }

    float accent = 0.0f;
    if (p->cls == PhoneClass::kVowel) {
      if (has_explicit_stress) {
        accent = pending_accent;
      } else if (word_needs_accent) {
        accent = 1.0f;
        word_needs_accent = false;
      }
      if (accent > 0.0f) {
        accent *=
            std::pow(vp.f0_downstep, static_cast<float>(accents_in_phrase));
        ++accents_in_phrase;
      }
    }
    pending_accent = 0.0f;

    if (p->cls == PhoneClass::kStop) {
      AppendStop(*p, vp, ti, out, &n, max_out);
      if (n > max_out) return -1;
    } else {
      if (n >= max_out) return -1;
      Segment s = SegFromPhone(*p);
      s.accent = accent;
      s.src_token = ti;
      out[n++] = s;
    }
  }

  if (n >= max_out) return -1;
  Segment tail = SegFromPhone(*sil);
  tail.dur_ms = vp.tail_ms;
  out[n++] = tail;

  for (int i = 0; i < n; ++i) {
    if (out[i].is_vowel) {
      out[i].dur_ms *= (out[i].accent > 0.0f) ? vp.stress_len_scale
                                                : vp.unstressed_len_scale;
    }
  }
  for (int i = 0; i < n; ++i) {
    if (out[i].is_silence) continue;
    const bool next_boundary = (i + 1 >= n) || out[i + 1].is_silence;
    if (next_boundary) out[i].dur_ms *= vp.prepausal_len_scale;
  }

  return n;
}

std::vector<Segment> BuildSegments(const std::vector<std::string>& phones,
                                   const VoiceParams& vp) {
  std::vector<const char*> ptrs;
  ptrs.reserve(phones.size());
  for (const std::string& p : phones) ptrs.push_back(p.c_str());
  std::vector<Segment> segs;
  segs.resize(phones.size() * 4 + 4);
  const int n = BuildSegments(ptrs.data(), static_cast<int>(ptrs.size()), vp,
                              segs.data(), static_cast<int>(segs.size()));
  if (n < 0) return {};
  segs.resize(static_cast<size_t>(n));
  return segs;
}

void SmoothBidir(float* v, size_t n, float tau_ms) {
  if (n == 0) return;
  const float alpha = std::exp(-kFrameMs / tau_ms);
  for (size_t i = 1; i < n; ++i) {
    v[i] = alpha * v[i - 1] + (1.0f - alpha) * v[i];
  }
  for (size_t i = n - 1; i-- > 0;) {
    v[i] = alpha * v[i + 1] + (1.0f - alpha) * v[i];
  }
}

void SmoothFwd(float* v, size_t n, float tau_ms) {
  if (n == 0) return;
  const float alpha = std::exp(-kFrameMs / tau_ms);
  for (size_t i = 1; i < n; ++i) {
    v[i] = alpha * v[i - 1] + (1.0f - alpha) * v[i];
  }
}

void SmoothAsym(float* v, size_t n, float attack_ms, float release_ms) {
  if (n == 0) return;
  const float a_att = std::exp(-kFrameMs / attack_ms);
  const float a_rel = std::exp(-kFrameMs / release_ms);
  float y = v[0];
  for (size_t i = 1; i < n; ++i) {
    const float target = v[i];
    const float a = (target > y) ? a_att : a_rel;
    y = a * y + (1.0f - a) * target;
    v[i] = y;
  }
}

size_t CountFrames(const std::vector<Segment>& segs, float dur_scale) {
  size_t n = 0;
  for (const Segment& s : segs) {
    int nf = static_cast<int>(std::lround(s.dur_ms * dur_scale / kFrameMs));
    if (nf < 1) nf = 1;
    n += static_cast<size_t>(nf);
  }
  return n;
}

void FillParamTracks(const std::vector<Segment>& segs, const VoiceParams& vp,
                     float dur_scale, bool question, ParamTracks& t) {
  // 2) Rasterize segments into piecewise-constant parameter frames.
  size_t k = 0;
  for (const Segment& s : segs) {
    int nf = static_cast<int>(std::lround(s.dur_ms * dur_scale / kFrameMs));
    if (nf < 1) nf = 1;
    for (int i = 0; i < nf && k < t.n; ++i, ++k) {
      t.f1[k] = s.f1;
      t.f2[k] = s.f2;
      t.f3[k] = s.f3;
      t.b1[k] = s.b1;
      t.b2[k] = s.b2;
      t.b3[k] = s.b3;
      t.av[k] = s.av;
      t.af[k] = s.af;
      t.ah[k] = s.ah;
      t.nasal[k] = s.nasal;
      t.fnp[k] = s.fnp;
      t.fnz[k] = s.fnz;
      t.fric_cf[k] = s.fric_cf;
      t.accent[k] = s.accent;
      t.major[k] = s.major_pause ? 1 : 0;
    }
  }
  const size_t nframes = t.n;
  if (nframes == 0) return;

  // 3) Smooth. Formants zero-phase (coarticulation); amplitudes forward only.
  SmoothBidir(t.f1, nframes, vp.formant_smooth_ms);
  SmoothBidir(t.f2, nframes, vp.formant_smooth_ms);
  SmoothBidir(t.f3, nframes, vp.formant_smooth_ms);
  SmoothFwd(t.av, nframes, vp.av_smooth_ms);
  SmoothAsym(t.af, nframes, vp.af_attack_ms, vp.af_release_ms);
  SmoothFwd(t.ah, nframes, vp.ah_smooth_ms);
  SmoothBidir(t.nasal, nframes, vp.nasal_smooth_ms);
  SmoothBidir(t.accent, nframes, 45.0f);

  // 4) F0 contour. Phrases split at sentence pauses; within each phrase the
  // pitch resets toward a (globally declining) top and declines locally; each
  // phrase end falls (statement) or rises (sentence-final question). Layered:
  // slow flutter + downstepped "hat" pitch accents.
  for (size_t i = 0; i < nframes; ++i) t.f0[i] = vp.f0_end;
  const float denom = static_cast<float>(nframes - 1 ? nframes - 1 : 1);

  auto flutter_at = [&](size_t i) -> float {
    if (vp.f0_flutter_hz <= 0.0f) return 0.0f;
    const float ts = static_cast<float>(i) * (kFrameMs / 1000.0f);
    const float fl = std::sin(2.0f * 3.14159265f * 12.7f * ts) +
                     std::sin(2.0f * 3.14159265f * 7.1f * ts) +
                     std::sin(2.0f * 3.14159265f * 4.7f * ts);
    return vp.f0_flutter_hz * (fl / 3.0f);
  };

  size_t i = 0;
  while (i < nframes) {
    if (t.major[i]) {
      t.f0[i] = vp.f0_end + flutter_at(i);
      ++i;
      continue;
    }
    const size_t start = i;
    while (i < nframes && !t.major[i]) ++i;
    const size_t end = i;  // exclusive
    const size_t len = end - start;
    bool is_last = true;
    for (size_t m = end; m < nframes; ++m) {
      if (!t.major[m] && t.av[m] > 0.0f) {
        is_last = false;
        break;
      }
    }
    for (size_t j = start; j < end; ++j) {
      const float gfrac = static_cast<float>(j) / denom;
      const float lf =
          (len > 1) ? static_cast<float>(j - start) / (len - 1) : 0.0f;
      float v = vp.f0_start + (vp.f0_end - vp.f0_start) * gfrac;
      v -= lf * vp.f0_declination_hz;
      if (lf > 0.8f) {
        const float e = (lf - 0.8f) / 0.2f;
        v += (question && is_last) ? e * vp.f0_question_rise_hz
                                   : -e * vp.final_fall_hz;
      }
      v += flutter_at(j);
      v += vp.f0_accent_hz * t.accent[j];
      t.f0[j] = v;
    }
  }

  // 4b) Voice identity scaling (shorter tract -> higher formants; higher
  // voice).
  if (vp.formant_scale != 1.0f) {
    const float fs = vp.formant_scale;
    for (size_t m = 0; m < nframes; ++m) {
      t.f1[m] *= fs;
      t.f2[m] *= fs;
      t.f3[m] *= fs;
      t.fnp[m] *= fs;
      t.fnz[m] *= fs;
      t.fric_cf[m] *= fs;
    }
  }
  if (vp.f0_scale != 1.0f) {
    for (size_t m = 0; m < nframes; ++m) t.f0[m] *= vp.f0_scale;
  }
}

SynthFrame FrameAt(const ParamTracks& t, size_t i) {
  SynthFrame fr;
  fr.f0 = t.f0[i];
  fr.f1 = t.f1[i];
  fr.f2 = t.f2[i];
  fr.f3 = t.f3[i];
  fr.b1 = t.b1[i];
  fr.b2 = t.b2[i];
  fr.b3 = t.b3[i];
  fr.av = t.av[i];
  fr.af = t.af[i];
  fr.ah = t.ah[i];
  fr.nasal = t.nasal[i];
  fr.fnp = t.fnp[i];
  fr.fnz = t.fnz[i];
  fr.fric_cf = t.fric_cf[i];
  return fr;
}

KlattParams MakeKlattParams(const VoiceParams& vp) {
  KlattParams kp;
  kp.voice_gain = vp.voice_gain;
  kp.fric_gain = vp.fric_gain;
  kp.asp_gain = vp.asp_gain;
  kp.fric_q = vp.fric_q;
  kp.lf_rd = vp.lf_rd;
  kp.tilt_db = vp.source_tilt_db;
  kp.breath = vp.breath;
  kp.jitter = vp.jitter;
  kp.shimmer = vp.shimmer;
  kp.glottal_open = vp.glottal_open;
  kp.glottal_close = vp.glottal_close;
  kp.f4 = vp.f4 * vp.formant_scale;
  kp.b4 = vp.b4;
  kp.f5 = vp.f5 * vp.formant_scale;
  kp.b5 = vp.b5;
  kp.f6 = (vp.f6 > 0.0f) ? vp.f6 * vp.formant_scale : vp.f6;
  kp.b6 = vp.b6;
  kp.bw_f0_coef = vp.bw_f0_coef;
  return kp;
}

}  // namespace synth_detail
}  // namespace tts
