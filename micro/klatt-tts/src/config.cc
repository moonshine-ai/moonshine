#include "config.h"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace tts {

namespace {

const char* ClassName(PhoneClass c) {
  switch (c) {
    case PhoneClass::kVowel:
      return "vowel";
    case PhoneClass::kNasal:
      return "nasal";
    case PhoneClass::kStop:
      return "stop";
    case PhoneClass::kFricative:
      return "fricative";
    case PhoneClass::kApproximant:
      return "approximant";
    case PhoneClass::kLateral:
      return "lateral";
    case PhoneClass::kSilence:
      return "silence";
  }
  return "?";
}

const char* SourceName(Source s) {
  switch (s) {
    case Source::kVoiced:
      return "voiced";
    case Source::kVoiceless:
      return "voiceless";
    case Source::kMixed:
      return "mixed";
    case Source::kSilence:
      return "silence";
  }
  return "?";
}

// Map a global parameter name to its storage, so load/dump share one source of
// truth and can't drift apart.
std::unordered_map<std::string, float*> GlobalFields(VoiceParams& vp) {
  return {
      {"voice_gain", &vp.voice_gain},
      {"fric_gain", &vp.fric_gain},
      {"asp_gain", &vp.asp_gain},
      {"fric_q", &vp.fric_q},
      {"lf_rd", &vp.lf_rd},
      {"source_tilt_db", &vp.source_tilt_db},
      {"breath", &vp.breath},
      {"glottal_open", &vp.glottal_open},
      {"glottal_close", &vp.glottal_close},
      {"f4", &vp.f4},
      {"b4", &vp.b4},
      {"f5", &vp.f5},
      {"b5", &vp.b5},
      {"f6", &vp.f6},
      {"b6", &vp.b6},
      {"bw_f0_coef", &vp.bw_f0_coef},
      {"formant_scale", &vp.formant_scale},
      {"f0_scale", &vp.f0_scale},
      {"output_gain", &vp.output_gain},
      {"formant_smooth_ms", &vp.formant_smooth_ms},
      {"av_smooth_ms", &vp.av_smooth_ms},
      {"af_attack_ms", &vp.af_attack_ms},
      {"af_release_ms", &vp.af_release_ms},
      {"ah_smooth_ms", &vp.ah_smooth_ms},
      {"nasal_smooth_ms", &vp.nasal_smooth_ms},
      {"f0_start", &vp.f0_start},
      {"f0_end", &vp.f0_end},
      {"final_fall_hz", &vp.final_fall_hz},
      {"f0_flutter_hz", &vp.f0_flutter_hz},
      {"jitter", &vp.jitter},
      {"shimmer", &vp.shimmer},
      {"f0_accent_hz", &vp.f0_accent_hz},
      {"f0_question_rise_hz", &vp.f0_question_rise_hz},
      {"f0_declination_hz", &vp.f0_declination_hz},
      {"f0_downstep", &vp.f0_downstep},
      {"stress_len_scale", &vp.stress_len_scale},
      {"unstressed_len_scale", &vp.unstressed_len_scale},
      {"prepausal_len_scale", &vp.prepausal_len_scale},
      {"duration_scale", &vp.duration_scale},
      {"lead_ms", &vp.lead_ms},
      {"tail_ms", &vp.tail_ms},
      {"stop_closure_voiced_ms", &vp.stop_closure_voiced_ms},
      {"stop_closure_voiceless_ms", &vp.stop_closure_voiceless_ms},
      {"stop_burst_ms", &vp.stop_burst_ms},
      {"stop_asp_ms", &vp.stop_asp_ms},
      {"stop_closure_av", &vp.stop_closure_av},
      {"stop_burst_av", &vp.stop_burst_av},
      {"stop_closure_f1", &vp.stop_closure_f1},
  };
}

// Apply one "<field> <value>" override to a phone. Returns false if `field` is
// not a recognized numeric field.
bool SetPhoneField(Phone& p, const std::string& field, float v) {
  if (field == "f1")
    p.f1 = v;
  else if (field == "f2")
    p.f2 = v;
  else if (field == "f3")
    p.f3 = v;
  else if (field == "b1")
    p.b1 = v;
  else if (field == "b2")
    p.b2 = v;
  else if (field == "b3")
    p.b3 = v;
  else if (field == "dur")
    p.dur_ms = v;
  else if (field == "fnp")
    p.fnp = v;
  else if (field == "fnz")
    p.fnz = v;
  else if (field == "fric_cf")
    p.fric_cf = v;
  else if (field == "av")
    p.av = v;
  else if (field == "af")
    p.af = v;
  else if (field == "ah")
    p.ah = v;
  else
    return false;
  return true;
}

}  // namespace

const Phone* VoiceParams::Lookup(const std::string& ipa) const {
  for (const Phone& p : phones) {
    if (ipa == p.ipa) return &p;
  }
  return nullptr;
}

VoiceParams DefaultVoiceParams() {
  VoiceParams vp;  // member initializers already hold the global defaults
  vp.phones = DefaultPhoneTable();
  return vp;
}

bool LoadVoiceConfig(const std::string& path, VoiceParams& vp) {
  std::ifstream f(path);
  if (!f) return false;

  auto globals = GlobalFields(vp);
  std::string line;
  int lineno = 0;
  while (std::getline(f, line)) {
    ++lineno;
    // Strip comments.
    const auto hash = line.find('#');
    if (hash != std::string::npos) line.erase(hash);

    std::istringstream ss(line);
    std::string key;
    if (!(ss >> key)) continue;  // blank line

    if (key == "phone") {
      std::string ipa;
      if (!(ss >> ipa)) {
        std::fprintf(stderr, "config:%d: 'phone' needs an IPA key\n", lineno);
        continue;
      }
      if (ipa == "_") ipa = " ";  // the space-keyed silence phone
      // Find the phone (numeric overrides only; inventory is fixed).
      Phone* target = nullptr;
      for (Phone& p : vp.phones) {
        if (ipa == p.ipa) {
          target = &p;
          break;
        }
      }
      if (target == nullptr) {
        std::fprintf(stderr, "config:%d: unknown phone '%s'\n", lineno,
                     ipa.c_str());
        continue;
      }
      std::string field;
      float val;
      while (ss >> field >> val) {
        if (!SetPhoneField(*target, field, val)) {
          std::fprintf(stderr, "config:%d: unknown phone field '%s'\n", lineno,
                       field.c_str());
        }
      }
    } else {
      float val;
      if (!(ss >> val)) {
        std::fprintf(stderr, "config:%d: '%s' needs a value\n", lineno,
                     key.c_str());
        continue;
      }
      auto it = globals.find(key);
      if (it == globals.end()) {
        std::fprintf(stderr, "config:%d: unknown key '%s'\n", lineno,
                     key.c_str());
        continue;
      }
      *it->second = val;
    }
  }
  return true;
}

bool DumpVoiceConfig(const std::string& path, const VoiceParams& vp) {
  std::ofstream f(path);
  if (!f) return false;

  f << "# Formant TTS voice parameters.\n";
  f << "# Generated by `tts --dump-config`. Edit and pass with --config.\n";
  f << "# Globals: '<key> <value>'.  Phones: 'phone <ipa> <field> <value> "
       "...'.\n\n";

  f << "# --- source gains ---\n";
  f << "voice_gain " << vp.voice_gain << "\n";
  f << "fric_gain " << vp.fric_gain << "\n";
  f << "asp_gain " << vp.asp_gain << "\n";
  f << "fric_q " << vp.fric_q << "\n\n";

  f << "# --- voice source (LF model) ---\n";
  f << "lf_rd " << vp.lf_rd
    << "        # LF shape: 0.3 tense .. 2.7 breathy; <=0 = Rosenberg\n";
  f << "source_tilt_db " << vp.source_tilt_db
    << "   # source spectral tilt, dB down at 3 kHz\n";
  f << "breath " << vp.breath
    << "       # open-phase aspiration mixed into voicing, 0..1\n";
  f << "glottal_open " << vp.glottal_open
    << "   # legacy Rosenberg rise (only if lf_rd<=0)\n";
  f << "glottal_close " << vp.glottal_close
    << "  # legacy Rosenberg fall (only if lf_rd<=0)\n\n";

  f << "# --- fixed high formants (f6 = higher-pole correction; <=0 disables) "
       "---\n";
  f << "f4 " << vp.f4 << "\nb4 " << vp.b4 << "\nf5 " << vp.f5 << "\nb5 "
    << vp.b5 << "\nf6 " << vp.f6 << "\nb6 " << vp.b6 << "\n";
  f << "bw_f0_coef " << vp.bw_f0_coef
    << "   # F0-dependent bandwidth widening (0=off)\n\n";

  f << "# --- voice identity (1.0 = tuned male voice) ---\n";
  f << "formant_scale " << vp.formant_scale
    << "   # vocal-tract length: scales all formant freqs\n";
  f << "f0_scale " << vp.f0_scale
    << "        # pitch: scales the whole F0 contour\n";
  f << "output_gain " << vp.output_gain
    << "     # streaming-only fixed makeup gain (batch peak-normalizes)\n\n";

  f << "# --- smoothing time constants (ms) ---\n";
  f << "formant_smooth_ms " << vp.formant_smooth_ms << "\n";
  f << "av_smooth_ms " << vp.av_smooth_ms << "\n";
  f << "af_attack_ms " << vp.af_attack_ms << "\n";
  f << "af_release_ms " << vp.af_release_ms << "\n";
  f << "ah_smooth_ms " << vp.ah_smooth_ms << "\n";
  f << "nasal_smooth_ms " << vp.nasal_smooth_ms << "\n\n";

  f << "# --- prosody ---\n";
  f << "f0_start " << vp.f0_start << "\nf0_end " << vp.f0_end
    << "\nfinal_fall_hz " << vp.final_fall_hz << "\n";
  f << "f0_flutter_hz " << vp.f0_flutter_hz << "   # slow F0 drift amplitude\n";
  f << "jitter " << vp.jitter
    << "          # per-cycle period perturbation, fraction\n";
  f << "shimmer " << vp.shimmer
    << "         # per-cycle amplitude perturbation, fraction\n";
  f << "f0_accent_hz " << vp.f0_accent_hz
    << "     # pitch accent on stressed vowels\n";
  f << "f0_question_rise_hz " << vp.f0_question_rise_hz
    << "  # final rise for '?'\n";
  f << "f0_declination_hz " << vp.f0_declination_hz
    << "    # within-phrase F0 fall\n";
  f << "f0_downstep " << vp.f0_downstep
    << "         # accent step-down per accent in a phrase\n";
  f << "stress_len_scale " << vp.stress_len_scale
    << "     # stressed-vowel lengthening\n";
  f << "unstressed_len_scale " << vp.unstressed_len_scale
    << "  # unstressed-vowel reduction\n";
  f << "prepausal_len_scale " << vp.prepausal_len_scale
    << "   # pre-pause lengthening\n\n";

  f << "# --- segment timing (ms) ---\n";
  f << "duration_scale " << vp.duration_scale << "\n";
  f << "lead_ms " << vp.lead_ms << "\ntail_ms " << vp.tail_ms << "\n";
  f << "stop_closure_voiced_ms " << vp.stop_closure_voiced_ms << "\n";
  f << "stop_closure_voiceless_ms " << vp.stop_closure_voiceless_ms << "\n";
  f << "stop_burst_ms " << vp.stop_burst_ms << "\n";
  f << "stop_asp_ms " << vp.stop_asp_ms << "\n";
  f << "stop_closure_av " << vp.stop_closure_av << "\n";
  f << "stop_burst_av " << vp.stop_burst_av << "\n";
  f << "stop_closure_f1 " << vp.stop_closure_f1 << "\n\n";

  f << "# --- phone table ---\n";
  f << "# phone <ipa>  f1 f2 f3  b1 b2 b3  dur  fnp fnz fric_cf  av af ah   # "
       "class source\n";
  for (const Phone& p : vp.phones) {
    const char* key = (std::string(p.ipa) == " ") ? "_" : p.ipa;
    f << "phone " << key << "  f1 " << p.f1 << " f2 " << p.f2 << " f3 " << p.f3
      << "  b1 " << p.b1 << " b2 " << p.b2 << " b3 " << p.b3 << "  dur "
      << p.dur_ms << "  fnp " << p.fnp << " fnz " << p.fnz << " fric_cf "
      << p.fric_cf << "  av " << p.av << " af " << p.af << " ah " << p.ah
      << "   # " << ClassName(p.cls) << " " << SourceName(p.src) << "\n";
  }
  return static_cast<bool>(f);
}

}  // namespace tts
