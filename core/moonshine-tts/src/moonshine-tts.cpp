#include "moonshine-tts.h"

#include <nlohmann/json.h>
#include <onnxruntime_cxx_api.h>

#include "debug-utils.h"
#include "g2p-path.h"
#include "kokoro-voice-levels.h"
#include "moonshine-asset-catalog.h"
#include "moonshine-g2p.h"
#include "ort-session-options.h"
#include "ort-utils-cxx.h"
#include "piper-tts.h"
#include "sentence-splitter.h"
#include "split-weights.h"
#include "string-utils.h"
#include "utf8-utils.h"
#include "zipvoice-tts.h"
#include "zipvoice-voices.h"

extern "C" {
#include <utf8proc.h>
}

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace moonshine_tts {

namespace {

constexpr std::string_view kVoiceMagic = "KVO1";

std::string utf8_nfc(std::string_view s) {
  const std::string tmp(s);
  utf8proc_uint8_t* p =
      utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(tmp.c_str()));
  if (p == nullptr) {
    return std::string(s);
  }
  std::string out(reinterpret_cast<char*>(p));
  std::free(p);
  return out;
}

void replace_utf8(std::string& s, std::string_view old_s,
                  std::string_view new_s) {
  size_t pos = 0;
  while ((pos = s.find(old_s, pos)) != std::string::npos) {
    s.replace(pos, old_s.size(), new_s);
    pos += new_s.size();
  }
}

/// Per-call overrides parsed from ``MoonshineTTS::synthesize`` option pairs.
/// Keys mirror
/// ``MoonshineTTSOptions::parse_options`` (including legacy
/// ``piper_normalize_audio`` /
/// ``piper_output_volume`` aliases for the effects step).
struct SynthesisOverrides {
  std::optional<double> speed;
  std::optional<bool> normalize_audio;
  std::optional<float> output_volume;

  bool empty() const {
    return !speed.has_value() && !normalize_audio.has_value() &&
           !output_volume.has_value();
  }
};

SynthesisOverrides parse_synthesis_overrides_from_pairs(
    const std::vector<std::pair<std::string, std::string>>& pairs) {
  SynthesisOverrides out;
  for (const auto& e : pairs) {
    const std::string key = replace_all(to_lowercase(e.first), "-", "_");
    const std::string value = trim(e.second);
    if (key == "speed") {
      out.speed = static_cast<double>(float_from_string(value.c_str()));
    } else if (key == "normalize_audio" || key == "piper_normalize_audio") {
      out.normalize_audio = bool_from_string(value.c_str());
    } else if (key == "output_volume" || key == "piper_output_volume") {
      out.output_volume = float_from_string(value.c_str());
    }
  }
  return out;
}

bool py_isspace_utf8_ch(std::string_view ch) {
  if (ch.empty()) {
    return false;
  }
  std::string tmp(ch);
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(tmp, 0, cp, adv) || adv != tmp.size()) {
    return false;
  }
  if (cp < 128) {
    return std::isspace(static_cast<unsigned char>(cp)) != 0;
  }
  const auto cat = utf8proc_category(static_cast<utf8proc_int32_t>(cp));
  return cat == UTF8PROC_CATEGORY_ZS || cat == UTF8PROC_CATEGORY_ZL ||
         cat == UTF8PROC_CATEGORY_ZP;
}

std::string collapse_whitespace_join_single_space(const std::string& s) {
  std::u32string u = utf8_str_to_u32(s);
  std::string out;
  bool pending_space = false;
  for (char32_t cp : u) {
    const bool sp =
        (cp < 128 && std::isspace(static_cast<unsigned char>(cp)) != 0) ||
        utf8proc_category(static_cast<utf8proc_int32_t>(cp)) ==
            UTF8PROC_CATEGORY_ZS ||
        utf8proc_category(static_cast<utf8proc_int32_t>(cp)) ==
            UTF8PROC_CATEGORY_ZL ||
        utf8proc_category(static_cast<utf8proc_int32_t>(cp)) ==
            UTF8PROC_CATEGORY_ZP;
    if (sp) {
      if (!out.empty()) {
        pending_space = true;
      }
      continue;
    }
    if (pending_space) {
      utf8_append_codepoint(out, U' ');
      pending_space = false;
    }
    utf8_append_codepoint(out, cp);
  }
  return out;
}

std::string normalize_lang_key(std::string_view raw) {
  std::string s = trim_ascii_ws_copy(raw);
  for (char& c : s) {
    if (c == ' ') {
      c = '_';
    } else if (c == '-') {
      c = '_';
    } else if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

struct LangProfile {
  char kokoro_lang = 'a';
  const char* default_voice = "af_heart";
  /// MoonshineG2P dialect id; nullptr when resolved only via
  /// ``resolve_lang_for_tts`` Spanish fallback.
  const char* g2p_dialect = "en_us";
};

const LangProfile* lookup_lang_profile(std::string_view key) {
  // Keys use underscore form; ``normalize_lang_key`` maps client ``-`` / spaces
  // to ``_`` before lookup.
  static const std::unordered_map<std::string, LangProfile> m{
      {"en_us", {'a', "af_heart", "en_us"}},
      {"en", {'a', "af_heart", "en_us"}},
      // UK Kokoro voice uses the same English rule + ONNX G2P assets as US
      // (``en_us`` under g2p_root).
      {"en_gb", {'b', "bf_emma", "en_gb"}},
      // Spanish G2P must be a concrete dialect id (same default as
      // spanish_rule_g2p.text_to_ipa).
      {"es", {'e', "ef_dora", "es-MX"}},
      {"fr", {'f', "ff_siwis", "fr"}},
      {"hi", {'h', "hf_alpha", "hi"}},
      {"hi_in", {'h', "hf_alpha", "hi"}},
      {"it", {'i', "if_sara", "it"}},
      {"pt_br", {'p', "pf_dora", "pt_br"}},
      {"pt", {'p', "pf_dora", "pt_br"}},
      {"ja", {'j', "jf_alpha", "ja"}},
      {"ja_jp", {'j', "jf_alpha", "ja"}},
      {"jp", {'j', "jf_alpha", "ja"}},
      {"zh", {'z', "zf_xiaobei", "zh"}},
      {"zh_hans", {'z', "zf_xiaobei", "zh"}},
      {"zh_cn", {'z', "zf_xiaobei", "zh"}},
      {"zt", {'z', "zf_xiaobei", "zh"}},
  };
  const std::string k = normalize_lang_key(key);
  const auto it = m.find(k);
  if (it == m.end()) {
    return nullptr;
  }
  return &it->second;
}

/// Fills *profile* and *g2p_dialect* for ``MoonshineG2P`` (Kokoro locale +
/// rule-based tag).
void resolve_lang_for_tts(const std::string& lk, const MoonshineG2POptions& opt,
                          LangProfile& profile, std::string& g2p_dialect) {
  if (const LangProfile* p = lookup_lang_profile(lk)) {
    profile = *p;
    g2p_dialect = p->g2p_dialect;
    return;
  }
  const std::string norm = normalize_rule_based_dialect_cli_key(lk);
  if (!norm.empty() &&
      dialect_resolves_to_spanish_rules(norm, opt.spanish_narrow_obstruents)) {
    profile = {'e', "ef_dora", nullptr};
    g2p_dialect = norm;
    return;
  }
  throw std::runtime_error("MoonshineTTS: unsupported --lang key \"" + lk +
                           "\"");
}

bool kokoro_tts_lang_supported_inner(std::string_view lang_cli,
                                     const MoonshineG2POptions& opt) {
  const std::string lk = normalize_lang_key(lang_cli);
  if (lookup_lang_profile(lk) != nullptr) {
    return true;
  }
  const std::string norm = normalize_rule_based_dialect_cli_key(lk);
  return !norm.empty() &&
         dialect_resolves_to_spanish_rules(norm, opt.spanish_narrow_obstruents);
}

bool voice_prefix_ok(char kokoro_lang, std::string_view voice) {
  static const std::unordered_map<char, std::vector<std::string_view>> pref{
      {'a', {"af_", "am_"}}, {'b', {"bf_", "bm_"}}, {'e', {"ef_", "em_"}},
      {'f', {"ff_"}},        {'h', {"hf_", "hm_"}}, {'i', {"if_", "im_"}},
      {'p', {"pf_", "pm_"}}, {'j', {"jf_", "jm_"}}, {'z', {"zf_", "zm_"}},
  };
  const auto it = pref.find(kokoro_lang);
  if (it == pref.end()) {
    return true;
  }
  for (std::string_view p : it->second) {
    if (voice.size() >= p.size() && voice.substr(0, p.size()) == p) {
      return true;
    }
  }
  return false;
}

/// If ``--lang`` is US English but the user asked for a British Kokoro voice id
/// (``bf_*`` / ``bm_*``), or the reverse, switch the Kokoro profile so
/// ``voice_prefix_ok`` and IPA normalization match the voice pack.
void maybe_align_en_profile_for_kokoro_voice(std::string_view voice,
                                             LangProfile& profile,
                                             std::string& g2p_dialect) {
  if (voice.size() < 3) {
    return;
  }
  const std::string_view p3 = voice.substr(0, 3);
  if (profile.kokoro_lang == 'a' && (p3 == "bf_" || p3 == "bm_")) {
    if (const LangProfile* gb = lookup_lang_profile("en_gb")) {
      profile = *gb;
      g2p_dialect = gb->g2p_dialect;
    }
  } else if (profile.kokoro_lang == 'b' && (p3 == "af_" || p3 == "am_")) {
    if (const LangProfile* us = lookup_lang_profile("en_us")) {
      profile = *us;
      g2p_dialect = us->g2p_dialect;
    }
  }
}

/// When the CLI language is not a Kokoro-backed locale (e.g. ``de`` for
/// Piper-only), but the user selected a Kokoro voice id, infer ``LangProfile``
/// / G2P dialect from the voice stem (``af_river`` → US English).
bool infer_lang_profile_from_kokoro_voice(std::string_view voice_sv,
                                          LangProfile& profile,
                                          std::string& g2p_dialect) {
  const std::string v = trim_ascii_ws_copy(voice_sv);
  if (v.empty()) {
    return false;
  }
  static constexpr const char* k_keys[] = {
      "en_us", "en_gb", "es", "fr", "hi", "it", "pt_br", "ja", "zh", "zh_hans"};
  for (const char* key : k_keys) {
    const LangProfile* p = lookup_lang_profile(key);
    if (p == nullptr) {
      continue;
    }
    if (voice_prefix_ok(p->kokoro_lang, v)) {
      profile = *p;
      g2p_dialect = p->g2p_dialect != nullptr ? std::string(p->g2p_dialect)
                                              : std::string();
      return true;
    }
  }
  return false;
}

/// Like ``resolve_lang_for_tts`` for Kokoro paths, but if *lk* is not
/// Kokoro-capable (Piper-only language), fall back to a profile derived from
/// *voice_for_infer* when non-empty.
void resolve_lang_for_kokoro(const std::string& lk,
                             const MoonshineG2POptions& g2p,
                             LangProfile& profile, std::string& g2p_dialect,
                             std::string_view voice_for_infer) {
  try {
    resolve_lang_for_tts(lk, g2p, profile, g2p_dialect);
  } catch (const std::runtime_error&) {
    if (!infer_lang_profile_from_kokoro_voice(voice_for_infer, profile,
                                              g2p_dialect)) {
      throw;
    }
  }
}

bool kokoro_voice_asset_exists(const std::string& voice_id,
                               const std::filesystem::path& voices_dir,
                               const FileInformationMap* tts_files,
                               const std::filesystem::path& g2p_root) {
  const auto voice_path = [&](const std::string& id) {
    return voices_dir / (id + ".kokorovoice");
  };
  if (tts_files != nullptr) {
    const std::string vk =
        std::string("kokoro/voices/") + voice_id + ".kokorovoice";
    const auto it = tts_files->entries.find(vk);
    if (it != tts_files->entries.end()) {
      const FileInformation& fi = it->second;
      if (fi.memory != nullptr && fi.memory_size > 0) {
        return true;
      }
      if (!fi.path.empty()) {
        const std::filesystem::path p =
            resolve_path_under_root(g2p_root, fi.path);
        return std::filesystem::is_regular_file(p);
      }
    }
  }
  return std::filesystem::is_regular_file(voice_path(voice_id));
}

// Kokoro-82M voice ids (hexgrad/Kokoro-82M VOICES.md). Bundles may ship a
// subset; availability is per asset.
static const char* const kKokoroVoiceCatalog[] = {
    "af_alloy",   "af_aoede",    "af_bella",    "af_heart",    "af_jessica",
    "af_kore",    "af_nicole",   "af_nova",     "af_river",    "af_sarah",
    "af_sky",     "am_adam",     "am_echo",     "am_eric",     "am_fenrir",
    "am_liam",    "am_michael",  "am_onyx",     "am_puck",     "am_santa",
    "bf_alice",   "bf_emma",     "bf_isabella", "bf_lily",     "bm_daniel",
    "bm_fable",   "bm_george",   "bm_lewis",    "ef_dora",     "em_alex",
    "em_santa",   "ff_siwis",    "hf_alpha",    "hf_beta",     "hm_omega",
    "hm_psi",     "if_sara",     "im_nicola",   "jf_alpha",    "jf_gongitsune",
    "jf_nezumi",  "jf_tebukuro", "jm_kumo",     "pf_dora",     "pm_alex",
    "pm_santa",   "zf_xiaobei",  "zf_xiaoni",   "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi",    "zm_yunxia",   "zm_yunyang",
};

std::string select_voice_id(char kokoro_lang, std::string_view requested,
                            std::string_view default_voice,
                            const std::filesystem::path& voices_dir,
                            const FileInformationMap* tts_files,
                            const std::filesystem::path& g2p_root) {
  const auto exists = [&](const std::string& id) {
    return kokoro_voice_asset_exists(id, voices_dir, tts_files, g2p_root);
  };
  auto log_available_kokoro_voices = [&]() {
    std::string available;
    for (const char* vid : kKokoroVoiceCatalog) {
      if (voice_prefix_ok(kokoro_lang, vid) &&
          kokoro_voice_asset_exists(vid, voices_dir, tts_files, g2p_root)) {
        if (!available.empty()) available += ", ";
        available += vid;
      }
    }
    if (available.empty()) {
      LOG("  Available Kokoro voices for this language: (none)");
    } else {
      LOGF("  Available Kokoro voices for this language: %s",
           available.c_str());
    }
  };

  if (!requested.empty()) {
    const std::string req(requested);
    if (voice_prefix_ok(kokoro_lang, req) && exists(req)) {
      return req;
    }
    if (!voice_prefix_ok(kokoro_lang, req)) {
      LOGF("Requested Kokoro voice '%s' has wrong prefix for language '%c'",
           req.c_str(), kokoro_lang);
    } else {
      LOGF("Requested Kokoro voice '%s' not found", req.c_str());
    }
    log_available_kokoro_voices();
  }

  const std::string def(default_voice);
  if (voice_prefix_ok(kokoro_lang, def) && exists(def)) {
    return def;
  }

  for (const char* vid : kKokoroVoiceCatalog) {
    if (!voice_prefix_ok(kokoro_lang, vid)) {
      continue;
    }
    const std::string cand(vid);
    if (!exists(cand)) {
      continue;
    }
    if (cand != def) {
      LOGF("Default Kokoro voice '%s' not found; using '%s' instead",
           def.c_str(), cand.c_str());
    }
    return cand;
  }

  // No matching files on disk: return a valid-prefix id for dependency
  // prefetch, else the default.
  if (!requested.empty()) {
    const std::string req(requested);
    if (voice_prefix_ok(kokoro_lang, req)) {
      return req;
    }
  }
  return def;
}

void apply_diphthong_map(std::string& s, char kokoro_lang) {
  static const std::array<std::pair<const char*, const char*>, 12> kAll{{
      {"t\u0361\u0283", "\u02A7"},  // t͡ʃ → ʧ (U+02A7)
      {"d\u0361\u0292", "\u02A4"},  // d͡ʒ → ʤ (U+02A4)
      {"t\u0283", "\u02A7"},
      {"d\u0292", "\u02A4"},
      {"e\u026a", "A"},
      {"a\u026a", "I"},
      {"a\u028a", "W"},
      {"o\u028a", "O"},
      {"ə\u028a", "Q"},
      {"ɔ\u026a", "Y"},
      {"ɝ", "ɜɹ"},
      {"ɚ", "əɹ"},
  }};
  if (kokoro_lang == 'a' || kokoro_lang == 'b') {
    for (const auto& pr : kAll) {
      replace_utf8(s, pr.first, pr.second);
    }
  } else {
    for (const auto& pr : kAll) {
      if (std::strcmp(pr.first, "ɝ") == 0 || std::strcmp(pr.first, "ɚ") == 0) {
        continue;
      }
      replace_utf8(s, pr.first, pr.second);
    }
  }
}

// Mandarin Chinese IPA normalization for Kokoro: Chao tone letters → arrow
// contour symbols, consonant mappings to Kokoro's inventory (ꭧ for retroflex, ʦ
// for dental affricate, ʨ for palatal), tone repositioning before final nasals.
// Must run before the generic vocab-filter pass.
void apply_chinese_kokoro_normalization(std::string& ipa) {
  // ── Consonant mappings (longest first) ──
  // Retroflex affricates: ʈʂʰ → ꭧʰ, ʈʂ → ꭧ  (ꭧ = U+AB67, \xea\xad\xa7)
  replace_utf8(ipa, "\xca\x88\xca\x82\xca\xb0",
               "\xea\xad\xa7\xca\xb0");                   // ʈʂʰ → ꭧʰ
  replace_utf8(ipa, "\xca\x88\xca\x82", "\xea\xad\xa7");  // ʈʂ → ꭧ

  // Palatal affricates: tɕʰ → ʨʰ, tɕ → ʨ  (ʨ = U+02A8, \xca\xa8)
  replace_utf8(ipa, "t\xc9\x95\xca\xb0", "\xca\xa8\xca\xb0");  // tɕʰ → ʨʰ
  replace_utf8(ipa, "t\xc9\x95", "\xca\xa8");                  // tɕ → ʨ

  // Dental affricates: tsʰ → ʦʰ, ts → ʦ  (ʦ = U+02A6, \xca\xa6)
  replace_utf8(ipa, "ts\xca\xb0", "\xca\xa6\xca\xb0");  // tsʰ → ʦʰ
  replace_utf8(ipa, "ts", "\xca\xa6");                  // ts → ʦ

  // Tie-bar removal (if present): t͡ɕ, t͡s, d͡z
  replace_utf8(ipa, "t\xcd\xa1\xc9\x95", "\xca\xa8");  // t͡ɕ → ʨ
  replace_utf8(ipa, "t\xcd\xa1s", "\xca\xa6");         // t͡s → ʦ

  // ── Vowel/rhoticity mappings ──
  // er/erhua: aɻ → ɚ  (before generic ɻ → ɻ, which is already in Kokoro vocab)
  replace_utf8(ipa, "a\xc9\xbb", "\xc9\x9a");  // aɻ → ɚ

  // Apical vowel after sibilants: ɯ → ɨ
  replace_utf8(ipa, "\xc9\xaf", "\xc9\xa8");  // ɯ → ɨ

  // -uo final: uɔ → wo (Kokoro uses 'wo' not 'uɔ')
  replace_utf8(ipa, "u\xc9\x94", "wo");  // uɔ → wo

  // -eng: ɤŋ → əŋ
  replace_utf8(ipa, "\xc9\xa4\xc5\x8b", "\xc9\x99\xc5\x8b");  // ɤŋ → əŋ

  // ── Chao tone letters → Kokoro arrow symbols ──
  // Multi-letter sequences first (longest first).
  // → = U+2192 (\xe2\x86\x92), ↗ = U+2197 (\xe2\x86\x97), ↓ = U+2193
  // (\xe2\x86\x93), ↘ = U+2198 (\xe2\x86\x98)
  replace_utf8(ipa, "\xcb\xa8\xcb\xa9\xcb\xa6",
               "\xe2\x86\x93");                           // ˨˩˦ (Tone 3) → ↓
  replace_utf8(ipa, "\xcb\xa5\xcb\xa5", "\xe2\x86\x92");  // ˥˥  (Tone 1) → →
  replace_utf8(ipa, "\xcb\xa7\xcb\xa5", "\xe2\x86\x97");  // ˧˥  (Tone 2) → ↗
  replace_utf8(ipa, "\xcb\xa5\xcb\xa9", "\xe2\x86\x98");  // ˥˩  (Tone 4) → ↘
  replace_utf8(ipa, "\xcb\xa9\xcb\xa9", "\xe2\x86\x93");  // ˩˩ → ↓
  replace_utf8(ipa, "\xcb\xa5\xcb\xa7", "\xe2\x86\x98");  // ˥˧ → ↘
  replace_utf8(ipa, "\xcb\xa7\xcb\xa9", "\xe2\x86\x93");  // ˧˩ → ↓
  replace_utf8(ipa, "\xcb\xa8\xcb\xa5", "\xe2\x86\x97");  // ˨˥ → ↗
  // Single remaining tone letters.
  replace_utf8(ipa, "\xcb\xa5", "\xe2\x86\x92");  // ˥ → →
  replace_utf8(ipa, "\xcb\xa6", "\xe2\x86\x92");  // ˦ → →
  replace_utf8(ipa, "\xcb\xa7", "");              // ˧ (neutral) → drop
  replace_utf8(ipa, "\xcb\xa8", "\xe2\x86\x93");  // ˨ → ↓
  replace_utf8(ipa, "\xcb\xa9", "\xe2\x86\x93");  // ˩ → ↓

  // ── Tone repositioning: move tone arrow before final nasals ──
  // Kokoro expects tones between the vowel and final nasal: pa→ŋ, pə↓n
  // Our G2P puts tones after the syllable: pɑŋ˥˥ → pɑŋ→ (after arrow
  // conversion) Need to swap: [nasal][arrow] → [arrow][nasal]
  static const std::string kArrows[] = {"\xe2\x86\x92", "\xe2\x86\x97",
                                        "\xe2\x86\x93", "\xe2\x86\x98"};
  for (const std::string& arrow : kArrows) {
    // Swap n + arrow → arrow + n
    {
      const std::string from = "n" + arrow;
      const std::string to = arrow + "n";
      replace_utf8(ipa, from, to);
    }
    // Swap ŋ + arrow → arrow + ŋ
    {
      const std::string from = "\xc5\x8b" + arrow;  // ŋ + arrow
      const std::string to = arrow + "\xc5\x8b";    // arrow + ŋ
      replace_utf8(ipa, from, to);
    }
  }
}

std::string normalize_ipa_to_kokoro(
    std::string ipa, char kokoro_lang,
    const std::unordered_set<std::string>& vocab_keys) {
  ipa = utf8_nfc(trim_ascii_ws_copy(ipa));
  apply_diphthong_map(ipa, kokoro_lang);
  if (kokoro_lang == 'h') {
    replace_utf8(ipa, ".", "");
    replace_utf8(ipa, "t\u032a", "t");  // t̪
    replace_utf8(ipa, "d\u032a", "d");  // d̪
  }
  if (kokoro_lang == 'z') {
    apply_chinese_kokoro_normalization(ipa);
  }
  std::string kept;
  for (const std::string& ch : utf8_split_codepoints(ipa)) {
    if (vocab_keys.count(ch) != 0 || py_isspace_utf8_ch(ch)) {
      kept += ch;
    }
  }
  return collapse_whitespace_join_single_space(kept);
}

std::vector<std::string> chunk_phonemes(const std::string& ps,
                                        int max_cp = 510) {
  std::vector<std::string> chunks;
  if (ps.empty()) {
    return chunks;
  }
  const std::u32string u = utf8_str_to_u32(ps);
  if (u.size() <= static_cast<size_t>(max_cp)) {
    chunks.push_back(trim_ascii_ws_copy(ps));
    return chunks;
  }
  std::u32string rest = u;
  auto u32_to_utf8 = [](const std::u32string& x) {
    std::string o;
    for (char32_t c : x) {
      utf8_append_codepoint(o, c);
    }
    return o;
  };
  auto trim_u32 = [&u32_to_utf8](std::u32string x) {
    while (!x.empty() && x.front() == U' ') {
      x.erase(x.begin());
    }
    while (!x.empty() && x.back() == U' ') {
      x.pop_back();
    }
    return trim_ascii_ws_copy(u32_to_utf8(x));
  };
  while (!rest.empty()) {
    if (rest.size() <= static_cast<size_t>(max_cp)) {
      const std::string piece = trim_u32(rest);
      if (!piece.empty()) {
        chunks.push_back(piece);
      }
      break;
    }
    const size_t win_len = static_cast<size_t>(max_cp) + 1;
    std::u32string window = rest.substr(0, win_len);
    int cut = -1;
    for (int i = static_cast<int>(window.size()) - 1; i >= 0; --i) {
      if (window[static_cast<size_t>(i)] == U' ') {
        cut = i;
        break;
      }
    }
    if (cut <= 0) {
      cut = max_cp;
    }
    std::u32string piece32 = rest.substr(0, static_cast<size_t>(cut));
    rest = rest.substr(static_cast<size_t>(cut));
    while (!rest.empty() && rest.front() == U' ') {
      rest.erase(rest.begin());
    }
    const std::string piece = trim_u32(piece32);
    if (!piece.empty()) {
      chunks.push_back(piece);
    }
  }
  chunks.erase(std::remove_if(chunks.begin(), chunks.end(),
                              [](const std::string& c) { return c.empty(); }),
               chunks.end());
  return chunks;
}

std::vector<int64_t> phoneme_str_to_input_ids(
    const std::string& phonemes,
    const std::unordered_map<std::string, int>& vocab) {
  std::vector<int64_t> ids;
  ids.push_back(0);
  for (const std::string& ch : utf8_split_codepoints(phonemes)) {
    const auto it = vocab.find(ch);
    if (it != vocab.end()) {
      ids.push_back(it->second);
    }
  }
  ids.push_back(0);
  return ids;
}

void read_kokorovoice_bytes(const uint8_t* data, size_t size,
                            std::string_view context_for_errors,
                            std::vector<float>& out_flat, uint32_t& rows,
                            uint32_t& cols) {
  if (data == nullptr || size < 4 + 4 + 4) {
    throw std::runtime_error("MoonshineTTS: voice buffer too small (" +
                             std::string(context_for_errors) + ")");
  }
  if (std::string_view(reinterpret_cast<const char*>(data), 4) != kVoiceMagic) {
    throw std::runtime_error("MoonshineTTS: bad magic (" +
                             std::string(context_for_errors) +
                             ") (expected KVO1)");
  }
  uint32_t r = 0;
  uint32_t c = 0;
  std::memcpy(&r, data + 4, 4);
  std::memcpy(&c, data + 8, 4);
  if (r == 0 || c == 0) {
    throw std::runtime_error("MoonshineTTS: invalid voice header (" +
                             std::string(context_for_errors) + ")");
  }
  const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
  const size_t need = 12 + n * sizeof(float);
  if (size < need) {
    throw std::runtime_error("MoonshineTTS: truncated voice data (" +
                             std::string(context_for_errors) + ")");
  }
  out_flat.resize(n);
  std::memcpy(out_flat.data(), data + 12, n * sizeof(float));
  rows = r;
  cols = c;
}

void read_kokorovoice(const std::filesystem::path& path,
                      std::vector<float>& out_flat, uint32_t& rows,
                      uint32_t& cols) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("MoonshineTTS: cannot open voice file " +
                             path.string());
  }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)),
                           std::istreambuf_iterator<char>());
  read_kokorovoice_bytes(buf.data(), buf.size(), path.string(), out_flat, rows,
                         cols);
}

}  // namespace

bool kokoro_tts_lang_supported(std::string_view lang_cli,
                               const MoonshineG2POptions& g2p_opt) {
  return kokoro_tts_lang_supported_inner(lang_cli, g2p_opt);
}

std::string ascii_lowercase_copy(std::string_view s) {
  std::string o(s);
  for (char& c : o) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return o;
}

std::filesystem::path tts_map_path(const FileInformationMap& m,
                                   std::string_view canonical_key) {
  const std::string k(canonical_key);
  const auto it = m.entries.find(k);
  if (it == m.entries.end()) {
    return std::filesystem::path(canonical_key);
  }
  return it->second.path;
}

PiperTTSOptions make_piper_options(std::string_view language,
                                   const MoonshineTTSOptions& opt) {
  PiperTTSOptions p;
  p.lang = std::string(language);
  const std::string onnx_key(kTtsPiperOnnxKey);
  if (opt.files.entries.find(onnx_key) != opt.files.entries.end()) {
    const std::filesystem::path rel = opt.tts_relative_path(kTtsPiperOnnxKey);
    if (!rel.empty()) {
      p.explicit_onnx_path = rel;
    }
  }
  const std::string onnx_json_key(kTtsPiperOnnxJsonKey);
  if (opt.files.entries.find(onnx_json_key) != opt.files.entries.end()) {
    const std::filesystem::path jr =
        opt.tts_relative_path(kTtsPiperOnnxJsonKey);
    if (!jr.empty()) {
      p.explicit_onnx_json_path = jr;
    }
  }
  const std::string pv_key(kTtsPiperVoicesKey);
  if (p.explicit_onnx_path.empty() &&
      opt.files.entries.find(pv_key) != opt.files.entries.end()) {
    const std::filesystem::path vr = opt.tts_relative_path(kTtsPiperVoicesKey);
    if (!vr.empty()) {
      p.voices_dir = resolve_path_under_root(opt.g2p_options.g2p_root, vr);
    }
  }
  const std::string pvj_key(kTtsPiperVoicesJsonKey);
  if (p.explicit_onnx_path.empty() &&
      opt.files.entries.find(pvj_key) != opt.files.entries.end()) {
    const std::filesystem::path vjr =
        opt.tts_relative_path(kTtsPiperVoicesJsonKey);
    if (!vjr.empty()) {
      p.voices_json_dir =
          resolve_path_under_root(opt.g2p_options.g2p_root, vjr);
    }
  }
  p.onnx_model = opt.voice;
  p.speed = opt.speed;
  p.g2p_options = opt.g2p_options;
  p.ort_provider_names = opt.ort_provider_names;
  p.coreml_cache_dir = opt.coreml_cache_dir;
  p.normalize_audio = opt.normalize_audio;
  p.output_volume = opt.output_volume;
  p.piper_noise_scale_override = opt.piper_noise_scale_override;
  p.piper_noise_w_override = opt.piper_noise_w_override;
  p.tts_asset_files = opt.files;
  return p;
}

std::vector<std::string> kokoro_vocoder_dependency_keys_with_options(
    std::string_view language, const MoonshineTTSOptions& opt) {
  MoonshineG2POptions g2p = opt.g2p_options;
  if (g2p.g2p_root.empty()) {
    g2p.g2p_root = std::filesystem::current_path();
  }
  LangProfile profile{};
  std::string g2p_dialect;
  const std::string lk = normalize_lang_key(language);
  resolve_lang_for_kokoro(lk, g2p, profile, g2p_dialect, opt.voice);
  maybe_align_en_profile_for_kokoro_voice(opt.voice, profile, g2p_dialect);
  std::filesystem::path model_path = resolve_path_under_root(
      g2p.g2p_root, tts_map_path(opt.files, kTtsKokoroModelKey));
  const std::filesystem::path voices_dir = model_path.parent_path() / "voices";
  // Dependency keys must name the *requested* voice even when the .kokorovoice
  // file is not on disk yet (select_voice_id falls back to the default when
  // missing, which would break download prefetch).
  const std::string req = trim_ascii_ws_copy(opt.voice);
  std::string vid;
  if (!req.empty() && voice_prefix_ok(profile.kokoro_lang, req)) {
    vid = req;
  } else {
    vid = select_voice_id(profile.kokoro_lang, opt.voice, profile.default_voice,
                          voices_dir, &opt.files, g2p.g2p_root);
  }
  // The two stages and no whole-utterance model. Running them back to back
  // produces the same samples that model does, so carrying it as well would
  // double the download for nothing.
  return {std::string(kTtsKokoroProsodyModelKey),
          std::string(kTtsKokoroProsodyWeightsKey),
          std::string(kTtsKokoroDecoderModelKey),
          std::string(kTtsKokoroDecoderWeightsKey),
          std::string(kTtsKokoroConfigJsonKey),
          std::string("kokoro/voices/") + vid + ".kokorovoice"};
}

std::vector<std::pair<std::string, bool>> list_kokoro_voices_with_availability(
    const std::string& lk, const MoonshineTTSOptions& opt) {
  MoonshineG2POptions g2p = opt.g2p_options;
  if (g2p.g2p_root.empty()) {
    g2p.g2p_root = std::filesystem::current_path();
  }
  LangProfile profile{};
  std::string g2p_dialect;
  resolve_lang_for_kokoro(lk, g2p, profile, g2p_dialect, opt.voice);
  maybe_align_en_profile_for_kokoro_voice(opt.voice, profile, g2p_dialect);
  MoonshineTTSOptions opt_scan = opt;
  opt_scan.g2p_options = g2p;
  std::filesystem::path model_path = resolve_path_under_root(
      g2p.g2p_root, tts_map_path(opt_scan.files, kTtsKokoroModelKey));
  const std::filesystem::path voices_dir = model_path.parent_path() / "voices";

  std::map<std::string, bool> by_id;
  for (const char* vid : kKokoroVoiceCatalog) {
    const std::string id(vid);
    if (!voice_prefix_ok(profile.kokoro_lang, id)) {
      continue;
    }
    by_id[id] = kokoro_voice_asset_exists(id, voices_dir, &opt_scan.files,
                                          g2p.g2p_root);
  }

  auto consider_extra = [&](const std::string& id) {
    if (!voice_prefix_ok(profile.kokoro_lang, id)) {
      return;
    }
    if (by_id.find(id) != by_id.end()) {
      return;
    }
    by_id[id] = kokoro_voice_asset_exists(id, voices_dir, &opt_scan.files,
                                          g2p.g2p_root);
  };

  if (std::filesystem::is_directory(voices_dir)) {
    for (const auto& ent : std::filesystem::directory_iterator(voices_dir)) {
      if (!ent.is_regular_file()) {
        continue;
      }
      const std::filesystem::path& p = ent.path();
      if (p.extension() == ".kokorovoice") {
        consider_extra(p.stem().string());
      }
    }
  }
  static const std::string k_prefix = "kokoro/voices/";
  static const std::string k_suffix = ".kokorovoice";
  for (const auto& pr : opt_scan.files.entries) {
    const std::string& key = pr.first;
    if (key.size() <= k_prefix.size() + k_suffix.size()) {
      continue;
    }
    if (key.compare(0, k_prefix.size(), k_prefix) != 0) {
      continue;
    }
    if (key.compare(key.size() - k_suffix.size(), k_suffix.size(), k_suffix) !=
        0) {
      continue;
    }
    consider_extra(key.substr(k_prefix.size(),
                              key.size() - k_prefix.size() - k_suffix.size()));
  }

  std::vector<std::pair<std::string, bool>> out;
  out.reserve(by_id.size());
  for (const auto& pr : by_id) {
    out.emplace_back(pr.first, pr.second);
  }
  return out;
}

std::vector<std::string> piper_vocoder_dependency_keys_with_options(
    std::string_view language, const MoonshineTTSOptions& opt) {
  const std::string onnx_key(kTtsPiperOnnxKey);
  const std::string json_key(kTtsPiperOnnxJsonKey);
  if (opt.files.entries.find(onnx_key) != opt.files.entries.end()) {
    return {onnx_key, json_key};
  }
  MoonshineG2POptions g2p = opt.g2p_options;
  if (g2p.g2p_root.empty()) {
    g2p.g2p_root = std::filesystem::current_path();
  }
  std::vector<std::string> models;
  std::string j;
  if (piper_default_model_bundle_relative_paths(language, g2p, &models, &j,
                                                opt.voice)) {
    models.push_back(std::move(j));
    return models;
  }
  return {};
}

ZipVoiceTTSOptions make_zipvoice_options(std::string_view language,
                                         const MoonshineTTSOptions& opt) {
  ZipVoiceTTSOptions z;
  z.lang = std::string(language);
  z.speed = opt.speed;
  z.g2p_options = opt.g2p_options;
  z.ort_provider_names = opt.ort_provider_names;
  z.coreml_cache_dir = opt.coreml_cache_dir;
  z.normalize_audio = opt.normalize_audio;
  z.output_volume = opt.output_volume;
  z.distill = opt.zipvoice_distill;
  z.num_step = opt.zipvoice_num_step;
  z.guidance_scale = opt.zipvoice_guidance_scale;
  z.t_shift = opt.zipvoice_t_shift;
  z.clone_sample_rate = opt.zipvoice_clone_sample_rate;
  z.clone_transcript = opt.zipvoice_clone_transcript;
  z.tts_asset_files = opt.files;
  z.voice_id = opt.voice;  // engine prefix already stripped
  const auto it =
      opt.files.entries.find(std::string(kTtsZipVoiceCloneAudioKey));
  if (it != opt.files.entries.end() && it->second.memory != nullptr &&
      it->second.memory_size >= sizeof(float)) {
    const size_t n = it->second.memory_size / sizeof(float);
    z.clone_pcm.resize(n);
    std::memcpy(z.clone_pcm.data(), it->second.memory, n * sizeof(float));
  }
  return z;
}

std::vector<std::string> zipvoice_vocoder_dependency_keys() {
  return {std::string(kTtsZipVoiceTextEncoderKey),
          std::string(kTtsZipVoiceFmDecoderKey),
          std::string(kTtsZipVoiceVocoderKey),
          std::string(kTtsZipVoiceTokensKey),
          std::string(kTtsZipVoiceModelJsonKey)};
}

bool zipvoice_asset_present(const MoonshineTTSOptions& opt,
                            std::string_view key) {
  const std::string k(key);
  const auto it = opt.files.entries.find(k);
  if (it != opt.files.entries.end() && it->second.memory != nullptr &&
      it->second.memory_size > 0) {
    return true;
  }
  std::filesystem::path p =
      (it != opt.files.entries.end() && !it->second.path.empty())
          ? resolve_path_under_root(opt.g2p_options.g2p_root, it->second.path)
          : resolve_path_under_root(opt.g2p_options.g2p_root,
                                    std::filesystem::path(k));
  return std::filesystem::is_regular_file(p);
}

bool zipvoice_assets_available(const MoonshineTTSOptions& opt) {
  return zipvoice_asset_present(opt, kTtsZipVoiceTextEncoderKey) &&
         zipvoice_asset_present(opt, kTtsZipVoiceFmDecoderKey) &&
         zipvoice_asset_present(opt, kTtsZipVoiceVocoderKey) &&
         zipvoice_asset_present(opt, kTtsZipVoiceTokensKey);
}

std::vector<std::pair<std::string, bool>>
list_zipvoice_voices_with_availability(const MoonshineTTSOptions& opt) {
  const bool available = zipvoice_assets_available(opt);
  size_t count = 0;
  const ZipVoiceBuiltinVoice* voices = zipvoice_builtin_voices(&count);
  std::vector<std::pair<std::string, bool>> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    out.emplace_back(std::string(voices[i].id), available);
  }
  return out;
}

/// Audio samples one Kokoro prosody frame is worth: 24 kHz at 40 frames per
/// second, fixed by the decoder's upsampling ratio.
inline constexpr int kKokoroSamplesPerFrame = 600;

struct KokoroTtsEngine {
  std::filesystem::path model_path_;
  std::filesystem::path config_path_;
  std::filesystem::path voices_dir_;
  FileInformationMap tts_files_;
  Ort::Env env_ = make_ort_env(ORT_LOGGING_LEVEL_WARNING, "moonshine_tts");
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  /// Float32 weights supplied to the graph on every inference, empty when the
  /// model was loaded in its single-file form. See split-weights.h.
  std::vector<SplitWeight> split_weights_{};

  /// Frame-rate features for one utterance, held between the prosody run and
  /// the decoder runs that consume slices of it.
  struct Prosody {
    std::vector<float> asr{};  ///< [1, channels, frames], row-major
    int64_t channels = 0;
    int frames = 0;
    std::vector<float> f0{};      ///< 2 per frame
    std::vector<float> energy{};  ///< 2 per frame
    std::vector<float> style{};
  };

  /// The same graph cut in two, so a chunk of an utterance can be decoded on
  /// its own. Present only when the stage files were shipped; see load_stages.
  Ort::Session prosody_session_{nullptr};
  Ort::Session decoder_session_{nullptr};
  std::vector<SplitWeight> prosody_weights_{};
  std::vector<SplitWeight> decoder_weights_{};
  bool stages_loaded_ = false;
  /// Whether `session_` holds a whole-utterance model. False for what we
  /// publish, which is the stages and nothing else.
  bool monolith_loaded_ = false;
  /// The utterance currently being decoded a slice at a time. Safe as engine
  /// state because a synthesizer runs one generation at a time.
  Prosody analyzed_{};

  std::unordered_map<std::string, int> vocab_{};
  std::unordered_set<std::string> vocab_keys_{};
  std::vector<float> voice_{};
  uint32_t voice_rows_ = 0;
  uint32_t voice_cols_ = 0;

  std::string voice_id_{};
  double speed_ = 1.0;
  bool normalize_audio_ = true;
  float output_volume_ = 1.F;
  /// ``speed`` ONNX input element type from the loaded graph (FP32 community
  /// ONNX vs double local export).
  ONNXTensorElementDataType speed_elem_type_ =
      ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  char kokoro_lang_ = 'a';
  LangProfile profile_{};
  std::string g2p_dialect_{};
  MoonshineG2POptions g2p_opt_{};
  std::unique_ptr<MoonshineG2P> g2p_{};
  /// Hugging Face ``onnx-community/Kokoro-82M-v1.0-ONNX`` quantized graph names
  /// the style vector ``style``; local torch exports use ``ref_s``.
  std::string style_input_name_ = "ref_s";
  bool log_profiling_ = false;

  ~KokoroTtsEngine() {
    for (auto& e : tts_files_.entries) {
      e.second.free();
    }
  }

  void detect_kokoro_style_input_name() {
    const std::vector<std::string> names = session_.GetInputNames();
    for (const std::string& n : names) {
      if (n == "style") {
        style_input_name_ = "style";
        return;
      }
    }
    style_input_name_ = "ref_s";
  }

  void detect_speed_input_element_type() {
    // Community HF models take speed as a float32 [1]; local torch exports use
    // a double scalar. The split form declares its weights as inputs too, so
    // the name is looked up rather than the usual third position trusted.
    const std::vector<std::string> names = session_.GetInputNames();
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] != "speed") {
        continue;
      }
      Ort::TypeInfo ti = session_.GetInputTypeInfo(i);
      if (ti.GetONNXType() != ONNX_TYPE_TENSOR) {
        return;
      }
      speed_elem_type_ = static_cast<ONNXTensorElementDataType>(
          ti.GetTensorTypeAndShapeInfo().GetElementType());
      return;
    }
  }

  /// Bytes for *key*, or false when neither the file map nor disk has it.
  ///
  /// A caller-supplied buffer wins, then the path the file map gives, then
  /// *shipped_path*, where the asset sits in the layout Moonshine ships.
  bool load_asset_if_present(std::string_view key,
                             const std::filesystem::path& shipped_path,
                             const uint8_t** out, size_t* out_len) {
    const std::string k(key);
    std::filesystem::path path = shipped_path;
    const auto it = tts_files_.entries.find(k);
    if (it != tts_files_.entries.end()) {
      if (it->second.has_memory()) {
        it->second.load(out, out_len);
        return *out_len > 0;
      }
      if (!it->second.path.empty()) {
        const std::filesystem::path mapped =
            resolve_path_under_root(g2p_opt_.g2p_root, it->second.path);
        if (std::filesystem::is_regular_file(mapped)) {
          path = mapped;
        }
      }
    }
    if (!std::filesystem::is_regular_file(path)) {
      return false;
    }
    FileInformation& fi = tts_files_.entries[k];
    fi.path = path;
    fi.load(out, out_len);
    return *out_len > 0;
  }

  void free_asset(std::string_view key) {
    const auto it = tts_files_.entries.find(std::string(key));
    if (it != tts_files_.entries.end()) {
      it->second.free();
    }
  }

  bool asset_in_memory(std::string_view key) const {
    const auto it = tts_files_.entries.find(std::string(key));
    return it != tts_files_.entries.end() && it->second.has_memory();
  }

  /// Opens the whole-utterance Kokoro graph, preferring the split pair.
  ///
  /// What we publish is the two stages, not this, so the usual outcome is that
  /// nothing is found and the caller falls back to running the stages back to
  /// back. It stays because a caller can point ``kokoro_model`` at a model of
  /// their own, which arrives as one graph, and because installs predating the
  /// stages still have the pair on disk.
  ///
  /// Returns whether a model was opened.
  bool load_model(const Ort::SessionOptions& session_opts) {
    std::filesystem::path graph_path = model_path_;
    graph_path.replace_extension(".model.ort");
    std::filesystem::path weights_path = model_path_;
    weights_path.replace_extension(".weights.ort");

    // Bytes handed to us under the anchor key mean that model, so only look
    // for a split pair beside it when the caller supplied no anchor bytes.
    const bool prefer_split = asset_in_memory(kTtsKokoroSplitModelKey) ||
                              !asset_in_memory(kTtsKokoroModelKey);

    const uint8_t* graph_buf = nullptr;
    size_t graph_len = 0;
    const uint8_t* weights_buf = nullptr;
    size_t weights_len = 0;
    if (prefer_split &&
        load_asset_if_present(kTtsKokoroSplitModelKey, graph_path, &graph_buf,
                              &graph_len)) {
      if (load_asset_if_present(kTtsKokoroSplitWeightsKey, weights_path,
                                &weights_buf, &weights_len)) {
        require_ort_model_bytes(graph_buf, graph_len, "Kokoro model");
        require_ort_model_bytes(weights_buf, weights_len, "Kokoro weights");
        // The weights session holds the int8 data and is released as soon as
        // it has produced the float32 tensors, so only those stay resident.
        split_weights_ = run_split_weights_model(env_, weights_buf, weights_len,
                                                 session_opts);
        free_asset(kTtsKokoroSplitWeightsKey);
        session_ = Ort::Session(env_, graph_buf, graph_len, session_opts);
        free_asset(kTtsKokoroSplitModelKey);
        LOGF_IF(log_profiling_,
                "KokoroTtsEngine: split model loaded (%zu + %zu bytes, %zu "
                "weight tensors)",
                graph_len, weights_len, split_weights_.size());
        return true;
      }
      free_asset(kTtsKokoroSplitModelKey);
    }

    const uint8_t* model_buf = nullptr;
    size_t model_len = 0;
    if (!load_asset_if_present(kTtsKokoroModelKey, model_path_, &model_buf,
                               &model_len)) {
      return false;
    }
    require_ort_model_bytes(model_buf, model_len, "Kokoro model");
    session_ = Ort::Session(env_, model_buf, model_len, session_opts);
    free_asset(kTtsKokoroModelKey);
    LOGF_IF(log_profiling_, "KokoroTtsEngine: model loaded (%zu bytes)",
            model_len);
    return true;
  }

  /// Opens the prosody/decoder pair, which is the form Kokoro is published in.
  ///
  /// Still optional: a caller who pointed ``kokoro_model`` at a model of their
  /// own, or an install predating these files, keeps working from the
  /// whole-utterance graph and simply streams a sentence at a time. Nothing
  /// here throws for that reason; the constructor is what insists on finding
  /// one form or the other.
  void load_stages(const Ort::SessionOptions& session_opts) {
    const std::filesystem::path dir = model_path_.parent_path();
    if (!open_stage(session_opts, dir / "prosody.model.ort",
                    dir / "prosody.weights.ort", kTtsKokoroProsodyModelKey,
                    kTtsKokoroProsodyWeightsKey, prosody_session_,
                    prosody_weights_)) {
      return;
    }
    if (!open_stage(session_opts, dir / "decoder.model.ort",
                    dir / "decoder.weights.ort", kTtsKokoroDecoderModelKey,
                    kTtsKokoroDecoderWeightsKey, decoder_session_,
                    decoder_weights_)) {
      prosody_session_ = Ort::Session(nullptr);
      prosody_weights_.clear();
      return;
    }
    stages_loaded_ = true;
    LOGF_IF(log_profiling_,
            "KokoroTtsEngine: prosody/decoder stages loaded (%zu + %zu weight "
            "tensors), sub-sentence streaming available",
            prosody_weights_.size(), decoder_weights_.size());
  }

  bool open_stage(const Ort::SessionOptions& session_opts,
                  const std::filesystem::path& graph_path,
                  const std::filesystem::path& weights_path,
                  std::string_view graph_key, std::string_view weights_key,
                  Ort::Session& out_session,
                  std::vector<SplitWeight>& out_weights) {
    const uint8_t* graph_buf = nullptr;
    size_t graph_len = 0;
    if (!load_asset_if_present(graph_key, graph_path, &graph_buf, &graph_len)) {
      return false;
    }
    const uint8_t* weights_buf = nullptr;
    size_t weights_len = 0;
    if (!load_asset_if_present(weights_key, weights_path, &weights_buf,
                               &weights_len)) {
      free_asset(graph_key);
      return false;
    }
    require_ort_model_bytes(graph_buf, graph_len, "Kokoro stage");
    require_ort_model_bytes(weights_buf, weights_len, "Kokoro stage weights");
    out_weights =
        run_split_weights_model(env_, weights_buf, weights_len, session_opts);
    free_asset(weights_key);
    out_session = Ort::Session(env_, graph_buf, graph_len, session_opts);
    free_asset(graph_key);
    return true;
  }

  /// Whether the decoder can be asked for a range of frames.
  bool supports_slicing() const { return stages_loaded_; }

  /// Run the prosody stage over a whole utterance and keep the result for the
  /// decoder runs that will consume slices of it.
  ///
  /// Reporting zero frames means this utterance cannot be sliced, and the
  /// caller should synthesize it whole instead.
  const Prosody& analyze(std::string_view text) {
    analyzed_ = run_prosody(text);
    return analyzed_;
  }

  /// Decode frames ``[first, last)`` of the utterance `analyze` last saw.
  std::vector<float> decode_analyzed(int first, int last) {
    return run_decoder(analyzed_, first, last);
  }

  Prosody run_prosody(std::string_view text) {
    if (!stages_loaded_) {
      return {};
    }
    const std::string ipa = g2p_->text_to_ipa(text, nullptr);
    if (trim_ascii_ws_copy(ipa).empty()) {
      return {};
    }
    const std::string phonemes =
        normalize_ipa_to_kokoro(ipa, kokoro_lang_, vocab_keys_);
    if (phonemes.empty()) {
      return {};
    }
    std::vector<int64_t> ids = phoneme_str_to_input_ids(phonemes, vocab_);
    // Slicing gains nothing on an utterance the model cannot take in one go,
    // and the caller has a whole-utterance path that handles it.
    if (ids.size() > 512) {
      return {};
    }
    return run_prosody_ids(ids, style_row_for(phonemes));
  }

  /// The row of the voice tensor a phoneme string of this length asks for.
  ///
  /// Kokoro's voices are a table rather than a vector: the style it speaks a
  /// long sentence with is not the one it speaks a short one with. Empty when
  /// the row is past the end of the tensor.
  std::vector<float> style_row_for(const std::string& phonemes) const {
    const std::u32string points = utf8_str_to_u32(phonemes);
    const size_t count = std::max<size_t>(points.size(), 1);
    const size_t row = std::min(
        count - 1, static_cast<size_t>(voice_rows_ > 0 ? voice_rows_ - 1 : 0));
    const size_t offset = row * static_cast<size_t>(voice_cols_);
    if (offset + static_cast<size_t>(voice_cols_) > voice_.size()) {
      return {};
    }
    return std::vector<float>(
        voice_.begin() + static_cast<std::ptrdiff_t>(offset),
        voice_.begin() + static_cast<std::ptrdiff_t>(offset + voice_cols_));
  }

  /// The prosody stage over one run of tokens, whatever produced them.
  Prosody run_prosody_ids(std::vector<int64_t>& ids, std::vector<float> style) {
    Prosody out;
    if (!stages_loaded_ || ids.empty() || style.empty()) {
      return out;
    }
    out.style = std::move(style);

    const int64_t token_count = static_cast<int64_t>(ids.size());
    const std::array<int64_t, 2> shape_ids{1, token_count};
    const std::array<int64_t, 2> shape_style{1,
                                             static_cast<int64_t>(voice_cols_)};
    std::vector<const char*> in_names{"input_ids", "style", "speed"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, ids.data(), ids.size(), shape_ids.data(), shape_ids.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_, out.style.data(), out.style.size(), shape_style.data(),
        shape_style.size()));
    float speed_value = static_cast<float>(speed_);
    const std::array<int64_t, 1> shape_speed{1};
    inputs.push_back(Ort::Value::CreateTensor<float>(mem_, &speed_value, 1,
                                                     shape_speed.data(), 1));
    append_split_weight_inputs(prosody_weights_, mem_, inputs, in_names);

    static const char* stage_out[] = {"asr", "f0", "n"};
    Ort::RunOptions run_opts{nullptr};
    auto outputs = prosody_session_.Run(
        run_opts, in_names.data(), inputs.data(), inputs.size(), stage_out, 3);

    const auto asr_info = outputs[0].GetTensorTypeAndShapeInfo();
    const std::vector<int64_t> asr_shape = asr_info.GetShape();
    if (asr_shape.size() != 3) {
      return {};
    }
    out.channels = asr_shape[1];
    out.frames = static_cast<int>(asr_shape[2]);
    const float* asr_data = outputs[0].GetTensorData<float>();
    out.asr.assign(asr_data, asr_data + asr_info.GetElementCount());
    for (int index = 1; index <= 2; ++index) {
      const auto info = outputs[index].GetTensorTypeAndShapeInfo();
      const float* data = outputs[index].GetTensorData<float>();
      std::vector<float>& target = index == 1 ? out.f0 : out.energy;
      target.assign(data, data + info.GetElementCount());
    }
    return out;
  }

  /// Decode frames ``[first, last)`` of an analyzed utterance.
  std::vector<float> run_decoder(const Prosody& prosody, int first, int last) {
    if (!stages_loaded_ || prosody.frames <= 0) {
      return {};
    }
    first = std::clamp(first, 0, prosody.frames);
    last = std::clamp(last, first, prosody.frames);
    if (last == first) {
      return {};
    }
    const int64_t span = last - first;

    // asr is [1, channels, frames]; a frame range is a column range, so the
    // slice has to be gathered rather than pointed at.
    std::vector<float> asr(static_cast<size_t>(prosody.channels * span));
    for (int64_t channel = 0; channel < prosody.channels; ++channel) {
      const float* source = prosody.asr.data() +
                            channel * static_cast<int64_t>(prosody.frames) +
                            first;
      std::copy(source, source + span,
                asr.begin() + static_cast<std::ptrdiff_t>(channel * span));
    }
    // f0 and n run at twice the frame rate.
    const size_t fine_first = static_cast<size_t>(first) * 2;
    const size_t fine_count = static_cast<size_t>(span) * 2;
    if (prosody.f0.size() < fine_first + fine_count ||
        prosody.energy.size() < fine_first + fine_count) {
      return {};
    }
    std::vector<float> f0(
        prosody.f0.begin() + static_cast<std::ptrdiff_t>(fine_first),
        prosody.f0.begin() +
            static_cast<std::ptrdiff_t>(fine_first + fine_count));
    std::vector<float> energy(
        prosody.energy.begin() + static_cast<std::ptrdiff_t>(fine_first),
        prosody.energy.begin() +
            static_cast<std::ptrdiff_t>(fine_first + fine_count));

    const std::array<int64_t, 3> shape_asr{1, prosody.channels, span};
    const std::array<int64_t, 2> shape_fine{1,
                                            static_cast<int64_t>(fine_count)};
    const std::array<int64_t, 2> shape_style{
        1, static_cast<int64_t>(prosody.style.size())};
    std::vector<float> style = prosody.style;

    std::vector<const char*> in_names{"asr", "f0", "n", "style"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_, asr.data(), asr.size(), shape_asr.data(), shape_asr.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_, f0.data(), f0.size(), shape_fine.data(), shape_fine.size()));
    inputs.push_back(
        Ort::Value::CreateTensor<float>(mem_, energy.data(), energy.size(),
                                        shape_fine.data(), shape_fine.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_, style.data(), style.size(), shape_style.data(),
        shape_style.size()));
    append_split_weight_inputs(decoder_weights_, mem_, inputs, in_names);

    static const char* stage_out[] = {"waveform"};
    Ort::RunOptions run_opts{nullptr};
    auto outputs = decoder_session_.Run(
        run_opts, in_names.data(), inputs.data(), inputs.size(), stage_out, 1);
    const auto info = outputs[0].GetTensorTypeAndShapeInfo();
    const float* data = outputs[0].GetTensorData<float>();
    return std::vector<float>(data, data + info.GetElementCount());
  }

  /// The per-utterance effects the whole-utterance path applies on the way out.
  void apply_output_effects(std::vector<float>& audio) const {
    apply_synthesis_output_effects(audio, normalize_audio_, output_volume_);
  }

  double speed() const { return speed_; }

  void set_speed(double s) {
    if (!(s > 0.0) || !std::isfinite(s)) {
      throw std::runtime_error(
          "MoonshineTTS: speed must be a positive finite number");
    }
    speed_ = s;
  }

  bool normalize_audio() const { return normalize_audio_; }
  void set_normalize_audio(bool on) { normalize_audio_ = on; }
  float output_volume() const { return output_volume_; }
  void set_output_volume(float v) { output_volume_ = v; }

  explicit KokoroTtsEngine(std::string_view language, MoonshineTTSOptions opt) {
    log_profiling_ = opt.log_profiling;
    TIMER_START_IF(log_profiling_, kokoro_engine_init);
    if (!(opt.speed > 0.0) || !std::isfinite(opt.speed)) {
      throw std::runtime_error(
          "MoonshineTTS: speed must be a positive finite number");
    }
    speed_ = opt.speed;
    normalize_audio_ = opt.normalize_audio;
    output_volume_ = opt.output_volume;
    g2p_opt_ = std::move(opt.g2p_options);
    tts_files_ = std::move(opt.files);
    const std::filesystem::path& root = g2p_opt_.g2p_root;
    model_path_ = resolve_path_under_root(
        root, tts_map_path(tts_files_, kTtsKokoroModelKey));
    require_ort_model_path(model_path_, "Kokoro model");
    config_path_ = resolve_path_under_root(
        root, tts_map_path(tts_files_, kTtsKokoroConfigJsonKey));
    voices_dir_ = model_path_.parent_path() / "voices";

    LOGF_IF(log_profiling_, "KokoroTtsEngine: model='%s', config='%s'",
            model_path_.c_str(), config_path_.c_str());

    const auto cit =
        tts_files_.entries.find(std::string(kTtsKokoroConfigJsonKey));
    if (cit == tts_files_.entries.end()) {
      throw std::runtime_error(
          "MoonshineTTS: missing Kokoro file map entry (config key)");
    }
    FileInformation& cfg_fi = cit->second;
    cfg_fi.path = config_path_;

    TIMER_START_IF(log_profiling_, kokoro_load_config);
    const uint8_t* cfg_buf = nullptr;
    size_t cfg_len = 0;
    cfg_fi.load(&cfg_buf, &cfg_len);
    if (cfg_len == 0) {
      cfg_fi.free();
      throw std::runtime_error("MoonshineTTS: empty Kokoro config (" +
                               config_path_.string() + ")");
    }
    {
      const std::string cfg_str(reinterpret_cast<const char*>(cfg_buf),
                                cfg_len);
      nlohmann::json j = nlohmann::json::parse(cfg_str);
      if (!j.contains("vocab") || !j["vocab"].is_object()) {
        cfg_fi.free();
        throw std::runtime_error(
            "MoonshineTTS: config.json missing vocab object");
      }
      for (auto it = j["vocab"].begin(); it != j["vocab"].end(); ++it) {
        const std::string key = it.key();
        vocab_[key] = it.value().get<int>();
        vocab_keys_.insert(key);
      }
    }
    cfg_fi.free();
    LOGF_IF(log_profiling_, "KokoroTtsEngine: loaded vocab with %zu tokens",
            vocab_.size());
    TIMER_END_IF(log_profiling_, kokoro_load_config);

    TIMER_START_IF(log_profiling_, kokoro_load_model);
    const Ort::SessionOptions kokoro_session_options =
        make_ort_session_options(opt.ort_provider_names, opt.coreml_cache_dir);
    monolith_loaded_ = load_model(kokoro_session_options);
    load_stages(kokoro_session_options);
    if (!monolith_loaded_ && !stages_loaded_) {
      const std::filesystem::path dir = model_path_.parent_path();
      throw std::runtime_error(
          "MoonshineTTS: no Kokoro model found. Looked for the stages " +
          (dir / "prosody.model.ort").string() +
          " plus decoder.model.ort and "
          "their weights, and for a whole-utterance model at " +
          model_path_.string() + ".");
    }
    TIMER_END_IF(log_profiling_, kokoro_load_model);

    if (monolith_loaded_) {
      detect_kokoro_style_input_name();
      detect_speed_input_element_type();
    }
    const std::string lk = normalize_lang_key(language);
    resolve_lang_for_kokoro(lk, g2p_opt_, profile_, g2p_dialect_, opt.voice);
    maybe_align_en_profile_for_kokoro_voice(opt.voice, profile_, g2p_dialect_);
    kokoro_lang_ = profile_.kokoro_lang;

    LOGF_IF(
        log_profiling_,
        "KokoroTtsEngine: language='%.*s', g2p_dialect='%s', kokoro_lang='%c'",
        (int)std::string_view(language).size(),
        std::string_view(language).data(), g2p_dialect_.c_str(), kokoro_lang_);

    TIMER_START_IF(log_profiling_, kokoro_g2p_init);
    g2p_ = std::make_unique<MoonshineG2P>(g2p_dialect_, g2p_opt_);
    TIMER_END_IF(log_profiling_, kokoro_g2p_init);

    voice_id_ = select_voice_id(kokoro_lang_, opt.voice, profile_.default_voice,
                                voices_dir_, &tts_files_, g2p_opt_.g2p_root);
    LOGF_IF(log_profiling_, "KokoroTtsEngine: voice='%s', speed=%.2f",
            voice_id_.c_str(), speed_);

    TIMER_START_IF(log_profiling_, kokoro_load_voice);
    reload_voice_tensor();
    LOGF_IF(log_profiling_, "KokoroTtsEngine: voice tensor %ux%u", voice_rows_,
            voice_cols_);
    TIMER_END_IF(log_profiling_, kokoro_load_voice);

    TIMER_END_IF(log_profiling_, kokoro_engine_init);
  }

  void reload_voice_tensor() {
    const std::string vk =
        std::string("kokoro/voices/") + voice_id_ + ".kokorovoice";
    const auto vit = tts_files_.entries.find(vk);
    if (vit != tts_files_.entries.end()) {
      FileInformation& vf = vit->second;
      if (vf.memory == nullptr || vf.memory_size == 0) {
        vf.path = resolve_path_under_root(g2p_opt_.g2p_root,
                                          tts_map_path(tts_files_, vk));
      }
      const uint8_t* vb = nullptr;
      size_t vz = 0;
      vf.load(&vb, &vz);
      read_kokorovoice_bytes(vb, vz, vk, voice_, voice_rows_, voice_cols_);
      vf.free();
      return;
    }
    const auto path = resolve_path_under_root(g2p_opt_.g2p_root,
                                              tts_map_path(tts_files_, vk));
    if (!std::filesystem::is_regular_file(path)) {
      const auto pt = voices_dir_ / (voice_id_ + ".pt");
      std::ostringstream msg;
      msg << "MoonshineTTS: missing voice file " << path.string();
      if (std::filesystem::is_regular_file(pt)) {
        msg << "\n  Export from PyTorch voice pack:\n  python "
               "scripts/export_kokoro_voice_for_cpp.py \""
            << pt.string() << "\" \"" << path.string() << '"';
      } else {
        msg << "\n  Install voices under " << voices_dir_.string()
            << " (e.g. python scripts/download_kokoro_onnx.py --out-dir "
            << model_path_.parent_path().string() << " --voices " << voice_id_
            << "), then export:\n  python "
               "scripts/export_kokoro_voice_for_cpp.py \""
            << (voices_dir_ / (voice_id_ + ".pt")).string() << "\" \""
            << path.string() << '"';
      }
      throw std::runtime_error(msg.str());
    }
    read_kokorovoice(path, voice_, voice_rows_, voice_cols_);
  }

  std::vector<float> synthesize(std::string_view text) {
    TIMER_START_IF(log_profiling_, kokoro_g2p);
    const std::string ipa = g2p_->text_to_ipa(text, nullptr);
    TIMER_END_IF(log_profiling_, kokoro_g2p);
    return synthesize_from_ipa(ipa);
  }

  /// Synthesize from an existing IPA phoneme string (skips G2P). The input is
  /// normalized to Kokoro's phoneme inventory just like the text path, so the
  /// IPA produced by ``MoonshineG2P::text_to_ipa`` /
  /// ``moonshine_text_to_phonemes`` is accepted directly.
  std::vector<float> synthesize_from_ipa(std::string_view ipa) {
    TIMER_START_IF(log_profiling_, kokoro_synthesize);

    if (trim_ascii_ws_copy(ipa).empty()) {
      return {};
    }

    TIMER_START_IF(log_profiling_, kokoro_normalize_ipa);
    std::string phonemes =
        normalize_ipa_to_kokoro(std::string(ipa), kokoro_lang_, vocab_keys_);
    TIMER_END_IF(log_profiling_, kokoro_normalize_ipa);
    if (phonemes.empty()) {
      return {};
    }
    const std::vector<std::string> chunks = chunk_phonemes(phonemes);
    if (chunks.empty()) {
      return {};
    }
    LOGF_IF(log_profiling_,
            "KokoroTtsEngine::synthesize: %zu phoneme chunk(s), "
            "phonemes='%.*s'%s",
            chunks.size(), (int)std::min(phonemes.size(), (size_t)300),
            phonemes.c_str(), phonemes.size() > 300 ? "..." : "");

    std::vector<float> wave_all;
    wave_all.reserve(chunks.size() * 8192);

    for (size_t ci = 0; ci < chunks.size(); ++ci) {
      const std::string& piece = chunks[ci];
      if (trim_ascii_ws_copy(piece).empty()) {
        continue;
      }
      std::vector<int64_t> ids = phoneme_str_to_input_ids(piece, vocab_);
      if (ids.size() > 512) {
        throw std::runtime_error(
            "MoonshineTTS: phoneme token sequence too long for Kokoro (>512)");
      }
      LOGF_IF(log_profiling_,
              "KokoroTtsEngine::synthesize: chunk %zu/%zu, %zu tokens", ci + 1,
              chunks.size(), ids.size());

      std::vector<float> ref_row = style_row_for(piece);
      if (ref_row.empty()) {
        throw std::runtime_error(
            "MoonshineTTS: voice tensor index out of range");
      }

      TIMER_START_IF(log_profiling_, kokoro_onnx_run);
      const std::vector<float> wave =
          monolith_loaded_ ? run_whole_model(ids, ref_row)
                           : run_stages_whole(ids, std::move(ref_row));
      TIMER_END_IF(log_profiling_, kokoro_onnx_run);
      wave_all.insert(wave_all.end(), wave.begin(), wave.end());
      LOGF_IF(log_profiling_,
              "KokoroTtsEngine::synthesize: chunk %zu produced %zu samples",
              ci + 1, wave.size());
    }

    apply_synthesis_output_effects(wave_all, normalize_audio_, output_volume_);

    LOGF_IF(log_profiling_,
            "KokoroTtsEngine::synthesize: total %zu samples (%.2fs at %dHz)",
            wave_all.size(),
            static_cast<double>(wave_all.size()) / MoonshineTTS::kSampleRateHz,
            MoonshineTTS::kSampleRateHz);
    TIMER_END_IF(log_profiling_, kokoro_synthesize);
    return wave_all;
  }

  /// One phoneme chunk through the whole-utterance graph.
  std::vector<float> run_whole_model(std::vector<int64_t>& ids,
                                     std::vector<float>& style) {
    const std::array<int64_t, 2> shape_ids{1, static_cast<int64_t>(ids.size())};
    const std::array<int64_t, 2> shape_style{
        1, static_cast<int64_t>(style.size())};
    std::vector<const char*> in_names{"input_ids", style_input_name_.c_str(),
                                      "speed"};
    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<int64_t>(
        mem_, ids.data(), ids.size(), shape_ids.data(), shape_ids.size()));
    inputs.push_back(Ort::Value::CreateTensor<float>(
        mem_, style.data(), style.size(), shape_style.data(),
        shape_style.size()));
    float speed_f = static_cast<float>(speed_);
    double speed_val = speed_;
    if (speed_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      const std::array<int64_t, 1> shape_speed{1};
      inputs.push_back(Ort::Value::CreateTensor<float>(mem_, &speed_f, 1,
                                                       shape_speed.data(), 1));
    } else {
      inputs.push_back(
          Ort::Value::CreateTensor<double>(mem_, &speed_val, 1, nullptr, 0));
    }
    append_split_weight_inputs(split_weights_, mem_, inputs, in_names);

    static const char* out_names[] = {"waveform"};
    Ort::RunOptions run_opts{nullptr};
    auto outputs = session_.Run(run_opts, in_names.data(), inputs.data(),
                                inputs.size(), out_names, 1);
    const Ort::Value& wav = outputs[0];
    const auto info = wav.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("MoonshineTTS: ONNX output is not float32");
    }
    const float* data = wav.GetTensorData<float>();
    return std::vector<float>(data, data + info.GetElementCount());
  }

  /// One phoneme chunk through the two stages, asking the decoder for every
  /// frame at once.
  ///
  /// This is what the whole-utterance graph does internally, so it returns the
  /// same samples, and it is why that graph no longer has to be downloaded.
  std::vector<float> run_stages_whole(std::vector<int64_t>& ids,
                                      std::vector<float> style) {
    const Prosody prosody = run_prosody_ids(ids, std::move(style));
    if (prosody.frames <= 0) {
      throw std::runtime_error(
          "MoonshineTTS: Kokoro prosody stage produced no frames");
    }
    return run_decoder(prosody, 0, prosody.frames);
  }
};

struct MoonshineTTS::Impl {
  std::unique_ptr<KokoroTtsEngine> kokoro_;
  std::unique_ptr<PiperTTS> piper_;
  std::unique_ptr<ZipVoiceTTS> zipvoice_;
  std::mutex synth_mu_;
  bool log_profiling_ = false;
  std::string language_{};
  ChunkPolicyOptions chunk_policy_{};
  /// The generation in flight, if any. One at a time: a synthesizer has one
  /// model and speaking two things at once is not a thing a caller wants.
  std::unique_ptr<TtsStream> session_{};
  /// Set by `stream_cancel`, cleared by the pull that reports it.
  bool cancel_pending_ = false;

  explicit Impl(std::string_view language, const MoonshineTTSOptions& opt_in) {
    MoonshineTTSOptions opt = opt_in;
    log_profiling_ = opt.log_profiling;
    language_ = std::string(language);
    chunk_policy_.first_chunk_seconds = opt.stream_first_chunk_seconds;
    chunk_policy_.tolerance_seconds = opt.stream_tolerance_seconds;
    chunk_policy_.crossfade_seconds = opt.stream_crossfade_seconds;
    chunk_policy_.growth = opt.stream_growth;
    TIMER_START_IF(log_profiling_, tts_init);
    for (const FileInformation& fi : opt.file_information) {
      const std::string map_key = fi.path.generic_string();
      if (map_key.empty()) {
        continue;
      }
      const bool is_tts_only =
          (map_key.size() >= 7 && map_key.compare(0, 7, "kokoro/") == 0) ||
          (map_key.size() >= 6 && map_key.compare(0, 6, "piper/") == 0) ||
          (map_key.size() >= 9 && map_key.compare(0, 9, "zipvoice/") == 0);
      if (is_tts_only) {
        opt.files.entries[map_key] = fi;
      } else {
        opt.g2p_options.files.entries[map_key] = fi;
      }
    }
    if (opt.g2p_options.g2p_root.empty()) {
      opt.g2p_options.g2p_root = std::filesystem::current_path();
    }
    if (log_profiling_) {
      opt.g2p_options.log_profiling = true;
    }
    opt.apply_voice_engine_prefix();
    std::string eng =
        ascii_lowercase_copy(trim_ascii_ws_copy(opt.vocoder_engine));
    if (eng.empty()) {
      eng = "auto";
    }
    if (eng != "kokoro" && eng != "piper" && eng != "zipvoice" &&
        eng != "auto") {
      throw std::runtime_error(
          "MoonshineTTS: vocoder_engine must be kokoro, piper, zipvoice, or "
          "auto (got \"" +
          eng + "\")");
    }
    const bool kokoro_ok =
        kokoro_tts_lang_supported_inner(language, opt.g2p_options);
    const bool use_zipvoice = (eng == "zipvoice");
    const bool use_kokoro =
        !use_zipvoice && ((eng == "kokoro") || (eng == "auto" && kokoro_ok));

    LOGF_IF(log_profiling_,
            "MoonshineTTS: language='%.*s', vocoder_engine='%s' (resolved=%s), "
            "voice='%s'",
            (int)std::string_view(language).size(),
            std::string_view(language).data(), opt_in.vocoder_engine.c_str(),
            use_zipvoice ? "zipvoice" : (use_kokoro ? "kokoro" : "piper"),
            opt.voice.c_str());

    if (use_zipvoice) {
      zipvoice_ =
          std::make_unique<ZipVoiceTTS>(make_zipvoice_options(language, opt));
    } else if (use_kokoro) {
      kokoro_ = std::make_unique<KokoroTtsEngine>(language, std::move(opt));
    } else {
      TIMER_START_IF(log_profiling_, piper_init);
      piper_ = std::make_unique<PiperTTS>(make_piper_options(language, opt));
      TIMER_END_IF(log_profiling_, piper_init);
    }
    TIMER_END_IF(log_profiling_, tts_init);
  }

  /// Kokoro and Piper cut inside a sentence when their stage models are
  /// installed; everything else streams a sentence at a time.
  ///
  /// An earlier version of this comment said sub-sentence chunking had been
  /// measured and rejected. That rested on a weighted log-mel distance, which
  /// disagrees with how the audio sounds. Judged by word error and by
  /// listening: chunks of about a second are intelligibility-neutral, and
  /// growing each chunk from the one before holds the level steady while
  /// costing less decoder work than a uniform grid, because the padding every
  /// chunk pays for is charged fewer times. See
  /// scripts/kokoro-stream-prototype.py for the measurements.
  ///
  /// Piper splits more cleanly still. Its generator reproduces the whole
  /// render from a padded slice, so its chunks need no crossfade and no
  /// searching for a quiet frame to cut on, and its own `decode_analyzed`
  /// handles the level and the resample to the output rate.
  std::unique_ptr<ChunkSource> make_chunk_source(
      const ChunkPolicyOptions& policy) {
    if (kokoro_ && kokoro_->supports_slicing()) {
      // Only one generation runs at a time, so the utterance being decoded can
      // sit on the engine between the prosody run and the decoder runs that
      // consume it.
      auto analyze = [this](std::string_view text) {
        SlicedDecodeChunkSource::Prosody out;
        const KokoroTtsEngine::Prosody& prosody = kokoro_->analyze(text);
        out.frames = prosody.frames;
        out.f0 = prosody.f0;
        out.energy = prosody.energy;
        return out;
      };
      // The one-shot path normalizes to the finished waveform's peak, which
      // streaming never gets to see. Standing in for it is a gain measured
      // offline for this voice, fixed for the whole utterance so the level
      // cannot lurch between chunks.
      const float gain = kokoro_->normalize_audio()
                             ? kokoro_streaming_gain(kokoro_->voice_id_) *
                                   kokoro_->output_volume()
                             : kokoro_->output_volume();
      auto decode = [this, gain](int first, int last) {
        std::vector<float> audio = kokoro_->decode_analyzed(first, last);
        apply_synthesis_output_effects(audio, /*normalize_audio=*/false, gain);
        return audio;
      };
      auto fallback = [this](std::string_view text) {
        return synthesize_unlocked(text);
      };
      return std::make_unique<SlicedDecodeChunkSource>(
          std::move(analyze), std::move(decode), std::move(fallback), policy,
          kKokoroSamplesPerFrame, MoonshineTTS::kSampleRateHz);
    }
    if (piper_ && piper_->supports_slicing()) {
      ChunkPolicyOptions exact = policy;
      exact.crossfade_seconds = 0.f;
      auto analyze = [this](std::string_view text) {
        return piper_->analyze(text);
      };
      auto decode = [this](int first, int last) {
        return piper_->decode_analyzed(first, last);
      };
      auto fallback = [this](std::string_view text) {
        return synthesize_unlocked(text);
      };
      return std::make_unique<ExactSliceChunkSource>(
          std::move(analyze), std::move(decode), std::move(fallback), exact,
          piper_->frames_per_second(), MoonshineTTS::kSampleRateHz);
    }
    return std::make_unique<WholeUtteranceChunkSource>(
        [this](std::string_view text) { return synthesize_unlocked(text); },
        MoonshineTTS::kSampleRateHz);
  }

  /// The streaming operations `MoonshineTTS` exposes as its own methods.
  ///
  /// The lock is held for the whole of each one, so a binding driving the
  /// stream from a worker thread cannot race a caller on the main thread. The
  /// chunk sources call `synthesize_unlocked` for the same reason.
  void stream_push_text(std::string_view text) {
    std::lock_guard<std::mutex> lock(synth_mu_);
    ensure_session();
    session_->push_text(text);
  }

  void stream_flush() {
    std::lock_guard<std::mutex> lock(synth_mu_);
    ensure_session();
    session_->flush();
  }

  void stream_end_input() {
    std::lock_guard<std::mutex> lock(synth_mu_);
    ensure_session();
    session_->end_input();
  }

  TtsStreamStatus stream_next_chunk(TtsChunk& out) {
    std::lock_guard<std::mutex> lock(synth_mu_);
    if (cancel_pending_) {
      cancel_pending_ = false;
      out = TtsChunk{};
      return TtsStreamStatus::kCancelled;
    }
    if (!session_) {
      return TtsStreamStatus::kNeedText;
    }
    const TtsStreamStatus status = session_->next_chunk(out);
    if (status == TtsStreamStatus::kEndOfStream) {
      session_.reset();
    }
    return status;
  }

  void stream_cancel() {
    std::lock_guard<std::mutex> lock(synth_mu_);
    if (!session_) {
      return;
    }
    session_->cancel();
    session_.reset();
    // Held for the next pull rather than reported here, because whoever
    // cancels is rarely the thread pulling chunks, and that thread has to
    // learn the audio stopped on purpose rather than for want of text.
    cancel_pending_ = true;
  }

  bool is_streaming() {
    std::lock_guard<std::mutex> lock(synth_mu_);
    return session_ != nullptr;
  }

  void ensure_session() {
    if (session_) {
      return;
    }
    SentenceSplitOptions split;
    split.language = language_;
    session_ = std::make_unique<TtsStream>(make_chunk_source(chunk_policy_),
                                           std::move(split));
  }

  std::vector<float> synthesize_unlocked(std::string_view text) {
    if (zipvoice_) {
      return zipvoice_->synthesize(text);
    }
    if (kokoro_) {
      return kokoro_->synthesize(text);
    }
    TIMER_START_IF(log_profiling_, piper_synthesize);
    auto result = piper_->synthesize(text);
    LOGF_IF(log_profiling_, "PiperTTS::synthesize: %zu samples (%.2fs at %dHz)",
            result.size(),
            static_cast<double>(result.size()) / MoonshineTTS::kSampleRateHz,
            MoonshineTTS::kSampleRateHz);
    TIMER_END_IF(log_profiling_, piper_synthesize);
    return result;
  }

  std::vector<float> synthesize_from_phonemes_unlocked(
      std::string_view phonemes) {
    if (zipvoice_) {
      return zipvoice_->synthesize_from_ipa(phonemes);
    }
    if (kokoro_) {
      return kokoro_->synthesize_from_ipa(phonemes);
    }
    return piper_->synthesize_from_ipa(phonemes);
  }

  /// A one-shot call while a generation is streaming would either interleave
  /// model runs with it or silently throw its audio away, so it is refused
  /// instead. Call `cancel_stream` first if the reply is no longer wanted.
  void require_not_streaming() const {
    if (session_) {
      throw std::runtime_error(
          "MoonshineTTS: a streaming generation is in progress. Finish it, or "
          "call cancel_stream(), before synthesizing directly.");
    }
  }

  std::vector<float> synthesize(std::string_view text) {
    std::lock_guard<std::mutex> lock(synth_mu_);
    require_not_streaming();
    return synthesize_unlocked(text);
  }

  std::vector<float> synthesize_from_phonemes(std::string_view phonemes) {
    std::lock_guard<std::mutex> lock(synth_mu_);
    require_not_streaming();
    return synthesize_from_phonemes_unlocked(phonemes);
  }

  std::vector<float> synthesize_from_phonemes_with_overrides(
      std::string_view phonemes, const SynthesisOverrides& ov) {
    return run_with_overrides(
        ov, [&] { return synthesize_from_phonemes_unlocked(phonemes); });
  }

  std::vector<float> synthesize_with_overrides(std::string_view text,
                                               const SynthesisOverrides& ov) {
    return run_with_overrides(ov, [&] { return synthesize_unlocked(text); });
  }

  /// Applies ``ov`` to the active engine, invokes ``produce`` while holding the
  /// synthesis lock, then restores the previous effect settings (even if
  /// ``produce`` throws).
  template <typename Produce>
  std::vector<float> run_with_overrides(const SynthesisOverrides& ov,
                                        Produce&& produce) {
    std::lock_guard<std::mutex> lock(synth_mu_);
    if (zipvoice_) {
      const double prev_speed = zipvoice_->speed();
      const bool prev_normalize = zipvoice_->normalize_audio();
      const float prev_volume = zipvoice_->output_volume();
      const auto apply_zv = [&](double speed, bool normalize, float volume) {
        zipvoice_->set_speed(speed);
        zipvoice_->set_normalize_audio(normalize);
        zipvoice_->set_output_volume(volume);
      };
      apply_zv(ov.speed.value_or(prev_speed),
               ov.normalize_audio.value_or(prev_normalize),
               ov.output_volume.value_or(prev_volume));
      try {
        std::vector<float> wave = produce();
        apply_zv(prev_speed, prev_normalize, prev_volume);
        return wave;
      } catch (...) {
        apply_zv(prev_speed, prev_normalize, prev_volume);
        throw;
      }
    }
    const double prev_speed = kokoro_ ? kokoro_->speed() : piper_->speed();
    const bool prev_normalize =
        kokoro_ ? kokoro_->normalize_audio() : piper_->normalize_audio();
    const float prev_volume =
        kokoro_ ? kokoro_->output_volume() : piper_->output_volume();
    const auto apply = [&](double speed, bool normalize, float volume) {
      if (kokoro_) {
        kokoro_->set_speed(speed);
        kokoro_->set_normalize_audio(normalize);
        kokoro_->set_output_volume(volume);
      } else {
        piper_->set_speed(speed);
        piper_->set_normalize_audio(normalize);
        piper_->set_output_volume(volume);
      }
    };
    apply(ov.speed.value_or(prev_speed),
          ov.normalize_audio.value_or(prev_normalize),
          ov.output_volume.value_or(prev_volume));
    try {
      std::vector<float> wave = produce();
      apply(prev_speed, prev_normalize, prev_volume);
      return wave;
    } catch (...) {
      apply(prev_speed, prev_normalize, prev_volume);
      throw;
    }
  }
};

MoonshineTTS::MoonshineTTS(std::string_view language,
                           const MoonshineTTSOptions& opt)
    : impl_(std::make_unique<Impl>(language, opt)) {}

MoonshineTTS::~MoonshineTTS() = default;

MoonshineTTS::MoonshineTTS(MoonshineTTS&&) noexcept = default;
MoonshineTTS& MoonshineTTS::operator=(MoonshineTTS&&) noexcept = default;

std::vector<float> MoonshineTTS::synthesize(std::string_view text) {
  return impl_->synthesize(text);
}

std::vector<float> MoonshineTTS::synthesize(
    std::string_view text,
    const std::vector<std::pair<std::string, std::string>>& option_overrides) {
  if (option_overrides.empty()) {
    return synthesize(text);
  }
  const SynthesisOverrides ov =
      parse_synthesis_overrides_from_pairs(option_overrides);
  if (ov.empty()) {
    return synthesize(text);
  }
  return impl_->synthesize_with_overrides(text, ov);
}

std::vector<float> MoonshineTTS::synthesize_from_phonemes(
    std::string_view phonemes) {
  return impl_->synthesize_from_phonemes(phonemes);
}

std::vector<float> MoonshineTTS::synthesize_from_phonemes(
    std::string_view phonemes,
    const std::vector<std::pair<std::string, std::string>>& option_overrides) {
  if (option_overrides.empty()) {
    return synthesize_from_phonemes(phonemes);
  }
  const SynthesisOverrides ov =
      parse_synthesis_overrides_from_pairs(option_overrides);
  if (ov.empty()) {
    return synthesize_from_phonemes(phonemes);
  }
  return impl_->synthesize_from_phonemes_with_overrides(phonemes, ov);
}

void MoonshineTTS::push_text(std::string_view text) {
  impl_->stream_push_text(text);
}

void MoonshineTTS::flush() { impl_->stream_flush(); }

void MoonshineTTS::end_input() { impl_->stream_end_input(); }

TtsStreamStatus MoonshineTTS::next_chunk(TtsChunk& out) {
  return impl_->stream_next_chunk(out);
}

void MoonshineTTS::cancel_stream() { impl_->stream_cancel(); }

bool MoonshineTTS::is_streaming() const { return impl_->is_streaming(); }

std::vector<std::string> MoonshineTTS::split_utterances(
    std::string_view text) const {
  SentenceSplitOptions split;
  split.language = impl_->language_;
  return split_sentences(text, split);
}

void write_wav_mono_pcm16(const std::filesystem::path& path,
                          const std::vector<float>& samples) {
  // parent_path() is empty for plain filenames like "out.wav";
  // create_directories("") throws on some libstdc++.
  const std::filesystem::path parent = path.parent_path();
  if (!parent.empty()) {
    std::filesystem::create_directories(parent);
  }
  std::vector<int16_t> pcm(samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    float x = samples[i];
    if (!std::isfinite(x)) {
      x = 0.f;
    }
    x = std::max(-1.f, std::min(1.f, x));
    pcm[i] = static_cast<int16_t>(std::lrint(x * 32767.f));
  }
  const uint32_t sample_rate =
      static_cast<uint32_t>(MoonshineTTS::kSampleRateHz);
  const uint32_t num_samples = static_cast<uint32_t>(pcm.size());
  const uint32_t byte_rate = sample_rate * 2;
  const uint16_t block_align = 2;
  const uint32_t data_bytes = num_samples * 2;
  const uint32_t riff_chunk_size = 36 + data_bytes;

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("write_wav_mono_pcm16: cannot open " +
                             path.string());
  }
  auto w4 = [&out](const char* s) { out.write(s, 4); };
  auto u32 = [&out](uint32_t v) {
    char b[4];
    b[0] = static_cast<char>(v & 0xff);
    b[1] = static_cast<char>((v >> 8) & 0xff);
    b[2] = static_cast<char>((v >> 16) & 0xff);
    b[3] = static_cast<char>((v >> 24) & 0xff);
    out.write(b, 4);
  };
  auto u16 = [&out](uint16_t v) {
    char b[2];
    b[0] = static_cast<char>(v & 0xff);
    b[1] = static_cast<char>((v >> 8) & 0xff);
    out.write(b, 2);
  };

  w4("RIFF");
  u32(riff_chunk_size);
  w4("WAVE");
  w4("fmt ");
  u32(16);
  u16(1);
  u16(1);
  u32(sample_rate);
  u32(byte_rate);
  u16(block_align);
  u16(16);
  w4("data");
  u32(data_bytes);
  out.write(reinterpret_cast<const char*>(pcm.data()),
            static_cast<std::streamsize>(pcm.size() * sizeof(int16_t)));
}

std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(
    std::string_view lang_cli, const MoonshineTTSOptions& opt_in) {
  MoonshineTTSOptions opt = opt_in;
  if (opt.g2p_options.g2p_root.empty()) {
    opt.g2p_options.g2p_root = std::filesystem::current_path();
  }
  opt.apply_voice_engine_prefix();
  std::string eng =
      ascii_lowercase_copy(trim_ascii_ws_copy(opt.vocoder_engine));
  if (eng.empty()) {
    eng = "auto";
  }
  if (eng != "kokoro" && eng != "piper" && eng != "zipvoice" && eng != "auto") {
    return {};
  }
  if (eng == "zipvoice") {
    return zipvoice_vocoder_dependency_keys();
  }
  const std::string lk = normalize_lang_key(lang_cli);
  const bool kokoro_ok = kokoro_tts_lang_supported_inner(lk, opt.g2p_options);
  const bool use_kokoro = (eng == "kokoro") || (eng == "auto" && kokoro_ok);
  if (use_kokoro) {
    return kokoro_vocoder_dependency_keys_with_options(lk, opt);
  }
  return piper_vocoder_dependency_keys_with_options(lk, opt);
}

std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(
    std::string_view lang_cli) {
  return moonshine_catalog_tts_vocoder_only_dependency_keys(
      lang_cli, MoonshineTTSOptions{});
}

std::vector<std::string>
moonshine_catalog_all_tts_vocoder_dependency_keys_union() {
  const std::vector<std::string> tags =
      moonshine_asset_catalog_all_registered_language_tags();
  std::unordered_set<std::string> seen;
  std::vector<std::string> out;
  for (const std::string& tag : tags) {
    for (std::string p :
         moonshine_catalog_tts_vocoder_only_dependency_keys(tag)) {
      if (seen.insert(p).second) {
        out.push_back(std::move(p));
      }
    }
  }
  return out;
}

std::vector<MoonshineTtsVoiceAvailability>
moonshine_list_tts_voices_with_availability(std::string_view language_cli,
                                            const MoonshineTTSOptions& opt_in) {
  MoonshineTTSOptions opt = opt_in;
  if (opt.g2p_options.g2p_root.empty()) {
    opt.g2p_options.g2p_root = std::filesystem::current_path();
  }
  opt.apply_voice_engine_prefix();
  std::string eng =
      ascii_lowercase_copy(trim_ascii_ws_copy(opt.vocoder_engine));
  if (eng.empty()) {
    eng = "auto";
  }
  if (eng != "kokoro" && eng != "piper" && eng != "zipvoice" && eng != "auto") {
    return {};
  }
  if (eng == "zipvoice") {
    std::vector<MoonshineTtsVoiceAvailability> zv;
    for (const auto& pr : list_zipvoice_voices_with_availability(opt)) {
      zv.push_back(MoonshineTtsVoiceAvailability{
          std::string("zipvoice_") + pr.first, pr.second});
    }
    return zv;
  }
  const std::string lk = normalize_lang_key(language_cli);
  const bool kokoro_ok = kokoro_tts_lang_supported_inner(lk, opt.g2p_options);
  const bool use_kokoro = (eng == "kokoro") || (eng == "auto" && kokoro_ok);
  std::vector<MoonshineTtsVoiceAvailability> out;
  if (eng == "auto") {
    MoonshineTTSOptions opt_k = opt;
    opt_k.voice.clear();
    MoonshineTTSOptions opt_p = opt;
    opt_p.voice.clear();
    if (kokoro_ok) {
      for (const auto& pr : list_kokoro_voices_with_availability(lk, opt_k)) {
        out.push_back(MoonshineTtsVoiceAvailability{
            std::string("kokoro_") + pr.first, pr.second});
      }
    }
    for (const auto& pr : piper_list_voices_with_availability(
             make_piper_options(std::string(language_cli), opt_p))) {
      out.push_back(MoonshineTtsVoiceAvailability{
          std::string("piper_") + pr.first, pr.second});
    }
    // ZipVoice zero-shot cloning voices are English-only; surface them under
    // the auto catalog so a
    // ``zipvoice_*`` voice validates without the caller having to pin the
    // engine first.
    if (lk.rfind("en", 0) == 0) {
      MoonshineTTSOptions opt_z = opt;
      opt_z.voice.clear();
      for (const auto& pr : list_zipvoice_voices_with_availability(opt_z)) {
        out.push_back(MoonshineTtsVoiceAvailability{
            std::string("zipvoice_") + pr.first, pr.second});
      }
    }
    std::sort(
        out.begin(), out.end(),
        [](const MoonshineTtsVoiceAvailability& a,
           const MoonshineTtsVoiceAvailability& b) { return a.id < b.id; });
    return out;
  }
  if (use_kokoro) {
    for (const auto& pr : list_kokoro_voices_with_availability(lk, opt)) {
      out.push_back(MoonshineTtsVoiceAvailability{
          std::string("kokoro_") + pr.first, pr.second});
    }
    return out;
  }
  for (const auto& pr : piper_list_voices_with_availability(
           make_piper_options(std::string(language_cli), opt))) {
    out.push_back(MoonshineTtsVoiceAvailability{
        std::string("piper_") + pr.first, pr.second});
  }
  return out;
}

}  // namespace moonshine_tts
