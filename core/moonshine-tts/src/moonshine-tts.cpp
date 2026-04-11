#include "moonshine-tts.h"

#include "debug-utils.h"
#include "moonshine-asset-catalog.h"
#include "g2p-path.h"
#include "moonshine-g2p.h"
#include "ort-onnx-external-data.h"
#include "piper-tts.h"
#include "utf8-utils.h"

#include <onnxruntime_cxx_api.h>

#include <nlohmann/json.h>

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
#include <sstream>
#include <stdexcept>
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

void replace_utf8(std::string& s, std::string_view old_s, std::string_view new_s) {
  size_t pos = 0;
  while ((pos = s.find(old_s, pos)) != std::string::npos) {
    s.replace(pos, old_s.size(), new_s);
    pos += new_s.size();
  }
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
  return cat == UTF8PROC_CATEGORY_ZS || cat == UTF8PROC_CATEGORY_ZL || cat == UTF8PROC_CATEGORY_ZP;
}

std::string collapse_whitespace_join_single_space(const std::string& s) {
  std::u32string u = utf8_str_to_u32(s);
  std::string out;
  bool pending_space = false;
  for (char32_t cp : u) {
    const bool sp = (cp < 128 && std::isspace(static_cast<unsigned char>(cp)) != 0) ||
                    utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_ZS ||
                    utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_ZL ||
                    utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_ZP;
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
  /// MoonshineG2P dialect id; nullptr when resolved only via ``resolve_lang_for_tts`` Spanish fallback.
  const char* g2p_dialect = "en_us";
};

const LangProfile* lookup_lang_profile(std::string_view key) {
  // Keys use underscore form; ``normalize_lang_key`` maps client ``-`` / spaces to ``_`` before lookup.
  static const std::unordered_map<std::string, LangProfile> m{
      {"en_us", {'a', "af_heart", "en_us"}},
      {"en", {'a', "af_heart", "en_us"}},
      // UK Kokoro voice uses the same English rule + ONNX G2P assets as US (``en_us`` under g2p_root).
      {"en_gb", {'b', "bf_emma", "en_us"}},
      // Spanish G2P must be a concrete dialect id (same default as spanish_rule_g2p.text_to_ipa).
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

/// Fills *profile* and *g2p_dialect* for ``MoonshineG2P`` (Kokoro locale + rule-based tag).
void resolve_lang_for_tts(const std::string& lk, const MoonshineG2POptions& opt, LangProfile& profile,
                          std::string& g2p_dialect) {
  if (const LangProfile* p = lookup_lang_profile(lk)) {
    profile = *p;
    g2p_dialect = p->g2p_dialect;
    return;
  }
  const std::string norm = normalize_rule_based_dialect_cli_key(lk);
  if (!norm.empty() && dialect_resolves_to_spanish_rules(norm, opt.spanish_narrow_obstruents)) {
    profile = {'e', "ef_dora", nullptr};
    g2p_dialect = norm;
    return;
  }
  throw std::runtime_error("MoonshineTTS: unsupported --lang key \"" + lk + "\"");
}

bool kokoro_tts_lang_supported_inner(std::string_view lang_cli, const MoonshineG2POptions& opt) {
  const std::string lk = normalize_lang_key(lang_cli);
  if (lookup_lang_profile(lk) != nullptr) {
    return true;
  }
  const std::string norm = normalize_rule_based_dialect_cli_key(lk);
  return !norm.empty() && dialect_resolves_to_spanish_rules(norm, opt.spanish_narrow_obstruents);
}

bool voice_prefix_ok(char kokoro_lang, std::string_view voice) {
  static const std::unordered_map<char, std::vector<std::string_view>> pref{
      {'a', {"af_", "am_"}}, {'b', {"bf_", "bm_"}}, {'e', {"ef_", "em_"}}, {'f', {"ff_"}},
      {'h', {"hf_", "hm_"}}, {'i', {"if_", "im_"}}, {'p', {"pf_", "pm_"}},
      {'j', {"jf_", "jm_"}}, {'z', {"zf_", "zm_"}},
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

/// If ``--lang`` is US English but the user asked for a British Kokoro voice id (``bf_*`` / ``bm_*``), or the
/// reverse, switch the Kokoro profile so ``voice_prefix_ok`` and IPA normalization match the voice pack.
void maybe_align_en_profile_for_kokoro_voice(std::string_view voice, LangProfile& profile,
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

/// When the CLI language is not a Kokoro-backed locale (e.g. ``de`` for Piper-only), but the user
/// selected a Kokoro voice id, infer ``LangProfile`` / G2P dialect from the voice stem (``af_river`` → US English).
bool infer_lang_profile_from_kokoro_voice(std::string_view voice_sv, LangProfile& profile,
                                          std::string& g2p_dialect) {
  const std::string v = trim_ascii_ws_copy(voice_sv);
  if (v.empty()) {
    return false;
  }
  static constexpr const char* k_keys[] = {"en_us", "en_gb", "es", "fr", "hi", "it",
                                             "pt_br", "ja", "zh", "zh_hans"};
  for (const char* key : k_keys) {
    const LangProfile* p = lookup_lang_profile(key);
    if (p == nullptr) {
      continue;
    }
    if (voice_prefix_ok(p->kokoro_lang, v)) {
      profile = *p;
      g2p_dialect = p->g2p_dialect != nullptr ? std::string(p->g2p_dialect) : std::string();
      return true;
    }
  }
  return false;
}

/// Like ``resolve_lang_for_tts`` for Kokoro paths, but if *lk* is not Kokoro-capable (Piper-only language),
/// fall back to a profile derived from *voice_for_infer* when non-empty.
void resolve_lang_for_kokoro(const std::string& lk, const MoonshineG2POptions& g2p, LangProfile& profile,
                             std::string& g2p_dialect, std::string_view voice_for_infer) {
  try {
    resolve_lang_for_tts(lk, g2p, profile, g2p_dialect);
  } catch (const std::runtime_error&) {
    if (!infer_lang_profile_from_kokoro_voice(voice_for_infer, profile, g2p_dialect)) {
      throw;
    }
  }
}

bool kokoro_voice_asset_exists(const std::string& voice_id, const std::filesystem::path& voices_dir,
                               const FileInformationMap* tts_files,
                               const std::filesystem::path& g2p_root) {
  const auto voice_path = [&](const std::string& id) { return voices_dir / (id + ".kokorovoice"); };
  if (tts_files != nullptr) {
    const std::string vk = std::string("kokoro/voices/") + voice_id + ".kokorovoice";
    const auto it = tts_files->entries.find(vk);
    if (it != tts_files->entries.end()) {
      const FileInformation& fi = it->second;
      if (fi.memory != nullptr && fi.memory_size > 0) {
        return true;
      }
      if (!fi.path.empty()) {
        const std::filesystem::path p = resolve_path_under_root(g2p_root, fi.path);
        return std::filesystem::is_regular_file(p);
      }
    }
  }
  return std::filesystem::is_regular_file(voice_path(voice_id));
}

// Kokoro-82M voice ids (hexgrad/Kokoro-82M VOICES.md). Bundles may ship a subset; availability is per asset.
static const char* const kKokoroVoiceCatalog[] = {
    "af_alloy",   "af_aoede",   "af_bella",   "af_heart",   "af_jessica", "af_kore",    "af_nicole",
    "af_nova",    "af_river",   "af_sarah",   "af_sky",     "am_adam",    "am_echo",    "am_eric",
    "am_fenrir",  "am_liam",    "am_michael", "am_onyx",    "am_puck",    "am_santa",   "bf_alice",
    "bf_emma",    "bf_isabella", "bf_lily",   "bm_daniel",  "bm_fable",   "bm_george",  "bm_lewis",
    "ef_dora",    "em_alex",    "em_santa",   "ff_siwis",   "hf_alpha",   "hf_beta",    "hm_omega",
    "hm_psi",     "if_sara",    "im_nicola",  "jf_alpha",   "jf_gongitsune", "jf_nezumi", "jf_tebukuro",
    "jm_kumo",    "pf_dora",    "pm_alex",    "pm_santa",   "zf_xiaobei", "zf_xiaoni",  "zf_xiaoxiao",
    "zf_xiaoyi",  "zm_yunjian", "zm_yunxi",   "zm_yunxia",  "zm_yunyang",
};

std::string select_voice_id(char kokoro_lang, std::string_view requested, std::string_view default_voice,
                            const std::filesystem::path& voices_dir, const FileInformationMap* tts_files,
                            const std::filesystem::path& g2p_root) {
  std::string v = requested.empty() ? std::string(default_voice) : std::string(requested);
  if (!requested.empty() && kokoro_voice_asset_exists(v, voices_dir, tts_files, g2p_root) &&
      voice_prefix_ok(kokoro_lang, v)) {
    return v;
  }
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
      LOGF("  Available Kokoro voices for this language: %s", available.c_str());
    }
  };
  if (!voice_prefix_ok(kokoro_lang, v)) {
    LOGF("Requested Kokoro voice '%s' has wrong prefix for language '%c', falling back to '%s'",
         v.c_str(), kokoro_lang, std::string(default_voice).c_str());
    log_available_kokoro_voices();
    v = std::string(default_voice);
  }
  if (!kokoro_voice_asset_exists(v, voices_dir, tts_files, g2p_root)) {
    LOGF("Requested Kokoro voice '%s' not found, falling back to '%s'",
         v.c_str(), std::string(default_voice).c_str());
    log_available_kokoro_voices();
    v = std::string(default_voice);
  }
  return v;
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

// Mandarin Chinese IPA normalization for Kokoro: Chao tone letters → arrow contour symbols,
// consonant mappings to Kokoro's inventory (ꭧ for retroflex, ʦ for dental affricate, ʨ for palatal),
// tone repositioning before final nasals. Must run before the generic vocab-filter pass.
void apply_chinese_kokoro_normalization(std::string& ipa) {
  // ── Consonant mappings (longest first) ──
  // Retroflex affricates: ʈʂʰ → ꭧʰ, ʈʂ → ꭧ  (ꭧ = U+AB67, \xea\xad\xa7)
  replace_utf8(ipa, "\xca\x88\xca\x82\xca\xb0", "\xea\xad\xa7\xca\xb0");  // ʈʂʰ → ꭧʰ
  replace_utf8(ipa, "\xca\x88\xca\x82", "\xea\xad\xa7");                    // ʈʂ → ꭧ

  // Palatal affricates: tɕʰ → ʨʰ, tɕ → ʨ  (ʨ = U+02A8, \xca\xa8)
  replace_utf8(ipa, "t\xc9\x95\xca\xb0", "\xca\xa8\xca\xb0");  // tɕʰ → ʨʰ
  replace_utf8(ipa, "t\xc9\x95", "\xca\xa8");                    // tɕ → ʨ

  // Dental affricates: tsʰ → ʦʰ, ts → ʦ  (ʦ = U+02A6, \xca\xa6)
  replace_utf8(ipa, "ts\xca\xb0", "\xca\xa6\xca\xb0");  // tsʰ → ʦʰ
  replace_utf8(ipa, "ts", "\xca\xa6");                    // ts → ʦ

  // Tie-bar removal (if present): t͡ɕ, t͡s, d͡z
  replace_utf8(ipa, "t\xcd\xa1\xc9\x95", "\xca\xa8");  // t͡ɕ → ʨ
  replace_utf8(ipa, "t\xcd\xa1s", "\xca\xa6");          // t͡s → ʦ

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
  // → = U+2192 (\xe2\x86\x92), ↗ = U+2197 (\xe2\x86\x97), ↓ = U+2193 (\xe2\x86\x93), ↘ = U+2198 (\xe2\x86\x98)
  replace_utf8(ipa, "\xcb\xa8\xcb\xa9\xcb\xa6", "\xe2\x86\x93");  // ˨˩˦ (Tone 3) → ↓
  replace_utf8(ipa, "\xcb\xa5\xcb\xa5", "\xe2\x86\x92");           // ˥˥  (Tone 1) → →
  replace_utf8(ipa, "\xcb\xa7\xcb\xa5", "\xe2\x86\x97");           // ˧˥  (Tone 2) → ↗
  replace_utf8(ipa, "\xcb\xa5\xcb\xa9", "\xe2\x86\x98");           // ˥˩  (Tone 4) → ↘
  replace_utf8(ipa, "\xcb\xa9\xcb\xa9", "\xe2\x86\x93");           // ˩˩ → ↓
  replace_utf8(ipa, "\xcb\xa5\xcb\xa7", "\xe2\x86\x98");           // ˥˧ → ↘
  replace_utf8(ipa, "\xcb\xa7\xcb\xa9", "\xe2\x86\x93");           // ˧˩ → ↓
  replace_utf8(ipa, "\xcb\xa8\xcb\xa5", "\xe2\x86\x97");           // ˨˥ → ↗
  // Single remaining tone letters.
  replace_utf8(ipa, "\xcb\xa5", "\xe2\x86\x92");  // ˥ → →
  replace_utf8(ipa, "\xcb\xa6", "\xe2\x86\x92");  // ˦ → →
  replace_utf8(ipa, "\xcb\xa7", "");               // ˧ (neutral) → drop
  replace_utf8(ipa, "\xcb\xa8", "\xe2\x86\x93");  // ˨ → ↓
  replace_utf8(ipa, "\xcb\xa9", "\xe2\x86\x93");  // ˩ → ↓

  // ── Tone repositioning: move tone arrow before final nasals ──
  // Kokoro expects tones between the vowel and final nasal: pa→ŋ, pə↓n
  // Our G2P puts tones after the syllable: pɑŋ˥˥ → pɑŋ→ (after arrow conversion)
  // Need to swap: [nasal][arrow] → [arrow][nasal]
  static const std::string kArrows[] = {
      "\xe2\x86\x92", "\xe2\x86\x97", "\xe2\x86\x93", "\xe2\x86\x98"};
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

std::string normalize_ipa_to_kokoro(std::string ipa, char kokoro_lang,
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

std::vector<std::string> chunk_phonemes(const std::string& ps, int max_cp = 510) {
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

std::vector<int64_t> phoneme_str_to_input_ids(const std::string& phonemes,
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

void read_kokorovoice_bytes(const uint8_t* data, size_t size, std::string_view context_for_errors,
                            std::vector<float>& out_flat, uint32_t& rows, uint32_t& cols) {
  if (data == nullptr || size < 4 + 4 + 4) {
    throw std::runtime_error("MoonshineTTS: voice buffer too small (" + std::string(context_for_errors) + ")");
  }
  if (std::string_view(reinterpret_cast<const char*>(data), 4) != kVoiceMagic) {
    throw std::runtime_error("MoonshineTTS: bad magic (" + std::string(context_for_errors) + ") (expected KVO1)");
  }
  uint32_t r = 0;
  uint32_t c = 0;
  std::memcpy(&r, data + 4, 4);
  std::memcpy(&c, data + 8, 4);
  if (r == 0 || c == 0) {
    throw std::runtime_error("MoonshineTTS: invalid voice header (" + std::string(context_for_errors) + ")");
  }
  const size_t n = static_cast<size_t>(r) * static_cast<size_t>(c);
  const size_t need = 12 + n * sizeof(float);
  if (size < need) {
    throw std::runtime_error("MoonshineTTS: truncated voice data (" + std::string(context_for_errors) + ")");
  }
  out_flat.resize(n);
  std::memcpy(out_flat.data(), data + 12, n * sizeof(float));
  rows = r;
  cols = c;
}

void read_kokorovoice(const std::filesystem::path& path, std::vector<float>& out_flat, uint32_t& rows,
                      uint32_t& cols) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("MoonshineTTS: cannot open voice file " + path.string());
  }
  std::vector<uint8_t> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  read_kokorovoice_bytes(buf.data(), buf.size(), path.string(), out_flat, rows, cols);
}

Ort::SessionOptions make_ort_options(const std::vector<std::string>& names) {
  (void)names;
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts.SetIntraOpNumThreads(0);
  opts.SetInterOpNumThreads(0);
  return opts;
}

}  // namespace

bool kokoro_tts_lang_supported(std::string_view lang_cli, const MoonshineG2POptions& g2p_opt) {
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

std::filesystem::path tts_map_path(const FileInformationMap& m, std::string_view canonical_key) {
  const std::string k(canonical_key);
  const auto it = m.entries.find(k);
  if (it == m.entries.end()) {
    return std::filesystem::path(canonical_key);
  }
  return it->second.path;
}

PiperTTSOptions make_piper_options(std::string_view language, const MoonshineTTSOptions& opt) {
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
    const std::filesystem::path jr = opt.tts_relative_path(kTtsPiperOnnxJsonKey);
    if (!jr.empty()) {
      p.explicit_onnx_json_path = jr;
    }
  }
  const std::string pv_key(kTtsPiperVoicesKey);
  if (p.explicit_onnx_path.empty() && opt.files.entries.find(pv_key) != opt.files.entries.end()) {
    const std::filesystem::path vr = opt.tts_relative_path(kTtsPiperVoicesKey);
    if (!vr.empty()) {
      p.voices_dir = resolve_path_under_root(opt.g2p_options.g2p_root, vr);
    }
  }
  const std::string pvj_key(kTtsPiperVoicesJsonKey);
  if (p.explicit_onnx_path.empty() && opt.files.entries.find(pvj_key) != opt.files.entries.end()) {
    const std::filesystem::path vjr = opt.tts_relative_path(kTtsPiperVoicesJsonKey);
    if (!vjr.empty()) {
      p.voices_json_dir = resolve_path_under_root(opt.g2p_options.g2p_root, vjr);
    }
  }
  p.onnx_model = opt.voice;
  p.speed = opt.speed;
  p.g2p_options = opt.g2p_options;
  p.ort_provider_names = opt.ort_provider_names;
  p.piper_normalize_audio = opt.piper_normalize_audio;
  p.piper_output_volume = opt.piper_output_volume;
  p.piper_noise_scale_override = opt.piper_noise_scale_override;
  p.piper_noise_w_override = opt.piper_noise_w_override;
  p.tts_asset_files = opt.files;
  return p;
}

std::vector<std::string> kokoro_vocoder_dependency_keys_with_options(std::string_view language,
                                                                     const MoonshineTTSOptions& opt) {
  MoonshineG2POptions g2p = opt.g2p_options;
  if (g2p.g2p_root.empty()) {
    g2p.g2p_root = std::filesystem::current_path();
  }
  LangProfile profile{};
  std::string g2p_dialect;
  const std::string lk = normalize_lang_key(language);
  resolve_lang_for_kokoro(lk, g2p, profile, g2p_dialect, opt.voice);
  maybe_align_en_profile_for_kokoro_voice(opt.voice, profile, g2p_dialect);
  std::filesystem::path model_path =
      resolve_path_under_root(g2p.g2p_root, tts_map_path(opt.files, kTtsKokoroModelOnnxKey));
  resolve_disk_model_file_path(model_path);
  const std::filesystem::path voices_dir = model_path.parent_path() / "voices";
  // Dependency keys must name the *requested* voice even when the .kokorovoice file is not on disk yet
  // (select_voice_id falls back to the default when missing, which would break download prefetch).
  const std::string req = trim_ascii_ws_copy(opt.voice);
  std::string vid;
  if (!req.empty() && voice_prefix_ok(profile.kokoro_lang, req)) {
    vid = req;
  } else {
    vid = select_voice_id(profile.kokoro_lang, opt.voice, profile.default_voice, voices_dir, &opt.files,
                          g2p.g2p_root);
  }
  return {std::string(kTtsKokoroModelOnnxKey), std::string(kTtsKokoroConfigJsonKey),
          std::string("kokoro/voices/") + vid + ".kokorovoice"};
}

std::vector<std::pair<std::string, bool>> list_kokoro_voices_with_availability(const std::string& lk,
                                                                               const MoonshineTTSOptions& opt) {
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
  std::filesystem::path model_path =
      resolve_path_under_root(g2p.g2p_root, tts_map_path(opt_scan.files, kTtsKokoroModelOnnxKey));
  resolve_disk_model_file_path(model_path);
  const std::filesystem::path voices_dir = model_path.parent_path() / "voices";

  std::map<std::string, bool> by_id;
  for (const char* vid : kKokoroVoiceCatalog) {
    const std::string id(vid);
    if (!voice_prefix_ok(profile.kokoro_lang, id)) {
      continue;
    }
    by_id[id] = kokoro_voice_asset_exists(id, voices_dir, &opt_scan.files, g2p.g2p_root);
  }

  auto consider_extra = [&](const std::string& id) {
    if (!voice_prefix_ok(profile.kokoro_lang, id)) {
      return;
    }
    if (by_id.find(id) != by_id.end()) {
      return;
    }
    by_id[id] = kokoro_voice_asset_exists(id, voices_dir, &opt_scan.files, g2p.g2p_root);
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
    if (key.compare(key.size() - k_suffix.size(), k_suffix.size(), k_suffix) != 0) {
      continue;
    }
    consider_extra(key.substr(k_prefix.size(), key.size() - k_prefix.size() - k_suffix.size()));
  }

  std::vector<std::pair<std::string, bool>> out;
  out.reserve(by_id.size());
  for (const auto& pr : by_id) {
    out.emplace_back(pr.first, pr.second);
  }
  return out;
}

std::vector<std::string> piper_vocoder_dependency_keys_with_options(std::string_view language,
                                                                    const MoonshineTTSOptions& opt) {
  const std::string onnx_key(kTtsPiperOnnxKey);
  const std::string json_key(kTtsPiperOnnxJsonKey);
  if (opt.files.entries.find(onnx_key) != opt.files.entries.end()) {
    return {onnx_key, json_key};
  }
  MoonshineG2POptions g2p = opt.g2p_options;
  if (g2p.g2p_root.empty()) {
    g2p.g2p_root = std::filesystem::current_path();
  }
  std::string o;
  std::string j;
  if (piper_default_model_bundle_relative_paths(language, g2p, &o, &j, opt.voice)) {
    return {std::move(o), std::move(j)};
  }
  return {};
}

struct KokoroTtsEngine {
  std::filesystem::path model_path_;
  std::filesystem::path config_path_;
  std::filesystem::path voices_dir_;
  FileInformationMap tts_files_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "moonshine_tts"};
  Ort::Session session_{nullptr};
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};

  std::unordered_map<std::string, int> vocab_{};
  std::unordered_set<std::string> vocab_keys_{};
  std::vector<float> voice_{};
  uint32_t voice_rows_ = 0;
  uint32_t voice_cols_ = 0;

  std::string voice_id_{};
  double speed_ = 1.0;
  /// ``speed`` ONNX input element type from the loaded graph (FP32 community ONNX vs double local export).
  ONNXTensorElementDataType speed_elem_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  char kokoro_lang_ = 'a';
  LangProfile profile_{};
  std::string g2p_dialect_{};
  MoonshineG2POptions g2p_opt_{};
  std::unique_ptr<MoonshineG2P> g2p_{};
  /// Hugging Face ``onnx-community/Kokoro-82M-v1.0-ONNX`` quantized graph names the style vector ``style``;
  /// local torch exports use ``ref_s``.
  std::string style_input_name_ = "ref_s";

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
    // Kokoro ONNX convention: inputs [0]=input_ids, [1]=ref_s|style, [2]=speed. Community HF models use
    // float32 speed [1]; local torch exports use double scalar.
    const size_t n_in = session_.GetInputCount();
    if (n_in < 3) {
      return;
    }
    Ort::TypeInfo ti = session_.GetInputTypeInfo(2);
    if (ti.GetONNXType() != ONNX_TYPE_TENSOR) {
      return;
    }
    const auto tinfo = ti.GetTensorTypeAndShapeInfo();
    speed_elem_type_ = static_cast<ONNXTensorElementDataType>(tinfo.GetElementType());
  }

  explicit KokoroTtsEngine(std::string_view language, MoonshineTTSOptions opt) {
    if (!(opt.speed > 0.0) || !std::isfinite(opt.speed)) {
      throw std::runtime_error("MoonshineTTS: speed must be a positive finite number");
    }
    speed_ = opt.speed;
    g2p_opt_ = std::move(opt.g2p_options);
    tts_files_ = std::move(opt.files);
    const std::filesystem::path& root = g2p_opt_.g2p_root;
    model_path_ = resolve_path_under_root(root, tts_map_path(tts_files_, kTtsKokoroModelOnnxKey));
    resolve_disk_model_file_path(model_path_);
    config_path_ = resolve_path_under_root(root, tts_map_path(tts_files_, kTtsKokoroConfigJsonKey));
    voices_dir_ = model_path_.parent_path() / "voices";

    const auto mit = tts_files_.entries.find(std::string(kTtsKokoroModelOnnxKey));
    const auto cit = tts_files_.entries.find(std::string(kTtsKokoroConfigJsonKey));
    if (mit == tts_files_.entries.end() || cit == tts_files_.entries.end()) {
      throw std::runtime_error("MoonshineTTS: missing Kokoro file map entries (model/config keys)");
    }
    FileInformation& model_fi = mit->second;
    FileInformation& cfg_fi = cit->second;
    // FileInformation::load opens `path` as-is; resolve against g2p_root here so defaults like
    // ``kokoro/config.json`` work when the process cwd is not the bundle (model_path_/config_path_
    // already use resolve_path_under_root).
    model_fi.path = model_path_;
    cfg_fi.path = config_path_;

    const uint8_t* cfg_buf = nullptr;
    size_t cfg_len = 0;
    cfg_fi.load(&cfg_buf, &cfg_len);
    if (cfg_len == 0) {
      cfg_fi.free();
      throw std::runtime_error("MoonshineTTS: empty Kokoro config (" + config_path_.string() + ")");
    }
    {
      const std::string cfg_str(reinterpret_cast<const char*>(cfg_buf), cfg_len);
      nlohmann::json j = nlohmann::json::parse(cfg_str);
      if (!j.contains("vocab") || !j["vocab"].is_object()) {
        cfg_fi.free();
        throw std::runtime_error("MoonshineTTS: config.json missing vocab object");
      }
      for (auto it = j["vocab"].begin(); it != j["vocab"].end(); ++it) {
        const std::string key = it.key();
        vocab_[key] = it.value().get<int>();
        vocab_keys_.insert(key);
      }
    }
    cfg_fi.free();

    const uint8_t* onnx_buf = nullptr;
    size_t onnx_len = 0;
    model_fi.load(&onnx_buf, &onnx_len);
    if (onnx_len == 0) {
      model_fi.free();
      throw std::runtime_error("MoonshineTTS: empty Kokoro ONNX (" + model_path_.string() + ")");
    }
    Ort::SessionOptions session_opts = make_ort_options(opt.ort_provider_names);
    ort_add_external_initializer_files_for_onnx_model_buffer(session_opts, tts_files_,
                                                               kTtsKokoroModelOnnxKey);
    session_ = Ort::Session(env_, onnx_buf, onnx_len, session_opts);
    model_fi.free();

    detect_kokoro_style_input_name();
    detect_speed_input_element_type();
    const std::string lk = normalize_lang_key(language);
    resolve_lang_for_kokoro(lk, g2p_opt_, profile_, g2p_dialect_, opt.voice);
    maybe_align_en_profile_for_kokoro_voice(opt.voice, profile_, g2p_dialect_);
    kokoro_lang_ = profile_.kokoro_lang;
    g2p_ = std::make_unique<MoonshineG2P>(g2p_dialect_, g2p_opt_);
    voice_id_ = select_voice_id(kokoro_lang_, opt.voice, profile_.default_voice, voices_dir_, &tts_files_,
                                g2p_opt_.g2p_root);
    reload_voice_tensor();
  }

  void reload_voice_tensor() {
    const std::string vk = std::string("kokoro/voices/") + voice_id_ + ".kokorovoice";
    const auto vit = tts_files_.entries.find(vk);
    if (vit != tts_files_.entries.end()) {
      FileInformation& vf = vit->second;
      if (vf.memory == nullptr || vf.memory_size == 0) {
        vf.path = resolve_path_under_root(g2p_opt_.g2p_root, tts_map_path(tts_files_, vk));
      }
      const uint8_t* vb = nullptr;
      size_t vz = 0;
      vf.load(&vb, &vz);
      read_kokorovoice_bytes(vb, vz, vk, voice_, voice_rows_, voice_cols_);
      vf.free();
      return;
    }
    const auto path =
        resolve_path_under_root(g2p_opt_.g2p_root, tts_map_path(tts_files_, vk));
    if (!std::filesystem::is_regular_file(path)) {
      const auto pt = voices_dir_ / (voice_id_ + ".pt");
      std::ostringstream msg;
      msg << "MoonshineTTS: missing voice file " << path.string();
      if (std::filesystem::is_regular_file(pt)) {
        msg << "\n  Export from PyTorch voice pack:\n  python scripts/export_kokoro_voice_for_cpp.py \""
            << pt.string() << "\" \"" << path.string() << '"';
      } else {
        msg << "\n  Install voices under " << voices_dir_.string()
            << " (e.g. python scripts/download_kokoro_onnx.py --out-dir "
            << model_path_.parent_path().string() << " --voices " << voice_id_
            << "), then export:\n  python scripts/export_kokoro_voice_for_cpp.py \""
            << (voices_dir_ / (voice_id_ + ".pt")).string() << "\" \"" << path.string() << '"';
      }
      throw std::runtime_error(msg.str());
    }
    read_kokorovoice(path, voice_, voice_rows_, voice_cols_);
  }

  std::vector<float> synthesize(std::string_view text) {
    const std::string ipa = g2p_->text_to_ipa(text, nullptr);
    if (trim_ascii_ws_copy(ipa).empty()) {
      return {};
    }
    std::string phonemes = normalize_ipa_to_kokoro(ipa, kokoro_lang_, vocab_keys_);
    if (phonemes.empty()) {
      return {};
    }
    const std::vector<std::string> chunks = chunk_phonemes(phonemes);
    if (chunks.empty()) {
      return {};
    }
    std::vector<float> wave_all;
    wave_all.reserve(chunks.size() * 8192);

    const char* in_names[3] = {"input_ids", style_input_name_.c_str(), "speed"};
    static const char* out_names[] = {"waveform"};

    for (const std::string& piece : chunks) {
      if (trim_ascii_ws_copy(piece).empty()) {
        continue;
      }
      std::vector<int64_t> ids = phoneme_str_to_input_ids(piece, vocab_);
      if (ids.size() > 512) {
        throw std::runtime_error("MoonshineTTS: phoneme token sequence too long for Kokoro (>512)");
      }
      const int64_t ntok = static_cast<int64_t>(ids.size());
      const std::array<int64_t, 2> shape_ids{1, ntok};

      const std::u32string pu = utf8_str_to_u32(piece);
      const size_t ncp = std::max<size_t>(pu.size(), 1);
      const size_t idx =
          std::min(ncp - 1, static_cast<size_t>(voice_rows_ > 0 ? voice_rows_ - 1 : 0));
      const size_t off = idx * static_cast<size_t>(voice_cols_);
      std::vector<float> ref_row(voice_cols_);
      if (off + voice_cols_ > voice_.size()) {
        throw std::runtime_error("MoonshineTTS: voice tensor index out of range");
      }
      std::copy(voice_.begin() + static_cast<std::ptrdiff_t>(off),
                voice_.begin() + static_cast<std::ptrdiff_t>(off + voice_cols_), ref_row.begin());
      const std::array<int64_t, 2> shape_ref{1, static_cast<int64_t>(voice_cols_)};

      std::vector<Ort::Value> inputs;
      inputs.push_back(Ort::Value::CreateTensor<int64_t>(
          mem_, ids.data(), ids.size(), shape_ids.data(), shape_ids.size()));
      inputs.push_back(Ort::Value::CreateTensor<float>(
          mem_, ref_row.data(), ref_row.size(), shape_ref.data(), shape_ref.size()));
      if (speed_elem_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        float speed_f = static_cast<float>(speed_);
        const std::array<int64_t, 1> shape_speed{1};
        inputs.push_back(
            Ort::Value::CreateTensor<float>(mem_, &speed_f, 1, shape_speed.data(), 1));
      } else {
        double speed_val = speed_;
        inputs.push_back(
            Ort::Value::CreateTensor<double>(mem_, &speed_val, 1, nullptr, 0));
      }

      Ort::RunOptions run_opts{nullptr};
      auto outputs = session_.Run(run_opts, in_names, inputs.data(), inputs.size(), out_names, 1);
      const Ort::Value& wav = outputs[0];
      const auto ti = wav.GetTensorTypeAndShapeInfo();
      const size_t n_el = ti.GetElementCount();
      if (ti.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("MoonshineTTS: ONNX output is not float32");
      }
      const float* wptr = wav.GetTensorData<float>();
      for (size_t i = 0; i < n_el; ++i) {
        wave_all.push_back(wptr[i]);
      }
    }
    return wave_all;
  }
};

struct MoonshineTTS::Impl {
  std::unique_ptr<KokoroTtsEngine> kokoro_;
  std::unique_ptr<PiperTTS> piper_;

  explicit Impl(std::string_view language, const MoonshineTTSOptions& opt_in) {
    MoonshineTTSOptions opt = opt_in;
    for (const moonshine_tts::FileInformation& fi : opt.file_information) {
      const std::string map_key = fi.path.generic_string();
      if (map_key.empty()) {
        continue;
      }
      const bool is_tts_only =
          (map_key.size() >= 7 && map_key.compare(0, 7, "kokoro/") == 0) ||
          (map_key.size() >= 6 && map_key.compare(0, 6, "piper/") == 0);
      if (is_tts_only) {
        opt.files.entries[map_key] = fi;
      } else {
        opt.g2p_options.files.entries[map_key] = fi;
      }
    }
    if (opt.g2p_options.g2p_root.empty()) {
      opt.g2p_options.g2p_root = std::filesystem::current_path();
    }
    opt.apply_voice_engine_prefix();
    std::string eng = ascii_lowercase_copy(trim_ascii_ws_copy(opt.vocoder_engine));
    if (eng.empty()) {
      eng = "auto";
    }
    if (eng != "kokoro" && eng != "piper" && eng != "auto") {
      throw std::runtime_error(
          "MoonshineTTS: vocoder_engine must be kokoro, piper, or auto (got \"" + eng + "\")");
    }
    const bool kokoro_ok = kokoro_tts_lang_supported_inner(language, opt.g2p_options);
    const bool use_kokoro = (eng == "kokoro") || (eng == "auto" && kokoro_ok);
    if (use_kokoro) {
      kokoro_ = std::make_unique<KokoroTtsEngine>(language, std::move(opt));
    } else {
      piper_ = std::make_unique<PiperTTS>(make_piper_options(language, opt));
    }
  }

  std::vector<float> synthesize(std::string_view text) {
    if (kokoro_) {
      return kokoro_->synthesize(text);
    }
    return piper_->synthesize(text);
  }
};

MoonshineTTS::MoonshineTTS(std::string_view language, const MoonshineTTSOptions& opt)
    : impl_(std::make_unique<Impl>(language, opt)) {}

MoonshineTTS::~MoonshineTTS() = default;

MoonshineTTS::MoonshineTTS(MoonshineTTS&&) noexcept = default;
MoonshineTTS& MoonshineTTS::operator=(MoonshineTTS&&) noexcept = default;

std::vector<float> MoonshineTTS::synthesize(std::string_view text) { return impl_->synthesize(text); }

void write_wav_mono_pcm16(const std::filesystem::path& path, const std::vector<float>& samples) {
  // parent_path() is empty for plain filenames like "out.wav"; create_directories("") throws on some libstdc++.
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
  const uint32_t sample_rate = static_cast<uint32_t>(MoonshineTTS::kSampleRateHz);
  const uint32_t num_samples = static_cast<uint32_t>(pcm.size());
  const uint32_t byte_rate = sample_rate * 2;
  const uint16_t block_align = 2;
  const uint32_t data_bytes = num_samples * 2;
  const uint32_t riff_chunk_size = 36 + data_bytes;

  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("write_wav_mono_pcm16: cannot open " + path.string());
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
  std::string eng = ascii_lowercase_copy(trim_ascii_ws_copy(opt.vocoder_engine));
  if (eng.empty()) {
    eng = "auto";
  }
  if (eng != "kokoro" && eng != "piper" && eng != "auto") {
    return {};
  }
  const std::string lk = normalize_lang_key(lang_cli);
  const bool kokoro_ok = kokoro_tts_lang_supported_inner(lk, opt.g2p_options);
  const bool use_kokoro = (eng == "kokoro") || (eng == "auto" && kokoro_ok);
  if (use_kokoro) {
    return kokoro_vocoder_dependency_keys_with_options(lk, opt);
  }
  return piper_vocoder_dependency_keys_with_options(lk, opt);
}

std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(std::string_view lang_cli) {
  return moonshine_catalog_tts_vocoder_only_dependency_keys(lang_cli, MoonshineTTSOptions{});
}

std::vector<std::string> moonshine_catalog_all_tts_vocoder_dependency_keys_union() {
  const std::vector<std::string> tags = moonshine_asset_catalog_all_registered_language_tags();
  std::unordered_set<std::string> seen;
  std::vector<std::string> out;
  for (const std::string& tag : tags) {
    for (std::string p : moonshine_catalog_tts_vocoder_only_dependency_keys(tag)) {
      if (seen.insert(p).second) {
        out.push_back(std::move(p));
      }
    }
  }
  return out;
}

std::vector<MoonshineTtsVoiceAvailability> moonshine_list_tts_voices_with_availability(
    std::string_view language_cli, const MoonshineTTSOptions& opt_in) {
  MoonshineTTSOptions opt = opt_in;
  if (opt.g2p_options.g2p_root.empty()) {
    opt.g2p_options.g2p_root = std::filesystem::current_path();
  }
  opt.apply_voice_engine_prefix();
  std::string eng = ascii_lowercase_copy(trim_ascii_ws_copy(opt.vocoder_engine));
  if (eng.empty()) {
    eng = "auto";
  }
  if (eng != "kokoro" && eng != "piper" && eng != "auto") {
    return {};
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
        out.push_back(MoonshineTtsVoiceAvailability{std::string("kokoro_") + pr.first, pr.second});
      }
    }
    for (const auto& pr :
         piper_list_voices_with_availability(make_piper_options(std::string(language_cli), opt_p))) {
      out.push_back(MoonshineTtsVoiceAvailability{std::string("piper_") + pr.first, pr.second});
    }
    std::sort(out.begin(), out.end(),
              [](const MoonshineTtsVoiceAvailability& a, const MoonshineTtsVoiceAvailability& b) {
                return a.id < b.id;
              });
    return out;
  }
  if (use_kokoro) {
    for (const auto& pr : list_kokoro_voices_with_availability(lk, opt)) {
      out.push_back(MoonshineTtsVoiceAvailability{std::string("kokoro_") + pr.first, pr.second});
    }
    return out;
  }
  for (const auto& pr : piper_list_voices_with_availability(make_piper_options(std::string(language_cli), opt))) {
    out.push_back(MoonshineTtsVoiceAvailability{std::string("piper_") + pr.first, pr.second});
  }
  return out;
}

}  // namespace moonshine_tts
