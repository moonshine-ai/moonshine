#include "ipa-postprocess.h"

#include "utf8-utils.h"

extern "C" {
#include <utf8proc.h>
}

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

namespace {

void replace_utf8_all(std::string& s, std::string_view old_utf8, std::string_view new_utf8) {
  size_t pos = 0;
  while ((pos = s.find(old_utf8, pos)) != std::string::npos) {
    s.replace(pos, old_utf8.size(), new_utf8);
    pos += new_utf8.size();
  }
}

std::string trim_copy(std::string t) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  t.erase(t.begin(), std::find_if(t.begin(), t.end(), not_space));
  t.erase(std::find_if(t.rbegin(), t.rend(), not_space).base(), t.end());
  return t;
}

std::string strip_length_markers_copy(std::string t) {
  const std::string mark = "\xCB\x90";  // UTF-8 U+02D0
  size_t p = 0;
  while ((p = t.find(mark, p)) != std::string::npos) {
    t.erase(p, mark.size());
  }
  return t;
}

void apply_shared_g2p_to_piper_replacements(std::string& s) {
  static const char kG2pRhoticStressed[] = "\xC9\x9D";      // U+025D ɝ
  static const char kEspeakRhoticStressed[] = "\xC9\x9C"   // U+025C ɜ
                                              "\xCB\x90";  // U+02D0 ː
  replace_utf8_all(s, kG2pRhoticStressed, kEspeakRhoticStressed);
}

void apply_korean_post_normalize_ipa(std::string& s) {
  // eSpeak-style primary stress on word-initial ``jʌ`` (e.g. 여보세요). UTF-8: j=0x6A, ʌ=U+028C CA 8C, ˈ=U+02C8 CB 88.
  static const char kJOpen[] = "j\xca\x8c";
  static const char kJStressOpen[] = "j\xcb\x88\xca\x8c";
  static const char kSpJOpen[] = " j\xca\x8c";
  static const char kSpJStressOpen[] = " j\xcb\x88\xca\x8c";
  if (s.size() >= sizeof(kJOpen) - 1 && s.compare(0, sizeof(kJOpen) - 1, kJOpen) == 0) {
    s.replace(0, sizeof(kJOpen) - 1, kJStressOpen);
  }
  size_t pos = 0;
  const size_t old_len = sizeof(kSpJOpen) - 1;
  const size_t new_len = sizeof(kSpJStressOpen) - 1;
  while ((pos = s.find(kSpJOpen, pos)) != std::string::npos) {
    s.replace(pos, old_len, kSpJStressOpen);
    pos += new_len;
  }
}

void apply_lang_specific_replacements(std::string& s, std::string_view piper_lang_key) {
  // Mirror ``piper_ipa_normalization.LANG_SPECIFIC_G2P_TO_PIPER_REPLACEMENTS`` (ordered; longest first per language).
  // Adjacent string literals: avoid ``\x..`` swallowing following hex letters (e.g. ``\xb0a``).
  static const std::vector<std::pair<std::string, std::string>> kKoG2pToEspeakLike = {
      {std::string("kams") + "\xca\xb0" + "ahamnida",
       std::string("\xc9\xa1\xcb\x88\xc9\x90ms\xc9\x90h\xcb\x8c\xc9\x90pnid\xcb\x8c\xc9\x90")},
      {std::string("has") + "\xca\xb0" + "ejo", std::string("h\xcb\x8c\xc9\x90sej\xcb\x8c") + "o"},
      {std::string("t\xc9\x9bhanminkuk") + "\xcc\x9a", std::string("d\xc9\x9bh\xcb\x88\xc9\x90nminq\xcb\x8c") + "uq"},
      {std::string("an\xc9\xb2j\xca\x8c\xc5\x8b"), std::string("\xcb\x88\xc9\x90nnj\xca\x8c\xc5\x8b")},
      {std::string("s") + "\xca\xb0" + "ejo", std::string("s\xcb\x8c") + "ejo"},
      {std::string("s") + "\xca\xb0" + "e", std::string("s\xcb\x8c") + "e"},
      {std::string("s") + "\xca\xb0", "s"},
  };
  static const std::unordered_map<std::string, std::vector<std::pair<std::string, std::string>>> kLangReplacements{};
  std::string_view eff = piper_lang_key;
  if (eff == "ko_kr" || eff == "korean") {
    eff = "ko";
  }
  if (eff == "ko") {
    for (const auto& pr : kKoG2pToEspeakLike) {
      replace_utf8_all(s, pr.first, pr.second);
    }
    apply_korean_post_normalize_ipa(s);
    return;
  }
  const std::string key(eff);
  const auto it = kLangReplacements.find(key);
  if (it == kLangReplacements.end()) {
    return;
  }
  for (const auto& pr : it->second) {
    replace_utf8_all(s, pr.first, pr.second);
  }
}

bool py_isspace_one_utf8_char(std::string_view ch) {
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
  const auto cat = static_cast<utf8proc_category_t>(utf8proc_category(static_cast<utf8proc_int32_t>(cp)));
  return cat == UTF8PROC_CATEGORY_ZS || cat == UTF8PROC_CATEGORY_ZL || cat == UTF8PROC_CATEGORY_ZP;
}

bool unicode_category_first_char_is_p_or_s(char32_t cp) {
  const char* cs = utf8proc_category_string(static_cast<utf8proc_int32_t>(cp));
  return cs != nullptr && (cs[0] == 'P' || cs[0] == 'S');
}

bool category_is_mn_or_me(char32_t cp) {
  const auto cat = static_cast<utf8proc_category_t>(utf8proc_category(static_cast<utf8proc_int32_t>(cp)));
  return cat == UTF8PROC_CATEGORY_MN || cat == UTF8PROC_CATEGORY_ME;
}

bool is_ipa_like_inventory_char(char32_t cp) {
  const auto cat = static_cast<utf8proc_category_t>(utf8proc_category(static_cast<utf8proc_int32_t>(cp)));
  switch (cat) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
      return true;
    default:
      break;
  }
  if (cp >= 0x250 && cp <= 0x2FF) {
    return true;
  }
  if (cp >= 0x300 && cp <= 0x36F) {
    return true;
  }
  if (cp >= 0x1D00 && cp <= 0x1D7F) {
    return true;
  }
  return false;
}

char32_t utf8_singleton_codepoint(std::string_view token) {
  std::string tmp(token);
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(tmp, 0, cp, adv) || adv != tmp.size()) {
    return 0;
  }
  return cp;
}

size_t utf8_prev_codepoint_start(const std::string& s, size_t char_start) {
  if (char_start == 0) {
    return 0;
  }
  size_t k = char_start;
  do {
    --k;
  } while (k > 0 && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80);
  return k;
}

// Map combining acute (U+0301) after a nucleus onto U+02C8 ˈ (Piper-style modifier stress). If the
// nucleus already has ˈ/ˌ immediately before it, drop the redundant acute only.
void rewrite_russian_combining_acute_to_primary_stress(std::string& s) {
  static const std::string kAcute("\xcc\x81");
  static const std::string kPri("\xcb\x88");
  static const std::string kSec("\xcb\x8c");  // U+02CC ˌ (secondary stress; not ʌ U+028C)
  size_t p = 0;
  while ((p = s.find(kAcute, p)) != std::string::npos) {
    if (p == 0) {
      s.erase(p, kAcute.size());
      continue;
    }
    size_t anchor = utf8_prev_codepoint_start(s, p);
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(s, anchor, cp, adv)) {
      s.erase(p, kAcute.size());
      continue;
    }
    if (cp == U'\u02B2') {
      const size_t prev = utf8_prev_codepoint_start(s, anchor);
      if (prev < anchor) {
        anchor = prev;
      }
    }
    if (anchor >= 2 && s.compare(anchor - 2, 2, kPri) == 0) {
      s.erase(p, kAcute.size());
      p = anchor;
      continue;
    }
    if (anchor >= 2 && s.compare(anchor - 2, 2, kSec) == 0) {
      s.erase(p, kAcute.size());
      p = anchor;
      continue;
    }
    const std::string nucleus = s.substr(anchor, p - anchor);
    const std::string replacement = kPri + nucleus;
    s.replace(anchor, (p + kAcute.size()) - anchor, replacement);
    p = anchor + replacement.size();
  }
}

// Russian G2P uses narrow-IPA tie bars and retroflex letters; Piper / Kokoro / espeak-ng expect
// digraph affricates, /ʃ/ /ʒ/, ASCII a/i/u/y for many vowel slots, retroflex ɭ (not velarized ɫ).
// NFC first; combining acute → ˈ before vowel remaps; stress+nucleus pairs (ˈɨ→ˈy, …) before bare
// ɨ→y so ˈ stays aligned. ɪ→i runs before the boundary pass that maps ASCII `` i `` back to ɪ.
void apply_russian_ipa_piper_style(std::string& s) {
  {
    std::string nfc = utf8_nfc_copy(s);
    s.swap(nfc);
  }
  static const std::vector<std::pair<std::string, std::string>> kAffricateSchwa = {
      {std::string("t") + "\xc9\x95" + "t" + "\xcb\x88" + "o", std::string("\xca\x83") + "to"},
      {std::string("t") + "\xc9\x95" + "t" + "\xcb\x88" + "\xc9\x94", std::string("\xca\x83") + "to"},
      {std::string("t") + "\xc9\x95" + "to", std::string("\xca\x83") + "to"},
      {std::string("t") + "\xc9\x95" + "t" + "\xca\x8c", std::string("\xca\x83") + "to"},
      {std::string("t") + "\xcd\xa1" + "\xc9\x95", std::string("t") + "\xca\x83" + "\xca\xb2"},
      {std::string("d") + "\xcd\xa1" + "\xca\x90", std::string("d") + "\xca\x90"},
      {std::string("t") + "\xcd\xa1" + "s", std::string("ts")},
      {std::string("d") + "\xcd\xa1" + "z", std::string("dz")},
      {std::string("t") + "\xc9\x95", std::string("t") + "\xca\x83" + "\xca\xb2"},
      {"\xca\x82", "\xca\x83"},
      {"\xc9\x90", "\xca\x8c"},
      {"\xc9\x99", "\xca\x8c"},
  };
  for (const auto& pr : kAffricateSchwa) {
    replace_utf8_all(s, pr.first, pr.second);
  }
  rewrite_russian_combining_acute_to_primary_stress(s);
  static const std::vector<std::pair<std::string, std::string>> kStressNucleus = {
      {"\xcb\x88\xc9\xa8", std::string("\xcb\x88") + "y"},        // ˈɨ → ˈy
      {"\xcb\x8c\xc9\xa8", std::string("\xcb\x8c") + "y"},        // ˌɨ → ˌy
      {"\xcb\x88\xc9\xab", std::string("\xcb\x88\xc9\xad")},       // ˈɫ → ˈɭ
      {"\xcb\x8c\xc9\xab", std::string("\xcb\x8c\xc9\xad")},       // ˌɫ → ˌɭ
      {"\xcb\x88\xca\x8c", std::string("\xcb\x88") + "a"},         // ˈʌ → ˈa
      {"\xcb\x8c\xca\x8c", std::string("\xcb\x8c") + "a"},         // ˌʌ → ˌa
      {"\xcb\x88\xc9\xaa", std::string("\xcb\x88") + "i"},        // ˈɪ → ˈi
      {"\xcb\x8c\xc9\xaa", std::string("\xcb\x8c") + "i"},        // ˌɪ → ˌi
      {"\xcb\x88\xca\x8a", std::string("\xcb\x88") + "u"},         // ˈʊ → ˈu
      {"\xcb\x8c\xca\x8a", std::string("\xcb\x8c") + "u"},         // ˌʊ → ˌu
      {"\xcb\x88\xca\x89", std::string("\xcb\x88") + "u"},         // ˈʉ → ˈu
      {"\xcb\x8c\xca\x89", std::string("\xcb\x8c") + "u"},         // ˌʉ → ˌu
  };
  for (const auto& pr : kStressNucleus) {
    replace_utf8_all(s, pr.first, pr.second);
  }
  static const std::vector<std::pair<std::string, std::string>> kBareNucleus = {
      {"\xc9\xab", "\xc9\xad"},  // ɫ → ɭ
      {"\xc9\xa8", "y"},         // ɨ → y
      {"\xca\x89", "u"},         // ʉ → u (U+0289)
      {"\xca\x8c", "a"},         // ʌ → a (U+028C)
      {"\xc9\xaa", "i"},         // ɪ → i (U+026A)
      {"\xca\x8a", "u"},         // ʊ → u (U+028A)
  };
  for (const auto& pr : kBareNucleus) {
    replace_utf8_all(s, pr.first, pr.second);
  }
  static const std::string kZhd("\xca\x90");
  static const std::string kZhj("\xca\x92");
  size_t p = 0;
  while ((p = s.find(kZhd, p)) != std::string::npos) {
    if (p >= 1 && s[p - 1] == 'd') {
      p += kZhd.size();
      continue;
    }
    s.replace(p, kZhd.size(), kZhj);
    p += kZhj.size();
  }
  replace_utf8_all(s, " i ", " \xc9\xaa ");
  replace_utf8_all(s, ")i ", ")\xc9\xaa ");
  replace_utf8_all(s, "\xc2\xab" "i ", "\xc2\xab" "\xc9\xaa ");
  replace_utf8_all(s, ", i ", ", \xc9\xaa ");
  if (s.size() >= 2 && s[0] == 'i' && s[1] == ' ') {
    static const std::string kIsp("\xc9\xaa ");
    s.replace(0, 2, kIsp);
  }
  for (const char sep : std::string_view(",;:")) {
    const std::string pat = std::string(1, sep) + " i ";
    const std::string rep = std::string(1, sep) + " \xc9\xaa ";
    replace_utf8_all(s, pat, rep);
  }
}

}  // namespace

std::string normalize_russian_ipa_piper_style(std::string ipa) {
  apply_russian_ipa_piper_style(ipa);
  return ipa;
}

std::string utf8_nfc_copy(std::string_view s) {
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

std::string normalize_g2p_ipa_for_piper(std::string_view ipa_utf8, std::string_view piper_lang_key) {
  std::string s = utf8_nfc_copy(ipa_utf8);
  apply_shared_g2p_to_piper_replacements(s);
  apply_lang_specific_replacements(s, piper_lang_key);
  return s;
}

std::string coerce_unknown_ipa_chars_to_piper_inventory(
    std::string_view ipa_utf8, const std::unordered_set<std::string>& phoneme_id_map_keys, const bool use_closest_scalar) {
  const std::string nfc = utf8_nfc_copy(ipa_utf8);
  std::vector<char32_t> pool;
  for (const std::string& k : phoneme_id_map_keys) {
    const auto parts = utf8_split_codepoints(k);
    if (parts.size() != 1) {
      continue;
    }
    const char32_t cp = utf8_singleton_codepoint(parts[0]);
    if (cp == 0) {
      continue;
    }
    if (is_ipa_like_inventory_char(cp)) {
      pool.push_back(cp);
    }
  }
  if (pool.empty()) {
    for (const std::string& k : phoneme_id_map_keys) {
      const auto parts = utf8_split_codepoints(k);
      if (parts.size() != 1) {
        continue;
      }
      const char32_t cp = utf8_singleton_codepoint(parts[0]);
      if (cp != 0) {
        pool.push_back(cp);
      }
    }
  }
  std::sort(pool.begin(), pool.end());
  pool.erase(std::unique(pool.begin(), pool.end()), pool.end());

  std::string out;
  out.reserve(nfc.size() + 8);
  for (const std::string& ch : utf8_split_codepoints(nfc)) {
    if (py_isspace_one_utf8_char(ch)) {
      out += ch;
      continue;
    }
    if (phoneme_id_map_keys.count(ch) != 0) {
      out += ch;
      continue;
    }
    const char32_t cp = utf8_singleton_codepoint(ch);
    if (cp == 0) {
      continue;
    }
    if (category_is_mn_or_me(cp)) {
      continue;
    }
    if (unicode_category_first_char_is_p_or_s(cp)) {
      continue;
    }
    if (use_closest_scalar && !pool.empty()) {
      char32_t best = pool[0];
      int best_metric = std::abs(static_cast<int>(cp) - static_cast<int>(best));
      for (size_t i = 1; i < pool.size(); ++i) {
        const char32_t cand = pool[i];
        const int d = std::abs(static_cast<int>(cp) - static_cast<int>(cand));
        if (d < best_metric || (d == best_metric && cand < best)) {
          best = cand;
          best_metric = d;
        }
      }
      utf8_append_codepoint(out, best);
    }
  }
  return out;
}

std::string ipa_to_piper_ready(std::string_view ipa_utf8, std::string_view piper_lang_key,
                               const std::unordered_set<std::string>& phoneme_id_map_keys, const bool apply_coercion) {
  std::string s = normalize_g2p_ipa_for_piper(ipa_utf8, piper_lang_key);
  if (apply_coercion) {
    s = coerce_unknown_ipa_chars_to_piper_inventory(s, phoneme_id_map_keys, true);
  }
  return s;
}

std::string normalize_g2p_ipa_for_piper_engines(std::string_view ipa_utf8) {
  std::string s(ipa_utf8);
  apply_shared_g2p_to_piper_replacements(s);
  return s;
}

std::vector<std::string> ipa_string_to_phoneme_tokens(const std::string& s) {
  std::string t = trim_copy(s);
  if (t.empty()) {
    return {};
  }
  if (t.find(' ') != std::string::npos) {
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : t) {
      if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
        if (!cur.empty()) {
          parts.push_back(cur);
          cur.clear();
        }
      } else {
        cur.push_back(ch);
      }
    }
    if (!cur.empty()) {
      parts.push_back(cur);
    }
    return parts;
  }
  return utf8_split_codepoints(t);
}

int levenshtein_distance(const std::vector<std::string>& a, const std::vector<std::string>& b) {
  const int la = static_cast<int>(a.size());
  const int lb = static_cast<int>(b.size());
  if (la == 0) {
    return lb;
  }
  if (lb == 0) {
    return la;
  }
  std::vector<int> prev(static_cast<size_t>(lb + 1));
  std::vector<int> cur(static_cast<size_t>(lb + 1));
  for (int j = 0; j <= lb; ++j) {
    prev[static_cast<size_t>(j)] = j;
  }
  for (int i = 1; i <= la; ++i) {
    cur[0] = i;
    for (int j = 1; j <= lb; ++j) {
      const int cost = a[static_cast<size_t>(i - 1)] == b[static_cast<size_t>(j - 1)] ? 0 : 1;
      cur[static_cast<size_t>(j)] =
          std::min({prev[static_cast<size_t>(j)] + 1, cur[static_cast<size_t>(j - 1)] + 1,
                    prev[static_cast<size_t>(j - 1)] + cost});
    }
    prev.swap(cur);
  }
  return prev[static_cast<size_t>(lb)];
}

int pick_closest_alternative_index(const std::vector<std::string>& predicted_phoneme_tokens,
                                    const std::vector<std::string>& ipa_alternatives,
                                    const int n_valid,
                                    const int extra_phonemes) {
  const int n = std::min(n_valid, static_cast<int>(ipa_alternatives.size()));
  if (n <= 0) {
    return 0;
  }
  int best_i = 0;
  int best_d = 1'000'000'000;
  for (int i = 0; i < n; ++i) {
    const auto cand = ipa_string_to_phoneme_tokens(ipa_alternatives[static_cast<size_t>(i)]);
    const int lim = static_cast<int>(cand.size()) + std::max(0, extra_phonemes);
    std::vector<std::string> prefix;
    const int take = std::min(lim, static_cast<int>(predicted_phoneme_tokens.size()));
    prefix.assign(predicted_phoneme_tokens.begin(),
                  predicted_phoneme_tokens.begin() + static_cast<ptrdiff_t>(take));
    const int d = levenshtein_distance(cand, prefix);
    if (d < best_d) {
      best_d = d;
      best_i = i;
    }
  }
  return best_i;
}

std::string pick_closest_cmudict_ipa(const std::vector<std::string>& predicted_phoneme_tokens,
                                     const std::vector<std::string>& cmudict_alternatives,
                                     const int extra_phonemes) {
  if (cmudict_alternatives.empty()) {
    return "";
  }
  if (cmudict_alternatives.size() == 1) {
    return cmudict_alternatives[0];
  }
  const int i = pick_closest_alternative_index(
      predicted_phoneme_tokens, cmudict_alternatives,
      static_cast<int>(cmudict_alternatives.size()), extra_phonemes);
  return cmudict_alternatives[static_cast<size_t>(i)];
}

std::optional<std::string> match_prediction_to_cmudict_ipa(const std::string& predicted,
                                                             const std::vector<std::string>& alts) {
  const std::string e0 = trim_copy(predicted);
  for (const auto& alt : alts) {
    if (trim_copy(alt) == e0) {
      return alt;
    }
  }
  const std::string e1 = strip_length_markers_copy(e0);
  for (const auto& alt : alts) {
    if (strip_length_markers_copy(trim_copy(alt)) == e1) {
      return alt;
    }
  }
  return std::nullopt;
}

}  // namespace moonshine_tts
