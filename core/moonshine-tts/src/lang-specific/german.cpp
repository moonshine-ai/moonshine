#include "german.h"
#include "g2p-word-log.h"
#include "ipa-postprocess.h"
#include "ipa-symbols.h"
#include "utf8-utils.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <istream>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moonshine_tts {
namespace {

using ipa::kPrimaryStressUtf8;
using ipa::kSecondaryStressUtf8;
using moonshine_tts::erase_utf8_substr;
using moonshine_tts::trim_ascii_ws_copy;
using moonshine_tts::utf8_decode_at;
using moonshine_tts::utf8_str_to_u32;

char32_t german_tolower_cp(char32_t c) {
  switch (c) {
  case U'Ä':
    return U'ä';
  case U'Ö':
    return U'ö';
  case U'Ü':
    return U'ü';
  default:
    break;
  }
  if (c >= U'A' && c <= U'Z') {
    return c + 32;
  }
  return c;
}

bool is_key_char(char32_t c) {
  if (c >= U'a' && c <= U'z') {
    return true;
  }
  return c == U'ä' || c == U'ö' || c == U'ü' || c == U'ß';
}

/// NFC-style key: lowercase letters + umlauts + ß only (Python ``normalize_lookup_key``).
std::string normalize_lookup_key_utf8(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    const char32_t cl = german_tolower_cp(cp);
    if (is_key_char(cl)) {
      utf8_append_codepoint(out, cl);
    }
    i += adv;
  }
  return out;
}

bool is_german_word_char(char32_t cp) {
  if (cp == U'-' || (cp >= U'0' && cp <= U'9')) {
    return true;
  }
  if (cp >= U'a' && cp <= U'z') {
    return true;
  }
  if (cp >= U'A' && cp <= U'Z') {
    return true;
  }
  if (cp == U'ä' || cp == U'ö' || cp == U'ü' || cp == U'ß' || cp == U'Ä' || cp == U'Ö' || cp == U'Ü') {
    return true;
  }
  // Mirror Python ``text_to_ipa`` tokenization ``[\w\-]+``: letters outside ASCII German still form
  // one token; :func:`normalize_lookup_key` / rules strips them so e.g. Greek glosses become "".
  if (cp >= 0x00C0u && cp <= 0x00FFu && cp != 0x00D7u && cp != 0x00F7u) {
    return true;
  }
  if (cp >= 0x0100u && cp <= 0x024Fu) {
    return true;
  }
  if (cp >= 0x0370u && cp <= 0x03FFu) {
    return true;
  }
  if (cp >= 0x1F00u && cp <= 0x1FFFu) {
    return true;
  }
  return false;
}

bool is_vowel_l(char32_t ch) {
  switch (ch) {
  case U'a':
  case U'e':
  case U'i':
  case U'o':
  case U'u':
  case U'y':
  case U'ä':
  case U'ö':
  case U'ü':
    return true;
  default:
    return false;
  }
}

std::optional<char32_t> char_before_for_ch(const std::u32string& s, size_t i) {
  if (i == 0) {
    return std::nullopt;
  }
  size_t j = i - 1;
  while (true) {
    if (s[j] == U'-') {
      return std::nullopt;
    }
    if (is_vowel_l(s[j])) {
      return s[j];
    }
    if (s[j] == U'h' && j > 0 && is_vowel_l(s[j - 1])) {
      return s[j - 1];
    }
    if (j == 0) {
      break;
    }
    --j;
  }
  return std::nullopt;
}

std::string ch_ipa_utf8(const std::u32string& full_word_nh, size_t i) {
  if (i > 1 && full_word_nh[i - 2] == U'a' && full_word_nh[i - 1] == U'u') {
    return "x";
  }
  const auto prev = char_before_for_ch(full_word_nh, i);
  if (!prev.has_value()) {
    return "\xC3\xA7"; // ç UTF-8
  }
  switch (*prev) {
  case U'a':
  case U'o':
  case U'u':
    return "x";
  default:
    return "\xC3\xA7";
  }
}

std::string final_devoice(std::string ipa) {
  if (ipa.empty()) {
    return ipa;
  }
  // ɡ U+0261 UTF-8 C9 A1
  if (ipa.size() >= 2 && static_cast<unsigned char>(ipa[ipa.size() - 2]) == 0xC9 &&
      static_cast<unsigned char>(ipa.back()) == 0xA1) {
    ipa.resize(ipa.size() - 2);
    ipa.push_back('k');
    return ipa;
  }
  const char last = ipa.back();
  if (last == 'b') {
    ipa.back() = 'p';
  } else if (last == 'd') {
    ipa.back() = 't';
  } else if (last == 'v') {
    ipa.back() = 'f';
  } else if (last == 'z') {
    ipa.back() = 's';
  }
  return ipa;
}

bool st_sp_at_morpheme_start(const std::u32string& hyphen_word, size_t compact_index) {
  if (compact_index == 0) {
    return true;
  }
  size_t pos = 0;
  size_t start = 0;
  while (start < hyphen_word.size()) {
    const size_t dash = hyphen_word.find(U'-', start);
    const size_t end = dash == std::u32string::npos ? hyphen_word.size() : dash;
    if (end > start) {
      if (compact_index == pos) {
        return true;
      }
      pos += (end - start);
    }
    if (dash == std::u32string::npos) {
      break;
    }
    start = dash + 1;
  }
  return false;
}

size_t unstressed_prefix_len_u32(const std::u32string& w) {
  static const std::u32string prefs[] = {U"wider", U"entgegen", U"ver", U"zer", U"miss",
                                         U"ent",   U"emp",      U"ge",  U"be",  U"er"};
  const size_t n = w.size();
  for (const auto& pref : prefs) {
    const size_t L = pref.size();
    if (n > L && w.compare(0, L, pref) == 0 && w[L] != U'-') {
      return L;
    }
  }
  return 0;
}

std::u32string strip_hyphens_u32(const std::u32string& w) {
  std::u32string o;
  o.reserve(w.size());
  for (char32_t c : w) {
    if (c != U'-') {
      o.push_back(c);
    }
  }
  return o;
}

std::vector<std::pair<size_t, size_t>> german_vowel_nucleus_spans(const std::u32string& w) {
  std::vector<std::pair<size_t, size_t>> spans;
  size_t i = 0;
  const size_t n = w.size();
  while (i < n) {
    if (w[i] == U'-') {
      ++i;
      continue;
    }
    if (!is_vowel_l(w[i])) {
      ++i;
      continue;
    }
    const size_t start = i;
    if (i + 1 < n) {
      const char32_t a = w[i];
      const char32_t b = w[i + 1];
      const std::u32string pair = w.substr(i, 2);
      if (pair == U"au" || pair == U"ei" || pair == U"eu" || pair == U"ai" || pair == U"äu" ||
          pair == U"ey" || pair == U"oi") {
        spans.emplace_back(start, i + 2);
        i += 2;
        continue;
      }
      if (pair == U"ie") {
        if (i + 2 >= n || w[i + 2] == U'-' || !is_vowel_l(w[i + 2])) {
          spans.emplace_back(start, i + 2);
          i += 2;
          continue;
        }
      }
      if (is_vowel_l(a) && b == a && (a == U'a' || a == U'o' || a == U'e' || a == U'i' || a == U'u')) {
        spans.emplace_back(start, i + 2);
        i += 2;
        continue;
      }
    }
    spans.emplace_back(start, start + 1);
    ++i;
  }
  return spans;
}

std::vector<std::u32string> german_orthographic_syllables_u32(const std::u32string& word_lower) {
  std::vector<std::u32string> out;
  std::u32string w;
  w.reserve(word_lower.size());
  for (char32_t c : word_lower) {
    if (c == U'-' || is_vowel_l(c) || (c >= U'a' && c <= U'z') || c == U'ä' || c == U'ö' ||
        c == U'ü' || c == U'ß') {
      w.push_back(c);
    }
  }
  if (w.empty()) {
    return out;
  }
  if (w.find(U'-') != std::u32string::npos) {
    size_t start = 0;
    while (start < w.size()) {
      size_t dash = w.find(U'-', start);
      const size_t end = dash == std::u32string::npos ? w.size() : dash;
      if (end > start) {
        const auto sub = w.substr(start, end - start);
        const auto part = german_orthographic_syllables_u32(sub);
        out.insert(out.end(), part.begin(), part.end());
      }
      if (dash == std::u32string::npos) {
        break;
      }
      start = dash + 1;
    }
    return out;
  }
  const auto spans = german_vowel_nucleus_spans(w);
  if (spans.empty()) {
    out.push_back(w);
    return out;
  }
  const size_t first_s = spans[0].first;
  std::u32string cur = w.substr(0, first_s);
  for (size_t idx = 0; idx < spans.size(); ++idx) {
    const size_t s = spans[idx].first;
    const size_t e = spans[idx].second;
    cur.append(w, s, e - s);
    if (idx + 1 < spans.size()) {
      const size_t next_s = spans[idx + 1].first;
      const std::u32string cluster = w.substr(e, next_s - e);
      out.push_back(cur);
      cur = cluster;
    } else {
      cur.append(w, e, w.size() - e);
      out.push_back(cur);
    }
  }
  out.erase(std::remove_if(out.begin(), out.end(),
                             [](const std::u32string& s) { return s.empty(); }),
            out.end());
  return out;
}

size_t default_stress_syllable_index(const std::vector<std::u32string>& syls, const std::u32string& wl) {
  if (syls.empty()) {
    return 0;
  }
  const size_t n = syls.size();
  if (n == 1) {
    return 0;
  }
  const std::u32string flat = strip_hyphens_u32(wl);
  auto ends_with = [&](const std::u32string& suf) {
    return flat.size() >= suf.size() &&
           flat.compare(flat.size() - suf.size(), suf.size(), suf) == 0;
  };
  if (ends_with(U"ung") || ends_with(U"schaft") || ends_with(U"tion") || ends_with(U"ismus")) {
    return n - 1;
  }
  const size_t plen = unstressed_prefix_len_u32(flat);
  if (plen > 0) {
    size_t acc = 0;
    for (size_t idx = 0; idx < syls.size(); ++idx) {
      acc += syls[idx].size();
      if (acc >= plen) {
        return std::min(idx + 1, n - 1);
      }
    }
  }
  return 0;
}

std::string insert_primary_stress_before_vowel_utf8(std::string s) {
  erase_utf8_substr(s, kPrimaryStressUtf8);
  static const char* kPat[] = {
      "aɪ̯", "aʊ̯", "ɔʏ̯", "iː", "eː", "aː", "oː", "uː", "ɪ", "ʊ",
      "a",   "ɛ",   "ə",   "i",  "o",  "ɔ",  "u",  "y",  "ø", "œ", "ʏ", "ɐ",
  };
  size_t best = std::string::npos;
  for (const char* pat : kPat) {
    const size_t p = s.find(pat);
    if (p != std::string::npos && (best == std::string::npos || p < best)) {
      best = p;
    }
  }
  if (best == std::string::npos) {
    return kPrimaryStressUtf8 + s;
  }
  return s.substr(0, best) + kPrimaryStressUtf8 + s.substr(best);
}

bool ipa_starts_with_nucleus(std::string_view rest) {
  static const std::string_view kNuc[] = {
      "aɪ̯", "aʊ̯", "ɔʏ̯", "ɛɪ̯", "iː", "eː", "aː", "oː", "uː", "yː", "øː",
      "ŋ̩",  "n̩",  "m̩",  "l̩",  "r̩",  "ə",  "ɛ",  "ɜ",  "ɪ",  "ʊ",  "ɐ̯",
      "ɐ",  "ɨ",  "ɵ",  "ø",  "œ",  "ʏ",  "y",  "ɔ",  "ɑ",  "æ",  "a",  "i",  "e",  "o",  "u",
  };
  for (std::string_view p : kNuc) {
    if (rest.size() >= p.size() && rest.substr(0, p.size()) == p) {
      return true;
    }
  }
  return false;
}

size_t ipa_skip_pre_nucleus(std::string_view s, size_t j) {
  if (j >= s.size()) {
    return j;
  }
  if (s.substr(j, kPrimaryStressUtf8.size()) == kPrimaryStressUtf8 ||
      s.substr(j, kSecondaryStressUtf8.size()) == kSecondaryStressUtf8) {
    return j;
  }
  if (ipa_starts_with_nucleus(s.substr(j))) {
    return j;
  }
  static const std::string_view kCons[] = {"t͡s", "p͡f", "d͡ʒ", "t͡ʃ", "tʃ", "ts"};
  for (std::string_view p : kCons) {
    if (s.size() - j >= p.size() && s.substr(j, p.size()) == p) {
      return j + p.size();
    }
  }
  // Advance one UTF-8 codepoint
  const unsigned char c0 = static_cast<unsigned char>(s[j]);
  if (c0 < 0x80) {
    return j + 1;
  }
  if ((c0 >> 5) == 0x6 && j + 1 < s.size()) {
    return j + 2;
  }
  if ((c0 >> 4) == 0xE && j + 2 < s.size()) {
    return j + 3;
  }
  if ((c0 >> 3) == 0x1E && j + 3 < s.size()) {
    return j + 4;
  }
  return j + 1;
}

std::string letters_to_ipa_no_stress(const std::u32string& syl_lower, const std::u32string& full_word_nh,
                                     const std::u32string& hyphen_word, size_t span_start) {
  std::string out;
  const std::u32string& s = syl_lower;
  const size_t n = s.size();
  size_t i = 0;
  while (i < n) {
    const size_t gi = span_start + i;
    const char32_t ch = s[i];

    if (ch == U'-') {
      ++i;
      continue;
    }

    auto append = [&](const char* u8) { out += u8; };

    if (i + 3 < n && s.substr(i, 4) == U"tsch") {
      append("tʃ");
      i += 4;
      continue;
    }
    if (i + 2 < n && s.substr(i, 3) == U"sch") {
      append("ʃ");
      i += 3;
      continue;
    }
    if (i + 2 < n && s.substr(i, 3) == U"chs") {
      append("ks");
      i += 3;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"ch") {
      out += ch_ipa_utf8(full_word_nh, gi);
      i += 2;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"ng") {
      append("ŋ");
      i += 2;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"nk") {
      append("ŋk");
      i += 2;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"pf") {
      append("pf");
      i += 2;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"qu") {
      append("kv");
      i += 2;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"st" && st_sp_at_morpheme_start(hyphen_word, gi)) {
      append("ʃt");
      i += 2;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"sp" && st_sp_at_morpheme_start(hyphen_word, gi)) {
      append("ʃp");
      i += 2;
      continue;
    }
    if (ch == U'h' && i + 1 < n && is_vowel_l(s[i + 1])) {
      ++i;
      continue;
    }
    if (ch == U'h') {
      ++i;
      continue;
    }
    if (ch == U'ß') {
      append("s");
      ++i;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"tz") {
      append("ts");
      i += 2;
      continue;
    }
    if (ch == U'z') {
      append("ts");
      ++i;
      continue;
    }
    if (i + 1 < n && s[i] == U'c' && s[i + 1] == U'k') {
      append("k");
      i += 2;
      continue;
    }
    if (i + 1 < n && s[i] == U'c' && (s[i + 1] == U'e' || s[i + 1] == U'i')) {
      append("ts");
      i += 2;
      continue;
    }
    if (ch == U'c') {
      append("k");
      ++i;
      continue;
    }
    if (ch == U'x') {
      append("ks");
      ++i;
      continue;
    }
    if (ch == U'q' && (i + 1 >= n || s[i + 1] != U'u')) {
      append("k");
      ++i;
      continue;
    }
    if (ch == U'j') {
      append("j");
      ++i;
      continue;
    }
    if (ch == U'v') {
      append("f");
      ++i;
      continue;
    }
    if (ch == U'w') {
      append("v");
      ++i;
      continue;
    }
    if (ch == U'y' && (i + 1 >= n || !is_vowel_l(s[i + 1]))) {
      append("ʏ");
      ++i;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"au") {
      append("aʊ̯");
      i += 2;
      continue;
    }
    if (i + 1 < n && (s.substr(i, 2) == U"ei" || s.substr(i, 2) == U"ai" || s.substr(i, 2) == U"ey")) {
      append("aɪ̯");
      i += 2;
      continue;
    }
    if (i + 1 < n && (s.substr(i, 2) == U"eu" || s.substr(i, 2) == U"äu")) {
      append("ɔʏ̯");
      i += 2;
      continue;
    }
    if (i + 1 < n && s.substr(i, 2) == U"ie" && (i + 2 >= n || s[i + 2] == U'-' || !is_vowel_l(s[i + 2]))) {
      append("iː");
      i += 2;
      continue;
    }
    if (i + 1 < n && is_vowel_l(s[i]) && s[i + 1] == s[i] &&
        (s[i] == U'a' || s[i] == U'o' || s[i] == U'e' || s[i] == U'i' || s[i] == U'u')) {
      switch (s[i]) {
      case U'a':
        append("aː");
        break;
      case U'e':
        append("eː");
        break;
      case U'i':
        append("iː");
        break;
      case U'o':
        append("oː");
        break;
      case U'u':
        append("uː");
        break;
      default:
        break;
      }
      i += 2;
      continue;
    }
    if (is_vowel_l(ch)) {
      if (ch == U'a') {
        append("a");
      } else if (ch == U'e') {
        append(i == n - 1 ? "ə" : "ɛ");
      } else if (ch == U'i') {
        append("ɪ");
      } else if (ch == U'o') {
        append("ɔ");
      } else if (ch == U'u') {
        append("ʊ");
      } else if (ch == U'ä') {
        append("ɛ");
      } else if (ch == U'ö') {
        append("ø");
      } else if (ch == U'ü') {
        append("ʏ");
      } else if (ch == U'y') {
        append("ʏ");
      }
      ++i;
      continue;
    }
    if (ch == U'r') {
      append("ʁ");
      ++i;
      continue;
    }
    if (i + 1 < n && s[i] == U's' && s[i + 1] == U's') {
      append("s");
      i += 2;
      continue;
    }
    if (ch == U's') {
      const bool prev_v = i > 0 && is_vowel_l(s[i - 1]);
      const bool next_v = i + 1 < n && is_vowel_l(s[i + 1]);
      append(prev_v && next_v ? "z" : "s");
      ++i;
      continue;
    }
    switch (ch) {
    case U'b':
      append("b");
      break;
    case U'd':
      append("d");
      break;
    case U'f':
      append("f");
      break;
    case U'g':
      append("\xC9\xA1"); // ɡ
      break;
    case U'k':
      append("k");
      break;
    case U'l':
      append("l");
      break;
    case U'm':
      append("m");
      break;
    case U'n':
      append("n");
      break;
    case U'p':
      append("p");
      break;
    case U't':
      append("t");
      break;
    default:
      break;
    }
    ++i;
  }

  // -ig → …ç (not …lich)
  std::u32string stem = syl_lower;
  while (!stem.empty() && stem.back() == U'-') {
    stem.pop_back();
  }
  if (stem.size() >= 2 && stem.substr(stem.size() - 2) == U"ig") {
    const bool lich = stem.size() >= 4 && stem.substr(stem.size() - 4) == U"lich";
    if (!lich && out.size() >= 2 && static_cast<unsigned char>(out[out.size() - 2]) == 0xC9 &&
        static_cast<unsigned char>(out.back()) == 0xA1) {
      out.resize(out.size() - 2);
      out += "\xC3\xA7";
    }
  }
  return final_devoice(std::move(out));
}

std::string rules_word_to_ipa_utf8(const std::string& raw_word, bool with_stress) {
  std::u32string wl;
  wl.reserve(raw_word.size());
  for (char32_t cp : utf8_str_to_u32(raw_word)) {
    const char32_t cl = german_tolower_cp(cp);
    if (cl == U'-' || is_vowel_l(cl) || (cl >= U'a' && cl <= U'z') || cl == U'ä' || cl == U'ö' ||
        cl == U'ü' || cl == U'ß') {
      wl.push_back(cl);
    }
  }
  if (wl.empty()) {
    return "";
  }
  const std::u32string wl_nh = strip_hyphens_u32(wl);
  const auto syls = german_orthographic_syllables_u32(wl);
  if (syls.empty()) {
    return "";
  }
  const size_t stress_idx =
      with_stress ? default_stress_syllable_index(syls, wl) : static_cast<size_t>(-1);
  size_t offset = 0;
  std::string ipa;
  for (size_t idx = 0; idx < syls.size(); ++idx) {
    std::string chunk = letters_to_ipa_no_stress(syls[idx], wl_nh, wl, offset);
    if (with_stress && idx == stress_idx && !chunk.empty()) {
      chunk = insert_primary_stress_before_vowel_utf8(std::move(chunk));
    }
    ipa += chunk;
    offset += syls[idx].size();
  }
  return ipa;
}

void load_german_lexicon_stream(std::istream& in, std::unordered_map<std::string, std::string>& out_map) {
  struct Slot {
    std::string ipa;
    bool from_lower = false;
  };
  std::unordered_map<std::string, Slot> tmp;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string surf = trim_ascii_ws_copy(std::string_view(line).substr(0, tab));
    std::string ipa = trim_ascii_ws_copy(std::string_view(line).substr(tab + 1));
    if (surf.empty()) {
      continue;
    }
    const std::string key = normalize_lookup_key_utf8(surf);
    if (key.empty()) {
      continue;
    }
    bool lower_lemma = true;
    size_t si = 0;
    while (si < surf.size()) {
      char32_t cp = 0;
      size_t adv = 0;
      utf8_decode_at(surf, si, cp, adv);
      if (german_tolower_cp(cp) != cp) {
        lower_lemma = false;
        break;
      }
      si += adv;
    }
    auto it = tmp.find(key);
    if (it == tmp.end()) {
      tmp[key] = Slot{std::move(ipa), lower_lemma};
    } else {
      if (lower_lemma && !it->second.from_lower) {
        it->second.ipa = std::move(ipa);
        it->second.from_lower = true;
      }
    }
  }
  out_map.clear();
  out_map.reserve(tmp.size());
  for (auto& e : tmp) {
    out_map.emplace(std::move(e.first), std::move(e.second.ipa));
  }
}

void load_german_lexicon_file(const std::filesystem::path& path,
                              std::unordered_map<std::string, std::string>& out_map) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("German lexicon: cannot open " + path.generic_string());
  }
  load_german_lexicon_stream(in, out_map);
}

}  // namespace

namespace {

bool g2p_all_ascii_digits(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  for (char c : s) {
    if (c < '0' || c > '9') {
      return false;
    }
  }
  return true;
}

std::string german_under_100_word(int n) {
  static const char* kDigitStandalone[] = {"null", "eins", "zwei", "drei", "vier", "fünf", "sechs",
                                           "sieben", "acht", "neun"};
  static const char* kUnitCompound[] = {"", "ein", "zwei", "drei", "vier", "fünf", "sechs",
                                        "sieben", "acht", "neun"};
  static const char* kTens[] = {"", "", "zwanzig", "dreißig", "vierzig", "fünfzig",
                                "sechzig", "siebzig", "achtzig", "neunzig"};
  if (n < 0 || n >= 100) {
    throw std::out_of_range("german_under_100_word");
  }
  if (n < 10) {
    return kDigitStandalone[n];
  }
  if (n < 20) {
    static const char* teens[] = {"zehn",    "elf",      "zwölf",    "dreizehn", "vierzehn",
                                  "fünfzehn", "sechzehn", "siebzehn", "achtzehn", "neunzehn"};
    return teens[n - 10];
  }
  const int t = n / 10;
  const int u = n % 10;
  const char* tw = kTens[t];
  if (u == 0) {
    return tw;
  }
  return std::string(kUnitCompound[u]) + "und" + tw;
}

std::string german_hundred_head(int h) {
  if (h == 1) {
    return "hundert";
  }
  if (h < 2 || h > 9) {
    throw std::out_of_range("german_hundred_head");
  }
  static const char* stems[] = {"zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun"};
  return std::string(stems[h - 2]) + "hundert";
}

void append_german_tokens_1_999(int n, std::vector<std::string>& out) {
  if (n < 1 || n > 999) {
    throw std::out_of_range("append_german_tokens_1_999");
  }
  if (n < 100) {
    out.push_back(german_under_100_word(n));
    return;
  }
  const int h = n / 100;
  const int r = n % 100;
  out.push_back(german_hundred_head(h));
  if (r != 0) {
    out.push_back(german_under_100_word(r));
  }
}

void append_german_tokens_thousands(int q, std::vector<std::string>& out) {
  if (q < 1 || q > 999) {
    throw std::out_of_range("append_german_tokens_thousands");
  }
  if (q == 1) {
    out.emplace_back("eintausend");
    return;
  }
  append_german_tokens_1_999(q, out);
  out.emplace_back("tausend");
}

void append_german_below_1_000_000(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 1'000'000) {
    throw std::out_of_range("append_german_below_1_000_000");
  }
  if (n < 1000) {
    append_german_tokens_1_999(n, out);
    return;
  }
  const int q = n / 1000;
  const int r = n % 1000;
  append_german_tokens_thousands(q, out);
  if (r != 0) {
    append_german_tokens_1_999(r, out);
  }
}

std::string expand_cardinal_digits_to_german_words(std::string_view s) {
  if (!g2p_all_ascii_digits(s)) {
    return std::string(s);
  }
  if (s.size() > 1 && s[0] == '0') {
    static const char* dw[] = {"null", "eins", "zwei", "drei", "vier", "fünf", "sechs",
                               "sieben", "acht", "neun"};
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) {
        o.push_back(' ');
      }
      o += dw[static_cast<size_t>(s[i] - '0')];
    }
    return o;
  }
  long long n = 0;
  for (char c : s) {
    n = n * 10 + (c - '0');
  }
  if (n > 999'999) {
    return std::string(s);
  }
  if (n == 0) {
    return "null";
  }
  std::vector<std::string> parts;
  append_german_below_1_000_000(static_cast<int>(n), parts);
  std::string o;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      o.push_back(' ');
    }
    o += parts[i];
  }
  return o;
}

std::string expand_german_digit_tokens_in_text(std::string text) {
  static const std::regex range_re(R"(\b(\d+)-(\d+)\b)", std::regex::ECMAScript);
  static const std::regex dig_re(R"(\b\d+\b)", std::regex::ECMAScript);
  std::string out;
  std::sregex_iterator it(text.begin(), text.end(), range_re);
  std::sregex_iterator end;
  size_t pos = 0;
  for (; it != end; ++it) {
    const std::smatch& m = *it;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    const size_t g1s = static_cast<size_t>(m.position(1));
    const size_t g1e = g1s + m.str(1).size();
    const size_t g2s = static_cast<size_t>(m.position(2));
    const size_t g2e = g2s + m.str(2).size();
    if (digit_ascii_span_expandable_python_w(text, g1s, g1e) &&
        digit_ascii_span_expandable_python_w(text, g2s, g2e)) {
      out += expand_cardinal_digits_to_german_words(m.str(1));
      out += " bis ";
      out += expand_cardinal_digits_to_german_words(m.str(2));
    } else {
      out += m.str();
    }
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  text = std::move(out);
  out.clear();
  pos = 0;
  std::sregex_iterator it2(text.begin(), text.end(), dig_re);
  for (; it2 != end; ++it2) {
    const std::smatch& m = *it2;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    const size_t gs = static_cast<size_t>(m.position());
    const size_t ge = gs + m.str().size();
    if (digit_ascii_span_expandable_python_w(text, gs, ge)) {
      out += expand_cardinal_digits_to_german_words(m.str());
    } else {
      out += m.str();
    }
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

}  // namespace

bool ipa_at_stress_mark(const std::string& ipa, size_t j) {
  if (ipa.size() - j >= kPrimaryStressUtf8.size() &&
      ipa.compare(j, kPrimaryStressUtf8.size(), kPrimaryStressUtf8) == 0) {
    return true;
  }
  return ipa.size() - j >= kSecondaryStressUtf8.size() &&
         ipa.compare(j, kSecondaryStressUtf8.size(), kSecondaryStressUtf8) == 0;
}

std::string GermanRuleG2p::normalize_ipa_stress_for_vocoder(std::string ipa) {
  if (ipa.empty()) {
    return ipa;
  }
  if (ipa.find(kPrimaryStressUtf8) == std::string::npos &&
      ipa.find(kSecondaryStressUtf8) == std::string::npos) {
    return ipa;
  }
  std::string out;
  size_t i = 0;
  while (i < ipa.size()) {
    const size_t pri = ipa.find(kPrimaryStressUtf8, i);
    const size_t sec = ipa.find(kSecondaryStressUtf8, i);
    size_t next = std::string::npos;
    const std::string* mark = nullptr;
    if (pri != std::string::npos && (sec == std::string::npos || pri <= sec)) {
      next = pri;
      mark = &kPrimaryStressUtf8;
    } else if (sec != std::string::npos) {
      next = sec;
      mark = &kSecondaryStressUtf8;
    }
    if (next == std::string::npos) {
      out.append(ipa, i, ipa.size() - i);
      break;
    }
    out.append(ipa, i, next - i);
    const size_t after_mark = next + mark->size();
    size_t j = after_mark;
    while (j < ipa.size() && !ipa_at_stress_mark(ipa, j) &&
           !ipa_starts_with_nucleus(std::string_view(ipa).substr(j))) {
      const size_t j2 = ipa_skip_pre_nucleus(ipa, j);
      if (j2 == j) {
        break;
      }
      j = j2;
    }
    out.append(ipa, after_mark, j - after_mark);
    out += *mark;
    i = j;
  }
  return out;
}

GermanRuleG2p::GermanRuleG2p(std::filesystem::path dict_tsv)
    : GermanRuleG2p(std::move(dict_tsv), Options{}) {}

GermanRuleG2p::GermanRuleG2p(std::filesystem::path dict_tsv, Options options)
    : options_(options) {
  load_german_lexicon_file(std::move(dict_tsv), lexicon_);
}

GermanRuleG2p::GermanRuleG2p(std::string dict_tsv_utf8, Options options)
    : options_(options) {
  std::istringstream in(std::move(dict_tsv_utf8));
  load_german_lexicon_stream(in, lexicon_);
}

std::string GermanRuleG2p::finalize_ipa(std::string ipa) const {
  if (!options_.with_stress) {
    erase_utf8_substr(ipa, kPrimaryStressUtf8);
    erase_utf8_substr(ipa, kSecondaryStressUtf8);
    return normalize_german_ipa_piper_style(std::move(ipa));
  }
  if (options_.vocoder_stress) {
    ipa = normalize_ipa_stress_for_vocoder(std::move(ipa));
  }
  return normalize_german_ipa_piper_style(std::move(ipa));
}

std::string GermanRuleG2p::lookup_or_rules(const std::string& raw_word) const {
  const std::string letters_only = normalize_lookup_key_utf8(raw_word);
  if (letters_only.empty()) {
    return "";
  }
  auto it = lexicon_.find(letters_only);
  if (it != lexicon_.end()) {
    return finalize_ipa(it->second);
  }
  static const std::string kSs{"\xC3\x9F"};  // ß UTF-8
  if (letters_only.find(kSs) != std::string::npos) {
    std::string alt = letters_only;
    size_t p = 0;
    while ((p = alt.find(kSs, p)) != std::string::npos) {
      alt.replace(p, kSs.size(), "ss");
      p += 2;
    }
    auto it2 = lexicon_.find(alt);
    if (it2 != lexicon_.end()) {
      return finalize_ipa(it2->second);
    }
  } else if (letters_only.find("ss") != std::string::npos) {
    std::string alt = letters_only;
    const size_t p = alt.find("ss");
    if (p != std::string::npos) {
      alt.replace(p, 2, kSs);
      auto it2 = lexicon_.find(alt);
      if (it2 != lexicon_.end()) {
        return finalize_ipa(it2->second);
      }
    }
  }
  if (letters_only.find('-') != std::string::npos) {
    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < letters_only.size()) {
      size_t dash = letters_only.find('-', start);
      const size_t end = dash == std::string::npos ? letters_only.size() : dash;
      if (end > start) {
        chunks.push_back(letters_only.substr(start, end - start));
      }
      if (dash == std::string::npos) {
        break;
      }
      start = dash + 1;
    }
    bool ok = true;
    std::string merged;
    for (size_t c = 0; c < chunks.size(); ++c) {
      auto hc = lexicon_.find(chunks[c]);
      if (hc == lexicon_.end()) {
        ok = false;
        break;
      }
      if (c > 0) {
        merged += '-';
      }
      merged += hc->second;
    }
    if (ok) {
      return finalize_ipa(std::move(merged));
    }
  }
  return finalize_ipa(rules_word_to_ipa_utf8(raw_word, options_.with_stress));
}

std::string GermanRuleG2p::word_to_ipa(const std::string& word) const {
  const std::string wraw = trim_ascii_ws_copy(word);
  if (wraw.empty()) {
    return "";
  }
  if (options_.expand_cardinal_digits && g2p_all_ascii_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_german_words(wraw);
    if (phrase != wraw) {
      return text_to_ipa_no_expand(phrase, nullptr);
    }
    return wraw;
  }
  if (!options_.expand_cardinal_digits) {
    static const std::regex dig_pass(R"(^[0-9]+(?:-[0-9]+)*$)", std::regex::ECMAScript);
    if (std::regex_match(wraw, dig_pass)) {
      return wraw;
    }
  }
  return lookup_or_rules(wraw);
}

std::string GermanRuleG2p::text_to_ipa_no_expand(const std::string& text,
                                                 std::vector<G2pWordLog>* per_word_log) const {
  std::string out;
  size_t pos = 0;
  const size_t n = text.size();
  while (pos < n) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(text, pos, cp, adv);
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
      out.push_back(' ');
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (cp != U' ' && cp != U'\t' && cp != U'\n' && cp != U'\r') {
          break;
        }
        pos += adv;
      }
      continue;
    }
    if (is_german_word_char(cp)) {
      const size_t start = pos;
      pos += adv;
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (!is_german_word_char(cp)) {
          break;
        }
        pos += adv;
      }
      const std::string tok = text.substr(start, pos - start);
      const std::string k = normalize_lookup_key_utf8(tok);
      std::string wipa = word_to_ipa(tok);
      if (per_word_log != nullptr) {
        per_word_log->push_back(
            G2pWordLog{tok, k, G2pWordPath::kRuleBasedG2p, std::move(wipa)});
        out += per_word_log->back().ipa;
      } else {
        out += wipa;
      }
      continue;
    }
    const size_t start = pos;
    pos += adv;
    while (pos < n) {
      utf8_decode_at(text, pos, cp, adv);
      if (is_german_word_char(cp) || cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
        break;
      }
      pos += adv;
    }
    out.append(text, start, pos - start);
  }
  std::string collapsed;
  bool in_sp = false;
  for (char c : out) {
    if (c == ' ') {
      if (!in_sp && !collapsed.empty()) {
        collapsed.push_back(' ');
      }
      in_sp = true;
    } else {
      in_sp = false;
      collapsed.push_back(c);
    }
  }
  while (!collapsed.empty() && collapsed.front() == ' ') {
    collapsed.erase(collapsed.begin());
  }
  while (!collapsed.empty() && collapsed.back() == ' ') {
    collapsed.pop_back();
  }
  return collapsed;
}

std::string GermanRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  if (options_.expand_cardinal_digits) {
    text = expand_german_digit_tokens_in_text(std::move(text));
  }
  return text_to_ipa_no_expand(text, per_word_log);
}

bool dialect_resolves_to_german_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "de" || s == "de-de" || s == "german";
}

std::vector<std::string> GermanRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first({"de", "de-DE", "de_de", "german"});
}

}  // namespace moonshine_tts
