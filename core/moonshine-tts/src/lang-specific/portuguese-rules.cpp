#include "portuguese-rules.h"

#include "ipa-symbols.h"
#include "utf8-utils.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moonshine_tts::portuguese_rules {

namespace {
const std::string& kPri = ipa::kPrimaryStressUtf8;
const std::string& kSec = ipa::kSecondaryStressUtf8;
}  // namespace

char32_t pt_tolower(char32_t c) {
  switch (c) {
  case U'À':
    return U'à';
  case U'Á':
    return U'á';
  case U'Â':
    return U'â';
  case U'Ã':
    return U'ã';
  case U'Ç':
    return U'ç';
  case U'É':
    return U'é';
  case U'Ê':
    return U'ê';
  case U'Í':
    return U'í';
  case U'Ó':
    return U'ó';
  case U'Ô':
    return U'ô';
  case U'Õ':
    return U'õ';
  case U'Ú':
    return U'ú';
  case U'Ü':
    return U'ü';
  case U'Ý':
    return U'ý';
  default:
    break;
  }
  if (c >= U'A' && c <= U'Z') {
    return c + 32;
  }
  if ((c >= U'\u00C0' && c <= U'\u00D6') || (c >= U'\u00D8' && c <= U'\u00DE')) {
    return c + 32;
  }
  return c;
}

namespace {

bool is_pt_key_cp(char32_t c) {
  c = pt_tolower(c);
  if (c == U'\u2019') {
    c = U'\'';
  }
  if (c >= U'a' && c <= U'z') {
    return true;
  }
  if (c == U'\'' || c == U'-') {
    return true;
  }
  return c == U'à' || c == U'á' || c == U'â' || c == U'ã' || c == U'ç' || c == U'é' || c == U'ê' ||
         c == U'í' || c == U'ó' || c == U'ô' || c == U'õ' || c == U'ú' || c == U'ü' || c == U'ý';
}

std::string normalize_lookup_key_utf8_impl(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    moonshine_tts::utf8_decode_at(word, i, cp, adv);
    if (cp == U'\u2019') {
      cp = U'\'';
    }
    const char32_t cl = pt_tolower(cp);
    if (is_pt_key_cp(cl)) {
      moonshine_tts::utf8_append_codepoint(out, cl);
    }
    i += adv;
  }
  return out;
}

}  // namespace

std::string normalize_lookup_key_utf8(const std::string& word) {
  return normalize_lookup_key_utf8_impl(word);
}

std::u32string utf8_to_u32_pt(const std::string& s) {
  return moonshine_tts::utf8_str_to_u32(s);
}

std::string u32_to_utf8_pt(const std::u32string& s) {
  std::string o;
  for (char32_t c : s) {
    moonshine_tts::utf8_append_codepoint(o, c);
  }
  return o;
}

bool is_allowed_pt_grapheme(char32_t c) {
  c = pt_tolower(c);
  if (c == U'-') {
    return true;
  }
  if (c >= U'a' && c <= U'z') {
    return true;
  }
  return c == U'à' || c == U'á' || c == U'â' || c == U'ã' || c == U'ç' || c == U'é' || c == U'ê' ||
         c == U'í' || c == U'ó' || c == U'ô' || c == U'õ' || c == U'ú' || c == U'ü' || c == U'ý';
}

std::u32string filter_pt_word_graphemes_utf8(const std::string& word) {
  const std::string t = moonshine_tts::trim_ascii_ws_copy(word);
  std::u32string o;
  size_t i = 0;
  while (i < t.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    moonshine_tts::utf8_decode_at(t, i, cp, adv);
    if (is_allowed_pt_grapheme(cp)) {
      o.push_back(pt_tolower(cp));
    }
    i += adv;
  }
  return o;
}

bool is_vowel_pt_u32(char32_t ch) {
  ch = pt_tolower(ch);
  return ch == U'a' || ch == U'e' || ch == U'i' || ch == U'o' || ch == U'u' || ch == U'à' || ch == U'á' ||
         ch == U'â' || ch == U'ã' || ch == U'é' || ch == U'ê' || ch == U'í' || ch == U'ó' || ch == U'ô' ||
         ch == U'õ' || ch == U'ú' || ch == U'ü' || ch == U'ý';
}

char32_t strip_accent_base_pt(char32_t c) {
  c = pt_tolower(c);
  switch (c) {
  case U'à':
  case U'á':
  case U'â':
  case U'ã':
    return U'a';
  case U'é':
  case U'ê':
    return U'e';
  case U'í':
    return U'i';
  case U'ó':
  case U'ô':
  case U'õ':
    return U'o';
  case U'ú':
  case U'ü':
    return U'u';
  case U'ý':
    return U'y';
  default:
    return c;
  }
}

bool should_hiatus_pt_u32(char32_t a, char32_t b) {
  const char32_t al = pt_tolower(a);
  const char32_t bl = pt_tolower(b);
  if (al == U'í' || al == U'ú' || al == U'ý' || bl == U'í' || bl == U'ú' || bl == U'ý') {
    return true;
  }
  const char32_t ba = strip_accent_base_pt(al);
  const char32_t bb = strip_accent_base_pt(bl);
  if (ba == bb) {
    return true;
  }
  if (al == U'ã' || al == U'õ' || bl == U'ã' || bl == U'õ') {
    if (ba <= U'u' && bb <= U'u' && ba >= U'a' && bb >= U'a' && (ba == U'a' || ba == U'e' || ba == U'o') &&
        (bb == U'a' || bb == U'e' || bb == U'o')) {
      return true;
    }
    return false;
  }
  const bool sa = (ba == U'a' || ba == U'e' || ba == U'o');
  const bool sb = (bb == U'a' || bb == U'e' || bb == U'o');
  if (sa && sb) {
    if (al == U'á' || al == U'é' || al == U'ó' || al == U'â' || al == U'ê' || al == U'ô' || bl == U'á' ||
        bl == U'é' || bl == U'ó' || bl == U'â' || bl == U'ê' || bl == U'ô') {
      return true;
    }
    if (ba == U'a' && bb == U'e') {
      return false;
    }
    if (ba == U'e' && bb == U'a') {
      return false;
    }
    return true;
  }
  return false;
}

std::vector<std::pair<size_t, size_t>> vowel_nucleus_spans_u32(const std::u32string& w) {
  std::vector<std::pair<size_t, size_t>> out;
  size_t i = 0;
  const size_t n = w.size();
  while (i < n) {
    if (!is_vowel_pt_u32(w[i])) {
      ++i;
      continue;
    }
    if (w[i] == U'ã' && i + 1 < n && w[i + 1] == U'o') {
      out.emplace_back(i, i + 2);
      i += 2;
      continue;
    }
    if (w[i] == U'ã' && i + 1 < n && w[i + 1] == U'e') {
      out.emplace_back(i, i + 2);
      i += 2;
      continue;
    }
    if (i + 1 < n && is_vowel_pt_u32(w[i + 1])) {
      if (should_hiatus_pt_u32(w[i], w[i + 1])) {
        out.emplace_back(i, i + 1);
        i += 1;
      } else {
        out.emplace_back(i, i + 2);
        i += 2;
      }
    } else {
      out.emplace_back(i, i + 1);
      i += 1;
    }
  }
  return out;
}

bool valid_onset2_end_u32(char32_t a, char32_t b) {
  static const std::u32string k2[] = {U"bl", U"br", U"cl", U"cr", U"dr", U"fl", U"fr", U"gl", U"gr",
                                      U"pl", U"pr", U"tr", U"ch"};
  std::u32string two;
  two.push_back(a);
  two.push_back(b);
  for (const auto& s : k2) {
    if (s == two) {
      return true;
    }
  }
  return false;
}

std::pair<std::u32string, std::u32string> split_intervocalic_cluster_u32(const std::u32string& cluster) {
  if (cluster.empty()) {
    return {{}, {}};
  }
  if (cluster == U"rr") {
    return {{}, U"rr"};
  }
  const size_t n = cluster.size();
  if (n >= 2 && cluster[n - 2] == U'l' && cluster[n - 1] == U'h') {
    return {cluster.substr(0, n - 2), cluster.substr(n - 2)};
  }
  if (n >= 2 && cluster[n - 2] == U'n' && cluster[n - 1] == U'h') {
    return {cluster.substr(0, n - 2), cluster.substr(n - 2)};
  }
  if (n >= 2 && valid_onset2_end_u32(cluster[n - 2], cluster[n - 1])) {
    return {cluster.substr(0, n - 2), cluster.substr(n - 2)};
  }
  return {cluster.substr(0, n - 1), cluster.substr(n - 1)};
}

std::vector<std::u32string> port_orthographic_syllables_u32(const std::u32string& w0) {
  if (w0.empty()) {
    return {};
  }
  if (w0.find(U'-') != std::u32string::npos) {
    std::vector<std::u32string> parts;
    size_t a = 0;
    while (a <= w0.size()) {
      size_t d = w0.find(U'-', a);
      if (d == std::u32string::npos) {
        d = w0.size();
      }
      if (d > a) {
        const std::u32string chunk = w0.substr(a, d - a);
        auto sub = port_orthographic_syllables_u32(chunk);
        parts.insert(parts.end(), sub.begin(), sub.end());
      }
      if (d == w0.size()) {
        break;
      }
      a = d + 1;
    }
    return parts;
  }
  const std::u32string& w = w0;
  const auto spans = vowel_nucleus_spans_u32(w);
  if (spans.empty()) {
    return {w};
  }
  std::vector<std::u32string> syllables;
  const size_t first_s = spans[0].first;
  std::u32string cur = w.substr(0, first_s);
  for (size_t idx = 0; idx < spans.size(); ++idx) {
    const auto [s, e] = spans[idx];
    cur.append(w.substr(s, e - s));
    if (idx + 1 < spans.size()) {
      const size_t next_s = spans[idx + 1].first;
      const std::u32string cluster = w.substr(e, next_s - e);
      const auto [coda, onset] = split_intervocalic_cluster_u32(cluster);
      syllables.push_back(cur + coda);
      cur = onset;
    } else {
      syllables.push_back(cur + w.substr(e));
    }
  }
  std::vector<std::u32string> nonempty;
  for (auto& sy : syllables) {
    if (!sy.empty()) {
      nonempty.push_back(std::move(sy));
    }
  }
  return nonempty;
}

bool accented_syllable_u32(const std::u32string& s) {
  for (char32_t c : s) {
    if (c == U'á' || c == U'à' || c == U'â' || c == U'é' || c == U'ê' || c == U'í' || c == U'ó' || c == U'ô' ||
        c == U'ú') {
      return true;
    }
  }
  return false;
}

size_t default_stressed_syllable_index_u32(const std::vector<std::u32string>& syls, const std::u32string& w) {
  if (syls.empty()) {
    return 0;
  }
  for (size_t i = 0; i < syls.size(); ++i) {
    if (accented_syllable_u32(syls[i])) {
      return i;
    }
  }
  const size_t n = syls.size();
  if (n == 1) {
    return 0;
  }
  const size_t ln = w.size();
  auto ends_with = [&](const std::u32string& suf) {
    return ln >= suf.size() && w.compare(ln - suf.size(), suf.size(), suf) == 0;
  };
  if (ends_with(U"ões") || ends_with(U"ãos") || ends_with(U"ão")) {
    return n - 1;
  }
  if (ends_with(U"ã") || ends_with(U"ãs")) {
    return n - 1;
  }
  if (ln == 0) {
    return 0;
  }
  const char32_t last = w[ln - 1];
  if (last == U's' && ln >= 2) {
    const char32_t prev = w[ln - 2];
    if (prev == U'a' || prev == U'e' || prev == U'i' || prev == U'o' || prev == U'u' || prev == U'á' ||
        prev == U'é' || prev == U'í' || prev == U'ó' || prev == U'ú' || prev == U'ã' || prev == U'õ' ||
        prev == U'â' || prev == U'ê' || prev == U'ô') {
      return (n >= 2) ? n - 2 : 0;
    }
  }
  if (last == U'a' || last == U'e' || last == U'o' || last == U'á' || last == U'é' || last == U'ó') {
    return (n >= 2) ? n - 2 : 0;
  }
  if (ends_with(U"em") || ends_with(U"ens") || ends_with(U"am")) {
    return (n >= 2) ? n - 2 : 0;
  }
  if (last == U'i' || last == U'u' || last == U'í' || last == U'ú') {
    return n - 1;
  }
  if (last == U'r' || last == U'l' || last == U'z' || last == U'x') {
    return n - 1;
  }
  if (last == U'n' && !ends_with(U"em")) {
    return n - 1;
  }
  if (last == U'e') {
    return (n >= 2) ? n - 2 : 0;
  }
  if (last == U'm') {
    return (n >= 2) ? n - 2 : 0;
  }
  return (n >= 2) ? n - 2 : 0;
}

std::string strip_stress_chars(std::string s) {
  moonshine_tts::erase_utf8_substr(s, kPri);
  moonshine_tts::erase_utf8_substr(s, kSec);
  return s;
}

std::string insert_primary_stress_before_vowel_utf8(std::string ipa) {
  ipa = strip_stress_chars(std::move(ipa));
  std::u32string u = utf8_to_u32_pt(ipa);
  for (size_t i = 0; i < u.size(); ++i) {
    const char32_t ch = u[i];
    if (ch == U'a' || ch == U'e' || ch == U'i' || ch == U'o' || ch == U'u' || ch == U'\u025B' || ch == U'\u0254' ||
        ch == U'\u0250' || ch == U'\u026A' || ch == U'\u028A' || ch == U'\u0268' || ch == U'\u0259' ||
        ch == U'\u00E6' || ch == U'\u0254') {
      std::string pre;
      for (size_t j = 0; j < i; ++j) {
        moonshine_tts::utf8_append_codepoint(pre, u[j]);
      }
      std::string post;
      for (size_t j = i; j < u.size(); ++j) {
        moonshine_tts::utf8_append_codepoint(post, u[j]);
      }
      return pre + kPri + post;
    }
  }
  return kPri + ipa;
}

std::optional<int> roman_to_int_ascii(std::string_view u) {
  std::string s(u);
  for (char& c : s) {
    c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  }
  if (s.empty()) {
    return std::nullopt;
  }
  for (char c : s) {
    if (c != 'I' && c != 'V' && c != 'X' && c != 'L' && c != 'C' && c != 'D' && c != 'M') {
      return std::nullopt;
    }
  }
  const std::unordered_map<char, int> vals{{'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}};
  int total = 0;
  size_t i = 0;
  while (i < s.size()) {
    const int v = vals.at(s[i]);
    if (i + 1 < s.size()) {
      const int nv = vals.at(s[i + 1]);
      if (nv > v) {
        total += nv - v;
        i += 2;
        continue;
      }
    }
    total += v;
    ++i;
  }
  if (total <= 0 || total >= 4000) {
    return std::nullopt;
  }
  return total;
}

std::optional<std::string> roman_numeral_token_to_ipa(const std::string& letters_lower, bool is_pt_pt) {
  if (letters_lower.find('-') != std::string::npos || letters_lower.find('\'') != std::string::npos) {
    return std::nullopt;
  }
  const auto n = roman_to_int_ascii(letters_lower);
  if (!n.has_value()) {
    return std::nullopt;
  }
  static const std::unordered_map<int, std::string> kCard = {
      {1, "\xcb\x88\xc5\xa9"},
      {2, "\xcb\x88\x64\xc9\x94\x6a\x73"},
      {3, "\xcb\x88\x74\xc9\xbe\xc9\x9b\x6a\x73"},
      {4, "\xcb\x88\x6b\x77\x61\x74\xc9\xbe\xca\x8a"},
      {5, "\xcb\x88\x73\x69\xcc\x83\x6b\xca\x8a"},
      {6, "\xcb\x88\x73\x65\x6a\x73"},
      {7, "\xcb\x88\x73\xc9\x9b\x74\xca\x83\x69"},
      {8, "\xcb\x88\xc9\x94\x6a\x74\xca\x8a"},
      {9, "\xcb\x88\x6e\xc9\x94\x76\x69"},
      {10, "\xcb\x88\x64\xc9\x9b\x6a\x73"},
      {11, "\xcb\x88\xc9\x94\xcc\x83\x7a\x69"},
      {12, "\xcb\x88\x64\xc9\x94\x7a\x69"},
      {13, "\xcb\x88\x74\xc9\xbe\xc9\x9b\x7a\x69"},
      {14, "\x6b\x61\xc9\xaa\xcb\x88\xc9\x94\xc9\xbe\x7a\x69"},
      {15, "\xcb\x88\x6b\x69\xcc\x83\x7a\x69"},
      {16, "\x64\xc9\x9b\xcb\x88\x7a\x65\x73\x65\x6a\x73"},
      {17, "\x64\xc9\x9b\xcb\x88\x7a\x65\x73\xc9\x9b\x74\xca\x83\x69"},
      {18, "\x64\xc9\x9b\xcb\x88\x7a\x65\x6a\x7a\x6a"},
      {19, "\x64\xc9\x9b\x7a\x65\x6e\xcb\x88\xc9\x94\x76\x69"},
      {20, "\xcb\x88\x76\xc4\xa9\x74\xca\x83\x69"},
      {21, "\x76\xc4\xa9\xcb\x88\x74\xca\x83\x69\xcb\x88\x65\xc5\xa9"},
      {30, "\xcb\x88\x74\xc9\xbe\xc4\xa9\x74\xca\x83\x69"},
      {40, "\x6b\x77\xc9\x90\xcb\x88\xc9\xbe\xe1\xba\xbd\x74\xc9\x90"},
      {50, "\xcb\x88\x73\xc4\xa9\x6b\x77\xe1\xba\xbd\x74\xc9\x90"},
      {60, "\xcb\x88\x73\x65\x73\x73\xe1\xba\xbd\x74\xca\x83\x69"},
      {70, "\x73\xc9\x9b\xcb\x88\x74\xe1\xba\xbd\x74\xca\x83\x69"},
      {80, "\xcb\x88\x6f\x6a\x74\xe1\xba\xbd\x74\xca\x83\x69"},
      {90, "\xcb\x88\x6e\xc9\x94\x76\xe1\xba\xbd\x74\xca\x83\x69"},
      {100, "\xcb\x88\x73\xe1\xba\xbd\x74\xca\x83\x69"},
  };
  const auto it = kCard.find(*n);
  if (it == kCard.end()) {
    return std::nullopt;
  }
  std::string ipa = it->second;
  if (is_pt_pt) {
    static const std::string from = "\xcb\x88\x76\xc4\xa9\x74\xca\x83\x69";
    static const std::string to = "\xcb\x88\x76\xc4\xa9\x74\xca\x83\xc9\xa8";
    size_t p = 0;
    while ((p = ipa.find(from, p)) != std::string::npos) {
      ipa.replace(p, from.size(), to);
      p += to.size();
    }
  }
  return ipa;
}

bool prev_global_vowel_u32(const std::u32string& full_word, size_t gidx) {
  if (gidx == 0) {
    return false;
  }
  size_t j = gidx - 1;
  while (true) {
    if (is_vowel_pt_u32(full_word[j])) {
      return true;
    }
    if (full_word[j] == U'-') {
      break;
    }
    if (j == 0) {
      break;
    }
    --j;
  }
  return false;
}

bool next_global_vowel_u32(const std::u32string& full_word, size_t gidx) {
  // Mirrors Python ``_next_global_vowel``: ``j = gidx + 1`` (first index examined is after *gidx*).
  size_t j = gidx + 1;
  while (j < full_word.size()) {
    if (is_vowel_pt_u32(full_word[j])) {
      return true;
    }
    if (full_word[j] == U'-') {
      break;
    }
    ++j;
  }
  return false;
}

bool syllable_has_u32(const std::u32string& s, char32_t ch) {
  return s.find(ch) != std::u32string::npos;
}

std::string letters_to_ipa_no_stress_u32(const std::u32string& s, bool is_pt_pt, const std::u32string& full_word,
                                         size_t span_start, bool stressed_syllable) {
  auto unstressed_vowel = [&](char32_t base) -> std::string {
    if (stressed_syllable) {
      return u32_to_utf8_pt(std::u32string(1, base));
    }
    if (is_pt_pt) {
      if (base == U'a') {
        return "\xc9\x90";
      }
      if (base == U'e') {
        return "\xc9\xa8";
      }
      if (base == U'i') {
        return "i";
      }
      if (base == U'o') {
        return "u";
      }
      if (base == U'u') {
        return "u";
      }
    } else {
      if (base == U'a') {
        return "\xc9\x90";
      }
      if (base == U'e') {
        return "\xc9\xaa";
      }
      if (base == U'i') {
        return "i";
      }
      if (base == U'o') {
        return "\xca\x8a";
      }
      if (base == U'u') {
        return "u";
      }
    }
    return u32_to_utf8_pt(std::u32string(1, base));
  };

  auto map_vowel_char = [&](char32_t ch) -> std::string {
    const char32_t cl = pt_tolower(ch);
    if (cl == U'à' || cl == U'á' || cl == U'â') {
      return "a";
    }
    if (cl == U'é') {
      return "\xc9\x9b";
    }
    if (cl == U'ê') {
      return "\xc9\x9b";
    }
    if (cl == U'í') {
      return "i";
    }
    if (cl == U'ó') {
      return "\xc9\x94";
    }
    if (cl == U'ô') {
      return "\xc9\x94";
    }
    if (cl == U'ú') {
      return "u";
    }
    if (cl == U'ã') {
      return "\xc9\x90\xcc\x83";
    }
    if (cl == U'õ') {
      return "o\xcc\x83";
    }
    if (cl == U'a') {
      return stressed_syllable ? "a" : unstressed_vowel(U'a');
    }
    if (cl == U'e') {
      if (stressed_syllable && syllable_has_u32(s, U'ê')) {
        return "\xc9\x9b";
      }
      return stressed_syllable ? "e" : unstressed_vowel(U'e');
    }
    if (cl == U'i') {
      return unstressed_vowel(U'i');
    }
    if (cl == U'o') {
      if (stressed_syllable && syllable_has_u32(s, U'ô')) {
        return "\xc9\x94";
      }
      return stressed_syllable ? "o" : unstressed_vowel(U'o');
    }
    if (cl == U'u') {
      return unstressed_vowel(U'u');
    }
    if (cl == U'ü') {
      return "w";
    }
    if (cl == U'ý' || cl == U'y') {
      return "i";
    }
    return "";
  };

  const size_t n = s.size();
  size_t i = 0;
  std::string out;
  while (i < n) {
    if (s[i] == U'-') {
      ++i;
      continue;
    }
    const size_t gi = span_start + i;

    if (s[i] == U'ã' && i + 1 < n && s[i + 1] == U'o') {
      out += "\xc9\x90\xcc\x83\x77\xcc\x83";
      i += 2;
      continue;
    }
    if (s[i] == U'ã' && i + 1 < n && s[i + 1] == U'e') {
      out += "\xc9\x90\xcc\x83\x6a\xcc\x83";
      i += 2;
      continue;
    }

    if (i + 1 < n && s[i] == U'c' && s[i + 1] == U'h') {
      out += "\xca\x83";
      i += 2;
      continue;
    }
    if (i + 1 < n && s[i] == U'n' && s[i + 1] == U'h') {
      out += "\xc9\xb2";
      i += 2;
      continue;
    }
    if (i + 1 < n && s[i] == U'l' && s[i + 1] == U'h') {
      out += "\xca\x8e";
      i += 2;
      continue;
    }
    if (i + 1 < n && s[i] == U'r' && s[i + 1] == U'r') {
      out += "\xca\x81";
      i += 2;
      continue;
    }

    if (i + 1 < n && s[i] == U'q' && s[i + 1] == U'u' && i + 2 < n) {
      const char32_t t = pt_tolower(s[i + 2]);
      if (t == U'e' || t == U'é' || t == U'ê' || t == U'i' || t == U'í') {
        out += "k";
        i += 2;
        continue;
      }
    }
    if (i + 1 < n && s[i] == U'g' && s[i + 1] == U'u' && i + 2 < n) {
      const char32_t t = pt_tolower(s[i + 2]);
      if (t == U'e' || t == U'é' || t == U'ê' || t == U'i' || t == U'í') {
        out += "\xc9\xa1";
        i += 2;
        continue;
      }
    }
    if (i + 1 < n && s[i] == U'q' && s[i + 1] == U'u') {
      out += "kw";
      i += 2;
      continue;
    }

    if (i + 1 < n && s[i] == U's' && s[i + 1] == U's') {
      out += "s";
      i += 2;
      continue;
    }

    if (s[i] == U'ç') {
      out += "s";
      ++i;
      continue;
    }

    if (s[i] == U'c' && i > 0 && s[i - 1] == U's' && i + 1 < n) {
      const char32_t v = pt_tolower(s[i + 1]);
      if (v == U'a' || v == U'á' || v == U'â' || v == U'e' || v == U'é' || v == U'ê' || v == U'i' || v == U'í' ||
          v == U'o' || v == U'ó' || v == U'ô' || v == U'u' || v == U'ú' || v == U'ã' || v == U'õ') {
        if (v == U'e' || v == U'é' || v == U'ê' || v == U'i' || v == U'í') {
          out += "\xca\x83";
        } else {
          out += "sk";
        }
        ++i;
        continue;
      }
    }

    if (s[i] == U'c' && i + 1 < n) {
      const char32_t v = pt_tolower(s[i + 1]);
      if (v == U'e' || v == U'é' || v == U'ê' || v == U'i' || v == U'í') {
        out += "s";
        ++i;
        continue;
      }
    }
    if (s[i] == U'c') {
      out += "k";
      ++i;
      continue;
    }

    if (s[i] == U'g' && i + 1 < n) {
      const char32_t v = pt_tolower(s[i + 1]);
      if (v == U'e' || v == U'é' || v == U'ê' || v == U'i' || v == U'í') {
        out += "\xca\x92";
        ++i;
        continue;
      }
    }
    if (s[i] == U'g') {
      out += "\xc9\xa1";
      ++i;
      continue;
    }

    if (s[i] == U'x') {
      if (gi == 0 && i + 1 < n) {
        const char32_t t = pt_tolower(s[i + 1]);
        if (t == U'e' || t == U'é' || t == U'i' || t == U'í') {
          out += "\xca\x92";
          i += 2;
          continue;
        }
      }
      const bool pv = prev_global_vowel_u32(full_word, gi);
      // Python: ``nv`` for ⟨x⟩ does *not* gate on ``i + 1 < n`` (unlike ⟨s⟩).
      const bool nv = next_global_vowel_u32(full_word, gi + 1);
      if (pv && nv) {
        out += (is_pt_pt ? "\xca\x83" : "\xca\x92");
      } else {
        out += "ks";
      }
      ++i;
      continue;
    }

    if (s[i] == U'h') {
      ++i;
      continue;
    }

    if (s[i] == U's') {
      const bool pv = (gi > 0) && prev_global_vowel_u32(full_word, gi - 1);
      const bool nv = (i + 1 < n) && next_global_vowel_u32(full_word, gi + 1);
      if (pv && nv) {
        out += (is_pt_pt ? "\xca\x92" : "z");
      } else {
        out += "s";
      }
      ++i;
      continue;
    }

    if (s[i] == U'z') {
      out += "z";
      ++i;
      continue;
    }

    if (s[i] == U'j') {
      out += "\xca\x92";
      ++i;
      continue;
    }

    if (s[i] == U'w' || s[i] == U'W') {
      out += "w";
      ++i;
      continue;
    }

    if (s[i] == U'r') {
      const size_t gidx = span_start + i;
      const bool at_word = (gidx == 0);
      char32_t prev_ch = 0;
      if (gidx > 0) {
        prev_ch = full_word[gidx - 1];
      }
      const bool after_cons =
          gidx > 0 && !is_vowel_pt_u32(prev_ch) && prev_ch != U'\'';
      const bool intervocal = gidx > 0 && is_vowel_pt_u32(prev_ch) && gidx + 1 < full_word.size() &&
                              is_vowel_pt_u32(full_word[gidx + 1]);
      if (at_word || after_cons || (i + 1 < n && s[i + 1] == U'r')) {
        out += "\xca\x81";
      } else if (intervocal) {
        out += "\xc9\xbe";
      } else {
        out += "\xc9\xbe";
      }
      ++i;
      continue;
    }

    const char32_t ch = s[i];
    if (is_vowel_pt_u32(ch)) {
      if (i + 1 < n && is_vowel_pt_u32(s[i + 1]) && !should_hiatus_pt_u32(ch, s[i + 1])) {
        const char32_t a = pt_tolower(ch);
        const char32_t b = pt_tolower(s[i + 1]);
        if ((a == U'a' || a == U'à' || a == U'á' || a == U'â') && (b == U'i' || b == U'í')) {
          out += "aj";
          i += 2;
          continue;
        }
        if ((a == U'a' || a == U'à' || a == U'á' || a == U'â') && (b == U'u' || b == U'ú')) {
          out += "aw";
          i += 2;
          continue;
        }
        if ((a == U'e' || a == U'é' || a == U'ê') && (b == U'i' || b == U'í')) {
          out += "ej";
          i += 2;
          continue;
        }
        if ((a == U'o' || a == U'ó' || a == U'ô') && (b == U'i' || b == U'í')) {
          out += "oj";
          i += 2;
          continue;
        }
        if ((a == U'e' || a == U'é' || a == U'ê') && (b == U'u' || b == U'ú')) {
          out += "ew";
          i += 2;
          continue;
        }
        if ((a == U'o' || a == U'ó' || a == U'ô') && (b == U'u' || b == U'ú')) {
          out += "ow";
          i += 2;
          continue;
        }
      }
      const std::string seg = map_vowel_char(ch);
      if (!seg.empty()) {
        out += seg;
      }
      ++i;
      continue;
    }

    const char32_t cl = pt_tolower(ch);
    if (cl == U'b') {
      out += "b";
    } else if (cl == U'd') {
      out += "d";
    } else if (cl == U'f') {
      out += "f";
    } else if (cl == U'l') {
      out += "l";
    } else if (cl == U'm') {
      out += "m";
    } else if (cl == U'n') {
      out += "n";
    } else if (cl == U'p') {
      out += "p";
    } else if (cl == U't') {
      out += "t";
    } else if (cl == U'v') {
      out += "v";
    } else if (cl == U'k') {
      out += "k";
    }
    ++i;
  }
  return out;
}

std::string rules_word_to_ipa_single_u32(const std::u32string& wl, bool is_pt_pt, bool with_stress) {
  const auto syls = port_orthographic_syllables_u32(wl);
  if (syls.empty()) {
    return "";
  }
  const int stress_idx =
      with_stress ? static_cast<int>(default_stressed_syllable_index_u32(syls, wl)) : -1;
  size_t offset = 0;
  std::string parts;
  for (size_t idx = 0; idx < syls.size(); ++idx) {
    std::string chunk = letters_to_ipa_no_stress_u32(syls[idx], is_pt_pt, wl, offset,
                                                     with_stress && static_cast<int>(idx) == stress_idx);
    if (with_stress && static_cast<int>(idx) == stress_idx && !chunk.empty()) {
      chunk = insert_primary_stress_before_vowel_utf8(std::move(chunk));
    }
    parts += chunk;
    offset += syls[idx].size();
  }
  return parts;
}

static const std::unordered_map<std::string, std::string> kXExc = {
    {"\x74\xc3\xa1\x78\x69", "\xcb\x88\x74\x61\x6b\x73\x69"},
    {"\x74\x61\x78\x69", "\xcb\x88\x74\x61\x6b\x73\x69"},
    {"\x6d\xc3\xa1\x78\x69\x6d\x6f", "\xcb\x88\x6d\x61\x6b\x73\x69\x6d\x75"},
    {"\x66\xc3\xaa\x6e\x69\x78", "\xcb\x88\x66\xc9\x9b\x6e\x69\x6b\x73"},
    {"\x66\xc3\xa9\x6e\x69\x78", "\xcb\x88\x66\xc9\x9b\x6e\x69\x6b\x73"},
};

const std::unordered_map<std::string, std::string>& sc_straddle_map_impl() {
  static const std::unordered_map<std::string, std::string> kScStraddle = {

    {"escola", "\xc9\xaa\x73\x6b\xcb\x88\xc9\x94\x6c\xc9\x90"},
    {"piscina", "\x70\x69\xca\x83\xcb\x88\x6b\x69\x6e\xc9\x90"},
    {"descer", "\x64\xc9\xaa\xca\x83\xcb\x88\x73\x65\xc9\xbe"},
  };
  return kScStraddle;
}


static const std::unordered_set<std::string> kPtFinalSExclude = {
    "\x61\x6e\xc3\xad\x73",     "\x62\xc3\xb4\x6e\x75\x73", "\x63\x61\x69\x73",         "\x63\x61\x6f\x73",
    "\x63\x6f\x73\x6d\x6f\x73", "\x66\x72\x61\x6e\x63\xc3\xaa\x73", "\x66\xc3\xa9\x6e\x69\x78",
    "\x69\x6e\x67\x6c\xc3\xaa\x73", "\x6c\xc3\xa1\x70\x69\x73", "\x6d\xc3\xaa\x73", "\x70\x61\xc3\xad\x73",
    "\x70\x6f\x72\x74\x75\x67\x75\xc3\xaa\x73", "\x74\x72\xc3\xaa\x73", "\x74\xc3\xb3\x72\x61\x78",
    "\x76\xc3\xad\x72\x75\x73",
};

bool vowel_grapheme_tail_pt(char32_t c) {
  c = pt_tolower(c);
  return c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u' || c == U'à' || c == U'á' || c == U'â' ||
         c == U'ã' || c == U'é' || c == U'ê' || c == U'í' || c == U'ó' || c == U'ô' || c == U'õ' || c == U'ú' ||
         c == U'ü';
}

std::string pt_pt_apply_rules_final_s_to_esh(std::string ipa, const std::string& letters_key) {
  if (ipa.empty()) {
    return ipa;
  }
  const std::u32string lku = utf8_to_u32_pt(letters_key);
  if (lku.size() < 4) {
    return ipa;
  }
  if (lku.back() != U's') {
    return ipa;
  }
  if (lku.size() >= 2 && lku[lku.size() - 2] == U's') {
    return ipa;
  }
  if (kPtFinalSExclude.count(letters_key) != 0) {
    return ipa;
  }
  auto ends_u32 = [](const std::u32string& s, const std::u32string& suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  };
  static const std::u32string kEs{U'\u00EA', U's'};
  static const std::u32string kAs{U'\u00E1', U's'};
  static const std::u32string kIs{U'\u00ED', U's'};
  static const std::u32string kUs{U'\u00FA', U's'};
  if (ends_u32(lku, kEs) || ends_u32(lku, kAs) || ends_u32(lku, kIs) || ends_u32(lku, kUs)) {
    return ipa;
  }
  if (!ends_u32(lku, U"as") && !ends_u32(lku, U"os") && !ends_u32(lku, U"es")) {
    return ipa;
  }
  const char32_t penult = lku[lku.size() - 2];
  if (!vowel_grapheme_tail_pt(penult)) {
    return ipa;
  }
  if (ipa.empty() || ipa.back() != 's') {
    return ipa;
  }
  ipa.pop_back();
  ipa += "\xca\x83";
  return ipa;
}

const std::unordered_map<std::string, std::string>& fw_br_map_impl() {
  static const std::unordered_map<std::string, std::string> kFwBr = {

    {"a", "\xc9\x90"},      {"o", "u"},       {"os", "\xca\x8a\x73"},   {"as", "\xc9\x90\x73"}, {"e", "i"},
    {"ou", "ow"},           {"em", "\xc9\x90\xcc\x83\x6a\xcc\x83"}, {"no", "n\xca\x8a"},     {"na", "n\xc9\x90"},
    {"nos", "n\xca\x8a\x73"}, {"nas", "n\xc9\x90\x73"}, {"de", "d\xca\x92\xc9\xaa"}, {"do", "d\xca\x8a"},
    {"da", "d\xc9\x90"},    {"dos", "d\xca\x8a\x73"}, {"das", "d\xc9\x90\x73"}, {"dum", "d\xc5\xa9"},
    {"duma", "\xcb\x88\x64\x75\x6d\xc9\x90"}, {"num", "n\xc5\xa9"}, {"numa", "\xcb\x88\x6e\x75\x6d\xc9\x90"},
    {"pelo", "\xcb\x88\x70\xc9\x9b\x6c\xca\x8a"}, {"pela", "\xcb\x88\x70\xc9\x9b\x6c\xc9\x90"},
    {"pelos", "\xcb\x88\x70\xc9\x9b\x6c\xca\x8a\x73"}, {"pelas", "\xcb\x88\x70\xc9\x9b\x6c\xc9\x90\x73"},
    {"com", "k\xc3\xb5"},   {"sem", "s\xc9\x90\xcc\x83\x6a\xcc\x83"}, {"por", "p\x6f\xc9\xbe"},
    {"para", "\xcb\x88\x70\x61\xc9\xbe\xc9\x90"}, {"que", "k\x69"}, {"n\xc3\xa3o", "\xcb\x88\x6e\xc9\x90\xcc\x83\x77\xcc\x83"},
    {"um", "\xc5\xa9"},     {"uma", "\xcb\x88\x75\x6d\xc9\x90"}, {"uns", "\xc5\xa9\x73"}, {"umas", "\xcb\x88\x75\x6d\xc9\x90\x73"},
    {"ao", "aw"},           {"aos", "aw\xca\x83"}, {"\xc3\xa0", "a"}, {"\xc3\xa0s", "\xc9\x90\xca\x83"},
  };
  return kFwBr;
}


const std::unordered_map<std::string, std::string>& fw_pt_map_impl() {
  static const std::unordered_map<std::string, std::string> kFwPt = {

    {"a", "\xc9\x90"}, {"o", "u"}, {"os", "u\xca\x83"}, {"as", "\xc9\x90\xca\x83"}, {"e", "\xc9\xa8"},
    {"ou", "ow"},     {"em", "\xc9\x90\xcc\x83\x6a\xcc\x83"}, {"no", "nu"}, {"na", "n\xc9\x90"},
    {"nos", "nu\xca\x83"}, {"nas", "n\xc9\x90\xca\x83"}, {"de", "d\xc9\xa8"}, {"do", "du"},
    {"da", "d\xc9\x90"}, {"dos", "du\xca\x83"}, {"das", "d\xc9\x90\xca\x83"}, {"dum", "d\xc5\xa9"},
    {"duma", "\xcb\x88\x64\x75\x6d\xc9\x90"}, {"num", "n\xc5\xa9"}, {"numa", "\xcb\x88\x6e\x75\x6d\xc9\x90"},
    {"pelo", "\xcb\x88\x70\xc9\x9b\x6c\x75"}, {"pela", "\xcb\x88\x70\xc9\x9b\x6c\xc9\x90"},
    {"pelos", "\xcb\x88\x70\xc9\x9b\x6c\x75\xca\x83"}, {"pelas", "\xcb\x88\x70\xc9\x9b\x6c\xc9\x90\xca\x83"},
    {"com", "k\xc3\xb5"}, {"sem", "s\xc9\x90\xcc\x83\x6a\xcc\x83"}, {"por", "p\x75\xc9\xbe"},
    {"para", "\xcb\x88\x70\xc9\x90\xc9\xbe\xc9\x90"}, {"que", "k\xc9\xa8"}, {"n\xc3\xa3o", "\xcb\x88\x6e\xc9\x90\xcc\x83\x77\xcc\x83"},
    {"um", "\xc5\xa9"}, {"uma", "\xcb\x88\x75\x6d\xc9\x90"}, {"uns", "\xc5\xa9\xca\x83"}, {"umas", "\xcb\x88\x75\x6d\xc9\x90\xca\x83"},
    {"ao", "aw"}, {"aos", "aw\xca\x83"}, {"\xc3\xa0", "a"}, {"\xc3\xa0s", "a\xca\x83"},
  };
  return kFwPt;
}


std::string rules_word_to_ipa_utf8(const std::string& raw, bool is_pt_pt, bool with_stress) {
  const std::u32string wl = filter_pt_word_graphemes_utf8(raw);
  if (wl.empty()) {
    return "";
  }
  const std::string wkey = normalize_lookup_key_utf8(u32_to_utf8_pt(wl));
  const auto xi = kXExc.find(wkey);
  if (xi != kXExc.end()) {
    if (!with_stress) {
      return strip_stress_chars(xi->second);
    }
    return xi->second;
  }
  const auto si = sc_straddle_map_impl().find(wkey);
  if (si != sc_straddle_map_impl().end()) {
    if (!with_stress) {
      return strip_stress_chars(si->second);
    }
    return si->second;
  }
  if (wl.find(U'-') != std::u32string::npos) {
    std::vector<std::u32string> chunks;
    size_t a = 0;
    while (a < wl.size()) {
      size_t d = wl.find(U'-', a);
      const size_t end = d == std::u32string::npos ? wl.size() : d;
      if (end > a) {
        chunks.push_back(wl.substr(a, end - a));
      }
      if (d == std::u32string::npos) {
        break;
      }
      a = d + 1;
    }
    if (chunks.size() > 1) {
      std::string joined;
      for (size_t c = 0; c < chunks.size(); ++c) {
        if (c > 0) {
          joined.push_back('-');
        }
        joined += rules_word_to_ipa_single_u32(chunks[c], is_pt_pt, with_stress);
      }
      return joined;
    }
  }
  return rules_word_to_ipa_single_u32(wl, is_pt_pt, with_stress);
}

const std::unordered_map<std::string, std::string>& fw_pt() {
  return fw_pt_map_impl();
}

const std::unordered_map<std::string, std::string>& fw_br() {
  return fw_br_map_impl();
}

const std::unordered_map<std::string, std::string>& sc_straddle() {
  return sc_straddle_map_impl();
}

}  // namespace moonshine_tts::portuguese_rules
