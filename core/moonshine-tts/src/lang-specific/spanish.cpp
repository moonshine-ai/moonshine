#include "g2p-word-log.h"
#include "spanish.h"
#include "spanish-unicode.h"
#include "utf8-utils.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <regex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace moonshine_tts {
namespace {

using moonshine_tts::utf8_append_codepoint;
using moonshine_tts::utf8_decode_at;

bool is_vowel_ch(char32_t ch) {
  const char32_t cl = ch == U'Ü' ? U'ü' : ch;
  switch (cl) {
  case U'a':
  case U'e':
  case U'i':
  case U'o':
  case U'u':
  case U'á':
  case U'é':
  case U'í':
  case U'ó':
  case U'ú':
  case U'ü':
    return true;
  default:
    return false;
  }
}

bool should_hiatus(char32_t a, char32_t b) {
  const char32_t al = a;
  const char32_t bl = b;
  if (al == U'í' && bl == U'o') {
    return true;
  }
  if (al == U'i' && bl == U'ó') {
    return false;
  }
  if (al == U'í' || al == U'ú' || bl == U'í' || bl == U'ú') {
    return true;
  }
  auto strip_one = [](char32_t c) -> char32_t {
    const char *r = spanish_unicode::strip_replacement_utf8(c);
    if (r == nullptr || *r == '\0') {
      return c;
    }
    char32_t o = 0;
    size_t adv = 0;
    std::string tmp(r);
    if (!tmp.empty() && utf8_decode_at(tmp, 0, o, adv) && adv == tmp.size()) {
      return o;
    }
    return c;
  };
  const char32_t ba = strip_one(al);
  const char32_t bb = strip_one(bl);
  if (ba == bb) {
    return true;
  }
  const bool sa = (ba == U'a' || ba == U'e' || ba == U'o');
  const bool sb = (bb == U'a' || bb == U'e' || bb == U'o');
  if (sa && sb) {
    if (al == U'á' || al == U'é' || al == U'ó' || bl == U'á' || bl == U'é' || bl == U'ó') {
      return true;
    }
    const std::u32string ae = {ba, bb};
    if (ae == U"ae" || ae == U"ea") {
      return false;
    }
    return true;
  }
  return false;
}

std::vector<std::pair<size_t, size_t>> vowel_nucleus_spans(const std::u32string &w) {
  std::vector<std::pair<size_t, size_t>> out;
  const size_t n = w.size();
  size_t i = 0;
  while (i < n) {
    const char32_t ch = w[i];
    if (ch == U'y') {
      if (w == U"y") {
        out.emplace_back(i, i + 1);
        ++i;
        continue;
      }
      if (i == 0 && i + 1 < n && is_vowel_ch(w[i + 1])) {
        ++i;
        continue;
      }
      if (i > 0 && is_vowel_ch(w[i - 1]) && i + 1 < n && is_vowel_ch(w[i + 1])) {
        ++i;
        continue;
      }
      if (i > 0 && is_vowel_ch(w[i - 1]) && i + 1 >= n) {
        out.emplace_back(i, i + 1);
        ++i;
        continue;
      }
      if (i > 0 && !is_vowel_ch(w[i - 1]) && (i + 1 >= n || !is_vowel_ch(w[i + 1]))) {
        out.emplace_back(i, i + 1);
        ++i;
        continue;
      }
      ++i;
      continue;
    }
    if (!is_vowel_ch(ch)) {
      ++i;
      continue;
    }
    if (i + 1 < n && is_vowel_ch(w[i + 1])) {
      if (should_hiatus(ch, w[i + 1])) {
        out.emplace_back(i, i + 1);
        ++i;
      } else {
        out.emplace_back(i, i + 2);
        i += 2;
      }
    } else {
      out.emplace_back(i, i + 1);
      ++i;
    }
  }
  return out;
}

bool is_valid_onset2(char32_t a, char32_t b) {
  static const std::u32string k[] = {U"bl", U"br", U"cl", U"cr", U"dr", U"fl", U"fr", U"gl", U"gr",
                                     U"pl", U"pr", U"tr", U"ch"};
  for (const auto &p : k) {
    if (p.size() == 2 && p[0] == a && p[1] == b) {
      return true;
    }
  }
  return false;
}

std::pair<std::u32string, std::u32string> split_intervocalic_cluster_u32(const std::u32string &cluster) {
  if (cluster.empty()) {
    return {{}, {}};
  }
  if (cluster.size() >= 2 && cluster[cluster.size() - 2] == U'r' && cluster[cluster.size() - 1] == U'r') {
    return {{}, cluster.substr(cluster.size() - 2)};
  }
  if (cluster.size() >= 2) {
    const char32_t a = cluster[cluster.size() - 2];
    const char32_t b = cluster[cluster.size() - 1];
    if (is_valid_onset2(a, b)) {
      return {cluster.substr(0, cluster.size() - 2), cluster.substr(cluster.size() - 2)};
    }
  }
  return {cluster.substr(0, cluster.size() - 1), cluster.substr(cluster.size() - 1)};
}

std::u32string clean_syllable_word(const std::u32string &word_lower) {
  std::u32string o;
  for (char32_t c : word_lower) {
    if ((c >= U'a' && c <= U'z') || c == U'á' || c == U'é' || c == U'í' || c == U'ó' || c == U'ú' ||
        c == U'ü' || c == U'ñ') {
      o.push_back(c);
    }
  }
  return o;
}

std::vector<std::string> orthographic_syllables_utf8(const std::string &word_lower_utf8) {
  const std::u32string w = clean_syllable_word(spanish_unicode::utf8_to_utf32(word_lower_utf8));
  if (w.empty()) {
    return {};
  }
  const auto spans = vowel_nucleus_spans(w);
  if (spans.empty()) {
    return {spanish_unicode::utf32_to_utf8(w)};
  }
  std::vector<std::string> syllables;
  const size_t first_s = spans[0].first;
  std::u32string cur = w.substr(0, first_s);
  for (size_t idx = 0; idx < spans.size(); ++idx) {
    const auto s = spans[idx].first;
    const auto e = spans[idx].second;
    cur += w.substr(s, e - s);
    if (idx + 1 < spans.size()) {
      const size_t next_s = spans[idx + 1].first;
      const std::u32string cluster = w.substr(e, next_s - e);
      const auto pr = split_intervocalic_cluster_u32(cluster);
      syllables.push_back(spanish_unicode::utf32_to_utf8(cur + pr.first));
      cur = pr.second;
    } else {
      syllables.push_back(spanish_unicode::utf32_to_utf8(cur + w.substr(e)));
    }
  }
  std::vector<std::string> nonempty;
  for (const auto &s : syllables) {
    if (!s.empty()) {
      nonempty.push_back(s);
    }
  }
  return nonempty;
}

size_t default_stressed_syllable_index_v2(const std::u32string &w_clean_lower) {
  const std::string wutf = spanish_unicode::utf32_to_utf8(w_clean_lower);
  const auto syl = orthographic_syllables_utf8(wutf);
  if (syl.empty()) {
    return 0;
  }
  if (std::any_of(w_clean_lower.begin(), w_clean_lower.end(), [](char32_t c) {
        return c == U'á' || c == U'é' || c == U'í' || c == U'ó' || c == U'ú';
      })) {
    for (size_t i = 0; i < syl.size(); ++i) {
      const std::u32string su = spanish_unicode::utf8_to_utf32(syl[i]);
      if (std::any_of(su.begin(), su.end(), [](char32_t c) {
            return c == U'á' || c == U'é' || c == U'í' || c == U'ó' || c == U'ú';
          })) {
        return i;
      }
    }
  }
  const size_t n = syl.size();
  if (n == 1) {
    return 0;
  }
  const std::string last_stripped =
      spanish_unicode::strip_accents_utf8(spanish_unicode::utf32_to_utf8(std::u32string(1, w_clean_lower.back())));
  char32_t last = U' ';
  if (!last_stripped.empty()) {
    last = spanish_unicode::utf8_to_utf32(last_stripped).front();
  }
  const bool ends_vowel = (last == U'a' || last == U'e' || last == U'i' || last == U'o' || last == U'u');
  if (ends_vowel || w_clean_lower.back() == U'n' || w_clean_lower.back() == U's') {
    return n >= 2 ? n - 2 : 0;
  }
  return n - 1;
}

struct XExc {
  const char *key;
  const char *ipa;
};

// UTF-8 IPA strings (must match spanish_rule_g2p._X_EXCEPTIONS)
static const XExc k_x_exceptions[] = {
    {"mexico", "\xcb\x88" "mexiko"},
    {"mejico", "\xcb\x88" "mexiko"},
    {"oaxaca", "wa" "\xcb\x88" "xaka"},
    {"texas", "\xcb\x88" "tekas"},
    {"ximena", "xi" "\xcb\x88" "mena"},
    {"xavier", "xa" "\xcb\x88" "bje" "\xc9\xbe"},
};

const char *lookup_x_exception(const std::string &wkey) {
  for (const auto &e : k_x_exceptions) {
    if (wkey == e.key) {
      return e.ipa;
    }
  }
  return nullptr;
}

std::string apply_nasal_assimilation(std::string s, const SpanishDialect &dialect) {
  if (!dialect.nasal_assimilation) {
    return s;
  }
  const std::u32string u = spanish_unicode::utf8_to_utf32(s);
  std::u32string o;
  o.reserve(u.size());
  for (size_t i = 0; i < u.size(); ++i) {
    if (u[i] == U'n' && i + 1 < u.size()) {
      const char32_t nxt = u[i + 1];
      if (nxt == U'k' || nxt == U'\u0261') {
        o.push_back(U'\u014b');
        continue;
      }
      if (nxt == U'p' || nxt == U'b' || nxt == U'm') {
        o.push_back(U'm');
        continue;
      }
      if (nxt == U'f') {
        o.push_back(U'\u0271');
        continue;
      }
    }
    o.push_back(u[i]);
  }
  return spanish_unicode::utf32_to_utf8(o);
}

std::string insert_primary_stress_before_vowel(const std::string &ipa) {
  std::u32string u = spanish_unicode::utf8_to_utf32(ipa);
  std::u32string no;
  for (char32_t c : u) {
    if (c != U'\u02c8') {
      no.push_back(c);
    }
  }
  for (size_t i = 0; i < no.size(); ++i) {
    const char32_t ch = no[i];
    if (ch == U'a' || ch == U'e' || ch == U'i' || ch == U'o' || ch == U'u') {
      std::u32string out;
      out.insert(out.end(), no.begin(), no.begin() + static_cast<std::ptrdiff_t>(i));
      out.push_back(U'\u02c8');
      out.insert(out.end(), no.begin() + static_cast<std::ptrdiff_t>(i), no.end());
      return spanish_unicode::utf32_to_utf8(out);
    }
  }
  return std::string("\xcb\x88") + spanish_unicode::utf32_to_utf8(no);
}

size_t count_primary_stress_utf8(const std::string &ipa) {
  size_t n = 0;
  size_t i = 0;
  while (i < ipa.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(ipa, i, cp, adv);
    if (cp == U'\u02c8') {
      ++n;
    }
    i += adv;
  }
  return n;
}

bool ipa_stress_at_start(const std::string &ipa) {
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(ipa, 0, cp, adv)) {
    return false;
  }
  return cp == U'\u02c8';
}

std::string apply_narrow_intervocalic_obstruents(std::string ipa) {
  for (;;) {
    std::u32string u = spanish_unicode::utf8_to_utf32(ipa);
    bool changed = false;
    auto is_v = [](char32_t c) {
      return c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u';
    };
    for (size_t i = 0; i < u.size(); ++i) {
      if (i == 0 || i + 1 >= u.size()) {
        continue;
      }
      if (!is_v(u[i - 1]) || !is_v(u[i + 1])) {
        continue;
      }
      if (u[i] == U'b') {
        u[i] = U'\u03b2';
        changed = true;
      } else if (u[i] == U'd') {
        u[i] = U'\u00f0';
        changed = true;
      } else if (u[i] == U'\u0261') {
        u[i] = U'\u0263';
        changed = true;
      }
    }
    std::string s2 = spanish_unicode::utf32_to_utf8(u);
    if (s2 == ipa) {
      break;
    }
    ipa = std::move(s2);
    if (!changed) {
      break;
    }
  }
  return ipa;
}

std::string apply_coda_s_weakening(std::string ipa, SpanishDialect::CodaS mode) {
  if (mode == SpanishDialect::CodaS::Keep || ipa.empty()) {
    return ipa;
  }
  if (!ipa.ends_with("s") || ipa.ends_with("ks")) {
    return ipa;
  }
  ipa.pop_back();
  if (mode == SpanishDialect::CodaS::H) {
    ipa.push_back('h');
  }
  return ipa;
}

std::string postprocess_lexical_ipa(std::string ipa, const SpanishDialect &dialect, bool with_stress) {
  if (!with_stress) {
    std::string s;
    size_t i = 0;
    while (i < ipa.size()) {
      char32_t cp = 0;
      size_t adv = 0;
      utf8_decode_at(ipa, i, cp, adv);
      if (cp != U'\u02c8') {
        s.append(ipa, i, adv);
      }
      i += adv;
    }
    ipa = std::move(s);
  } else if (count_primary_stress_utf8(ipa) == 0 || ipa_stress_at_start(ipa)) {
    ipa = insert_primary_stress_before_vowel(ipa);
  }
  if (dialect.narrow_intervocalic_obstruents) {
    ipa = apply_narrow_intervocalic_obstruents(ipa);
  }
  ipa = apply_coda_s_weakening(std::move(ipa), dialect.coda_s_mode);
  return ipa;
}

bool prev_phoneme_was_vowel(const std::vector<std::string> &out) {
  if (out.empty()) {
    return false;
  }
  const std::string &last = out.back();
  return last.find('a') != std::string::npos || last.find('e') != std::string::npos ||
         last.find('i') != std::string::npos || last.find('o') != std::string::npos ||
         last.find('u') != std::string::npos;
}

bool y_is_consonant(const std::u32string &letters_lower, size_t i) {
  if (i >= letters_lower.size()) {
    return false;
  }
  const char32_t ch = letters_lower[i];
  if (ch != U'y') {
    return false;
  }
  const bool prev_v = i > 0 && is_vowel_ch(letters_lower[i - 1]);
  const bool next_v = i + 1 < letters_lower.size() && is_vowel_ch(letters_lower[i + 1]);
  if (prev_v && next_v) {
    return true;
  }
  if (i == 0 && next_v) {
    return true;
  }
  if (!prev_v && !next_v && i == letters_lower.size() - 1) {
    return false;
  }
  if (!prev_v && next_v) {
    return true;
  }
  return false;
}

char32_t to_lower_cp(char32_t c) {
  std::string tmp;
  utf8_append_codepoint(tmp, c);
  const std::string lo = spanish_unicode::unicode_lower_utf8(tmp);
  const std::u32string u = spanish_unicode::utf8_to_utf32(lo);
  return u.empty() ? c : u.front();
}

std::string letters_to_ipa_no_stress(const std::u32string &syl_lower, const SpanishDialect &dialect,
                                     size_t grapheme_offset) {
  const std::u32string &lw = syl_lower;
  size_t i = 0;
  const size_t n = lw.size();
  std::vector<std::string> out;

  auto peek_vowel = [&](size_t j) -> bool {
    size_t k = j;
    while (k < n) {
      if (lw[k] == U'h') {
        ++k;
        continue;
      }
      return is_vowel_ch(lw[k]);
    }
    return false;
  };

  while (i < n) {
    const char32_t ch = lw[i];

    if (ch == U'h') {
      ++i;
      continue;
    }

    if (ch == U'y') {
      if (lw == U"y") {
        out.emplace_back("i");
        ++i;
        continue;
      }
      if (y_is_consonant(lw, i)) {
        out.push_back(dialect.y_consonant_ipa);
        ++i;
        continue;
      }
      out.emplace_back("i");
      ++i;
      continue;
    }

    if (ch == U'ñ') {
      out.emplace_back("\xc9\xb2"); // ɲ UTF-8
      ++i;
      continue;
    }

    if (i + 1 < n && lw[i] == U'r' && lw[i + 1] == U'r') {
      out.push_back(dialect.trill_ipa);
      i += 2;
      continue;
    }

    if (i + 1 < n && lw[i] == U'c' && lw[i + 1] == U'h') {
      out.emplace_back("t\u0283"); // t + ʃ
      i += 2;
      continue;
    }

    if (i + 1 < n && lw[i] == U'l' && lw[i + 1] == U'l') {
      out.push_back(dialect.yeismo ? dialect.y_consonant_ipa : dialect.ll_ipa);
      i += 2;
      continue;
    }

    if (ch == U'q' && i + 2 < n && lw[i + 1] == U'u' &&
        (lw[i + 2] == U'e' || lw[i + 2] == U'i' || lw[i + 2] == U'é' || lw[i + 2] == U'í')) {
      static const std::unordered_map<char32_t, const char *> vmap = {
          {U'e', "e"}, {U'i', "i"}, {U'é', "e"}, {U'í', "i"}};
      out.emplace_back("k");
      out.emplace_back(vmap.at(lw[i + 2]));
      i += 3;
      continue;
    }

    if (ch == U'g' && i + 2 < n && lw[i + 1] == U'ü' &&
        (lw[i + 2] == U'e' || lw[i + 2] == U'i' || lw[i + 2] == U'é' || lw[i + 2] == U'í')) {
      static const std::unordered_map<char32_t, const char *> vmap = {
          {U'e', "e"}, {U'i', "i"}, {U'é', "e"}, {U'í', "i"}};
      out.emplace_back("\xc9\xa1"); // ɡ
      out.emplace_back("w");
      out.emplace_back(vmap.at(lw[i + 2]));
      i += 3;
      continue;
    }

    if (ch == U'g' && i + 2 < n && lw[i + 1] == U'u' &&
        (lw[i + 2] == U'e' || lw[i + 2] == U'i' || lw[i + 2] == U'é' || lw[i + 2] == U'í')) {
      static const std::unordered_map<char32_t, const char *> vmap = {
          {U'e', "e"}, {U'i', "i"}, {U'é', "e"}, {U'í', "i"}};
      out.emplace_back("\xc9\xa1");
      out.emplace_back(vmap.at(lw[i + 2]));
      i += 3;
      continue;
    }

    if (ch == U'g' && i + 1 < n && (lw[i + 1] == U'e' || lw[i + 1] == U'i' || lw[i + 1] == U'é' ||
                                    lw[i + 1] == U'í')) {
      out.push_back(dialect.voiceless_velar_fricative);
      ++i;
      continue;
    }

    if (ch == U'c' && i + 3 < n && lw.substr(i, 4) == U"ción") {
      out.push_back(dialect.ce_ci_z_ipa);
      out.emplace_back("jon");
      i += 4;
      continue;
    }
    if (ch == U'c' && i + 2 < n && lw.substr(i, 3) == U"ció") {
      out.push_back(dialect.ce_ci_z_ipa);
      out.emplace_back("jo");
      i += 3;
      continue;
    }

    if (ch == U'c' && i + 1 < n &&
        (lw[i + 1] == U'e' || lw[i + 1] == U'i' || lw[i + 1] == U'é' || lw[i + 1] == U'í')) {
      out.push_back(dialect.ce_ci_z_ipa);
      ++i;
      continue;
    }

    if (ch == U'z') {
      out.push_back(dialect.ce_ci_z_ipa);
      ++i;
      continue;
    }

    if (ch == U'x') {
      const size_t abs_pos = grapheme_offset + i;
      const bool next_v = peek_vowel(i + 1);
      if (abs_pos == 0 && next_v) {
        out.push_back(dialect.x_initial_before_vowel);
      } else if (prev_phoneme_was_vowel(out) && next_v) {
        out.push_back(dialect.x_intervocalic_default);
      } else {
        out.push_back(dialect.x_intervocalic_default);
      }
      ++i;
      continue;
    }

    if (ch == U'j') {
      out.push_back(dialect.voiceless_velar_fricative);
      ++i;
      continue;
    }

    if (ch == U'c') {
      out.emplace_back("k");
      ++i;
      continue;
    }

    if (ch == U'r') {
      const bool at_word_start = (i == 0);
      const bool after_lns =
          i > 0 && (lw[i - 1] == U'l' || lw[i - 1] == U'n' || lw[i - 1] == U's');
      if (at_word_start || after_lns) {
        out.push_back(dialect.trill_ipa);
      } else if (prev_phoneme_was_vowel(out) && peek_vowel(i + 1)) {
        out.push_back(dialect.tap_ipa);
      } else {
        out.push_back(dialect.tap_ipa);
      }
      ++i;
      continue;
    }

    const char *simple = nullptr;
    switch (ch) {
    case U'a':
      simple = "a";
      break;
    case U'e':
      simple = "e";
      break;
    case U'i':
      simple = "i";
      break;
    case U'o':
      simple = "o";
      break;
    case U'u':
      simple = "u";
      break;
    case U'á':
      simple = "a";
      break;
    case U'é':
      simple = "e";
      break;
    case U'í':
      simple = "i";
      break;
    case U'ó':
      simple = "o";
      break;
    case U'ú':
      simple = "u";
      break;
    case U'ü':
      simple = "w";
      break;
    case U'b':
    case U'v':
      simple = "b";
      break;
    case U'd':
      simple = "d";
      break;
    case U'f':
      simple = "f";
      break;
    case U'k':
      simple = "k";
      break;
    case U'l':
      simple = "l";
      break;
    case U'm':
      simple = "m";
      break;
    case U'n':
      simple = "n";
      break;
    case U'p':
      simple = "p";
      break;
    case U's':
      simple = "s";
      break;
    case U't':
      simple = "t";
      break;
    case U'w':
      simple = "w";
      break;
    case U'g':
      simple = "\xc9\xa1";
      break;
    default:
      break;
    }
    if (simple != nullptr) {
      out.emplace_back(simple);
      ++i;
      continue;
    }

    ++i;
  }

  std::string ipa;
  for (const auto &p : out) {
    ipa += p;
  }
  return apply_nasal_assimilation(std::move(ipa), dialect);
}

std::u32string filter_word_letters_utf32(const std::string &wraw) {
  std::u32string o;
  size_t i = 0;
  while (i < wraw.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(wraw, i, cp, adv);
    if ((cp >= U'a' && cp <= U'z') || (cp >= U'A' && cp <= U'Z') || cp == U'á' || cp == U'é' ||
        cp == U'í' || cp == U'ó' || cp == U'ú' || cp == U'ü' || cp == U'ñ' || cp == U'Á' ||
        cp == U'É' || cp == U'Í' || cp == U'Ó' || cp == U'Ú' || cp == U'Ü' || cp == U'Ñ') {
      o.push_back(cp);
    }
    i += adv;
  }
  return o;
}

SpanishDialect make_common(const std::string &id, std::string ce_ci_z_ipa, bool yeismo,
                             std::string y_cons, std::string ll_ipa, std::string vv, SpanishDialect::CodaS coda,
                             bool narrow_obs) {
  SpanishDialect d;
  d.id = id;
  d.ce_ci_z_ipa = std::move(ce_ci_z_ipa);
  d.yeismo = yeismo;
  d.y_consonant_ipa = std::move(y_cons);
  d.ll_ipa = std::move(ll_ipa);
  d.x_intervocalic_default = "ks";
  d.x_initial_before_vowel = "s";
  d.voiceless_velar_fricative = std::move(vv);
  d.trill_ipa = "r";
  d.tap_ipa = "\xc9\xbe"; // ɾ
  d.nasal_assimilation = false;
  d.narrow_intervocalic_obstruents = narrow_obs;
  d.coda_s_mode = coda;
  return d;
}

} // namespace

#include "spanish-numbers.cpp"

std::vector<std::string> spanish_dialect_cli_ids() {
  return {"es-419", "es-AR", "es-BO", "es-CL", "es-CO", "es-CU", "es-DO", "es-EC", "es-ES",
          "es-ES-distincion", "es-GT", "es-MX", "es-PE", "es-PR", "es-PY", "es-UY", "es-VE"};
}

std::vector<std::string> SpanishRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first(spanish_dialect_cli_ids());
}

SpanishDialect spanish_dialect_from_cli_id(const std::string &cli_id, bool narrow_intervocalic_obstruents) {
  size_t a = 0;
  size_t b = cli_id.size();
  while (a < b && std::isspace(static_cast<unsigned char>(cli_id[a]))) {
    ++a;
  }
  while (b > a && std::isspace(static_cast<unsigned char>(cli_id[b - 1]))) {
    --b;
  }
  const std::string key = cli_id.substr(a, b - a);
  if (key.empty()) {
    throw std::invalid_argument("empty dialect id");
  }
  if (key == "es-MX") {
    return make_common("es-MX", "s", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-419") {
    return make_common("es-419", "s", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-ES") {
    return make_common("es-ES", "\xce\xb8", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-ES-distincion") {
    return make_common("es-ES-distincion", "\xce\xb8", true, "\xca\x9d", "\xca\x8e", "x",
                       SpanishDialect::CodaS::Keep, narrow_intervocalic_obstruents);
  }
  if (key == "es-CO") {
    return make_common("es-CO", "s", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-VE") {
    return make_common("es-VE", "s", true, "\xca\x9d", "\xca\x8e", "h", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-EC") {
    return make_common("es-EC", "s", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-PE") {
    return make_common("es-PE", "s", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-CL") {
    return make_common("es-CL", "s", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::H,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-AR") {
    return make_common("es-AR", "s", true, "\xca\x92", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-UY") {
    return make_common("es-UY", "s", true, "\xca\x92", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-BO") {
    return make_common("es-BO", "s", false, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-PY") {
    return make_common("es-PY", "s", false, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-CU") {
    return make_common("es-CU", "s", true, "\xca\x9d", "\xca\x8e", "h", SpanishDialect::CodaS::H,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-DO") {
    return make_common("es-DO", "s", true, "\xca\x9d", "\xca\x8e", "h", SpanishDialect::CodaS::H,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-PR") {
    return make_common("es-PR", "s", true, "\xca\x9d", "\xca\x8e", "h", SpanishDialect::CodaS::H,
                       narrow_intervocalic_obstruents);
  }
  if (key == "es-GT") {
    return make_common("es-GT", "s", true, "\xca\x9d", "\xca\x8e", "x", SpanishDialect::CodaS::Keep,
                       narrow_intervocalic_obstruents);
  }
  throw std::invalid_argument("unknown dialect id \"" + key + "\"");
}

SpanishRuleG2p::SpanishRuleG2p(SpanishDialect dialect, bool with_stress, bool expand_cardinal_digits)
    : dialect_(std::move(dialect)),
      with_stress_(with_stress),
      expand_cardinal_digits_(expand_cardinal_digits) {}

std::string SpanishRuleG2p::word_to_ipa(const std::string &word) const {
  const std::string wraw = [&]() {
    size_t a = 0, b = word.size();
    while (a < b && std::isspace(static_cast<unsigned char>(word[a]))) {
      ++a;
    }
    while (b > a && std::isspace(static_cast<unsigned char>(word[b - 1]))) {
      --b;
    }
    return word.substr(a, b - a);
  }();
  if (wraw.empty()) {
    return "";
  }
  if (expand_cardinal_digits_ && es_ascii_all_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_spanish_words(wraw);
    if (phrase != wraw) {
      return text_to_ipa_no_expand(phrase, nullptr);
    }
    return wraw;
  }
  if (!expand_cardinal_digits_) {
    static const std::regex dig_pass(R"(^[0-9]+$)", std::regex::ECMAScript);
    if (std::regex_match(wraw, dig_pass)) {
      return wraw;
    }
  }
  const std::string wkey = spanish_unicode::word_key(wraw);
  const char *exc = lookup_x_exception(wkey);
  if (exc != nullptr && dialect_.id.starts_with("es")) {
    return postprocess_lexical_ipa(std::string(exc), dialect_, with_stress_);
  }

  const std::u32string letters = filter_word_letters_utf32(wraw);
  if (letters.empty()) {
    return "";
  }
  std::u32string lw;
  for (char32_t c : letters) {
    lw.push_back(to_lower_cp(c));
  }

  const auto syl = orthographic_syllables_utf8(spanish_unicode::utf32_to_utf8(lw));
  const size_t stress_idx =
      with_stress_ ? default_stressed_syllable_index_v2(clean_syllable_word(lw)) : static_cast<size_t>(-1);
  size_t offset = 0;
  std::vector<std::string> parts;
  for (const auto &s : syl) {
    const std::u32string su = spanish_unicode::utf8_to_utf32(s);
    parts.push_back(letters_to_ipa_no_stress(su, dialect_, offset));
    offset += su.size();
  }
  if (with_stress_ && !parts.empty() && stress_idx < parts.size()) {
    parts[stress_idx] = insert_primary_stress_before_vowel(parts[stress_idx]);
  }
  std::string ipa;
  for (const auto &p : parts) {
    ipa += p;
  }
  if (dialect_.narrow_intervocalic_obstruents) {
    ipa = apply_narrow_intervocalic_obstruents(ipa);
  }
  ipa = apply_coda_s_weakening(std::move(ipa), dialect_.coda_s_mode);
  return ipa;
}

std::string SpanishRuleG2p::text_to_ipa_no_expand(const std::string &text,
                                                  std::vector<G2pWordLog> *per_word_log) const {
  std::string out;
  size_t pos = 0;
  const size_t n = text.size();
  while (pos < n) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(text, pos, cp, adv);
    if (spanish_unicode::is_space_char(cp)) {
      out.push_back(' ');
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (!spanish_unicode::is_space_char(cp)) {
          break;
        }
        pos += adv;
      }
      continue;
    }
    if (spanish_unicode::is_word_char(cp)) {
      size_t start = pos;
      pos += adv;
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (!spanish_unicode::is_word_char(cp)) {
          break;
        }
        pos += adv;
      }
      const std::string tok = text.substr(start, pos - start);
      const std::string k = spanish_unicode::word_key(tok);
      const char *exc = lookup_x_exception(k);
      std::string wipa;
      if (exc != nullptr) {
        wipa = postprocess_lexical_ipa(std::string(exc), dialect_, with_stress_);
      } else {
        wipa = word_to_ipa(tok);
      }
      if (per_word_log != nullptr) {
        per_word_log->push_back(
            G2pWordLog{tok, k, G2pWordPath::kRuleBasedG2p, std::move(wipa)});
        out += per_word_log->back().ipa;
      } else {
        out += wipa;
      }
      continue;
    }
    size_t start = pos;
    pos += adv;
    while (pos < n) {
      utf8_decode_at(text, pos, cp, adv);
      if (spanish_unicode::is_word_char(cp) || spanish_unicode::is_space_char(cp)) {
        break;
      }
      pos += adv;
    }
    out.append(text, start, pos - start);
  }
  // Collapse spaces like Python re.sub(r" +", " ", out).strip()
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

std::string SpanishRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog> *per_word_log) {
  if (expand_cardinal_digits_) {
    text = expand_spanish_digit_tokens_in_text(std::move(text));
  }
  return text_to_ipa_no_expand(text, per_word_log);
}

std::string spanish_word_to_ipa(const std::string &word, const SpanishDialect &dialect, bool with_stress,
                                bool expand_cardinal_digits) {
  return SpanishRuleG2p(dialect, with_stress, expand_cardinal_digits).word_to_ipa(word);
}

std::string spanish_text_to_ipa(const std::string &text, const SpanishDialect &dialect, bool with_stress,
                                std::vector<G2pWordLog> *per_word_log, bool expand_cardinal_digits) {
  return SpanishRuleG2p(dialect, with_stress, expand_cardinal_digits).text_to_ipa(text, per_word_log);
}

} // namespace moonshine_tts
