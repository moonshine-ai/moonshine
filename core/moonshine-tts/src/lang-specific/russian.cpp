#include "russian.h"

#include "g2p-word-log.h"
#include "ipa-symbols.h"
#include "german.h"
#include "utf8-utils.h"

#include <cctype>
#include <algorithm>
#include <cstdint>
#include <cwctype>
#include <fstream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moonshine_tts {
namespace {

const std::string& kPri = ipa::kPrimaryStressUtf8;
const std::string& kSec = ipa::kSecondaryStressUtf8;
const std::string kPalatal{"\xCA\xB2"};  // ʲ

using moonshine_tts::erase_utf8_substr;
using moonshine_tts::trim_ascii_ws_copy;
using moonshine_tts::utf8_decode_at;

bool is_unicode_mn(char32_t cp) {
  if (cp >= 0x300 && cp <= 0x36F) {
    return true;
  }
  if (cp >= 0x1AB0 && cp <= 0x1AFF) {
    return true;
  }
  if (cp >= 0x1DC0 && cp <= 0x1DFF) {
    return true;
  }
  if (cp >= 0x20D0 && cp <= 0x20FF) {
    return true;
  }
  if (cp >= 0xFE20 && cp <= 0xFE2F) {
    return true;
  }
  return false;
}

bool is_combining_mark(char32_t cp) {
  return is_unicode_mn(cp);
}

char32_t russian_tolower_cp(char32_t c) {
  if (c == U'\u0401') {
    return U'\u0451';
  }
  if (c >= U'\u0410' && c <= U'\u042F') {
    return c + 32;
  }
  return c;
}

bool is_russian_vowel_letter(char32_t c) {
  c = russian_tolower_cp(c);
  static const std::unordered_set<char32_t> vowels{
      U'\u0430', U'\u0435', U'\u0451', U'\u0438', U'\u043E', U'\u0443', U'\u044B',
      U'\u044D', U'\u044E', U'\u044F'};
  return vowels.count(c) != 0;
}

bool is_russian_lex_key_cp(char32_t c) {
  c = russian_tolower_cp(c);
  if (c == U'-') {
    return true;
  }
  if (c >= U'\u0430' && c <= U'\u044F') {
    return true;
  }
  return c == U'\u0451';
}

/// Python ``unicodedata.normalize("NFD", c)`` for selected precomposed characters (wiki + Cyrillic).
const std::unordered_map<char32_t, std::u32string>& canonical_nfd_map() {
  static const std::unordered_map<char32_t, std::u32string> kMap = {
      {U'\u00E0', U"\u0061\u0300"}, {U'\u00E9', U"\u0065\u0301"}, {U'\u00ED', U"\u0069\u0301"},
      {U'\u0117', U"\u0065\u0307"}, {U'\u0160', U"\u0053\u030C"}, {U'\u0161', U"\u0073\u030C"},
      {U'\u016B', U"\u0075\u0304"}, {U'\u03AE', U"\u03B7\u0301"}, {U'\u03CC', U"\u03BF\u0301"},
      {U'\u0400', U"\u0415\u0300"}, {U'\u0401', U"\u0415\u0308"}, {U'\u0403', U"\u0413\u0301"},
      {U'\u0407', U"\u0406\u0308"}, {U'\u040C', U"\u041A\u0301"}, {U'\u040D', U"\u0418\u0300"},
      {U'\u040E', U"\u0423\u0306"}, {U'\u0419', U"\u0418\u0306"}, {U'\u0439', U"\u0438\u0306"},
      {U'\u0450', U"\u0435\u0300"}, {U'\u0451', U"\u0435\u0308"}, {U'\u0453', U"\u0433\u0301"},
      {U'\u0457', U"\u0456\u0308"}, {U'\u045C', U"\u043A\u0301"}, {U'\u045D', U"\u0438\u0300"},
      {U'\u045E', U"\u0443\u0306"}, {U'\u0476', U"\u0474\u030F"}, {U'\u0477', U"\u0475\u030F"},
      {U'\u04C1', U"\u0416\u0306"}, {U'\u04C2', U"\u0436\u0306"}, {U'\u04D0', U"\u0410\u0306"},
      {U'\u04D1', U"\u0430\u0306"}, {U'\u04D2', U"\u0410\u0308"}, {U'\u04D3', U"\u0430\u0308"},
      {U'\u04D6', U"\u0415\u0306"}, {U'\u04D7', U"\u0435\u0306"}, {U'\u04DA', U"\u04D8\u0308"},
      {U'\u04DB', U"\u04D9\u0308"}, {U'\u04DC', U"\u0416\u0308"}, {U'\u04DD', U"\u0436\u0308"},
      {U'\u04DE', U"\u0417\u0308"}, {U'\u04DF', U"\u0437\u0308"}, {U'\u04E2', U"\u0418\u0304"},
      {U'\u04E3', U"\u0438\u0304"}, {U'\u04E4', U"\u0418\u0308"}, {U'\u04E5', U"\u0438\u0308"},
      {U'\u04E6', U"\u041E\u0308"}, {U'\u04E7', U"\u043E\u0308"}, {U'\u04EA', U"\u04E8\u0308"},
      {U'\u04EB', U"\u04E9\u0308"}, {U'\u04EC', U"\u042D\u0308"}, {U'\u04ED', U"\u044D\u0308"},
      {U'\u04EE', U"\u0423\u0304"}, {U'\u04EF', U"\u0443\u0304"}, {U'\u04F0', U"\u0423\u0308"},
      {U'\u04F1', U"\u0443\u0308"}, {U'\u04F2', U"\u0423\u030B"}, {U'\u04F3', U"\u0443\u030B"},
      {U'\u04F4', U"\u0427\u0308"}, {U'\u04F5', U"\u0447\u0308"}, {U'\u04F8', U"\u042B\u0308"},
      {U'\u04F9', U"\u044B\u0308"},
  };
  return kMap;
}

void append_nfd_expansion(char32_t cp, std::u32string& out) {
  const auto& m = canonical_nfd_map();
  const auto it = m.find(cp);
  if (it != m.end()) {
    out.append(it->second);
    return;
  }
  out.push_back(cp);
}

std::string u32_to_utf8(const std::u32string& s) {
  std::string out;
  for (char32_t cp : s) {
    utf8_append_codepoint(out, cp);
  }
  return out;
}

char32_t unicode_tolower_like_python(char32_t cp) {
  const char32_t ru = russian_tolower_cp(cp);
  if (ru != cp) {
    return ru;
  }
  if (cp <= 0x7Fu) {
    return static_cast<char32_t>(std::tolower(static_cast<unsigned char>(cp)));
  }
  if (cp <= static_cast<char32_t>(std::numeric_limits<wint_t>::max())) {
    const wint_t w = std::towlower(static_cast<wint_t>(cp));
    if (w != static_cast<wint_t>(cp) && w != static_cast<wint_t>(WEOF)) {
      return static_cast<char32_t>(w);
    }
  }
  return cp;
}

/// Mirrors Python ``normalize_lookup_key`` (lower + NFD + Mn strip + Cyrillic key filter).
std::string normalize_lookup_key_utf8(const std::string& word) {
  const std::string trimmed = trim_ascii_ws_copy(word);
  std::u32string u32;
  size_t i = 0;
  while (i < trimmed.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(trimmed, i, cp, adv);
    u32.push_back(unicode_tolower_like_python(cp));
    i += adv;
  }
  std::u32string expanded;
  for (char32_t cp : u32) {
    append_nfd_expansion(cp, expanded);
  }
  std::u32string no_mn;
  for (char32_t cp : expanded) {
    if (!is_unicode_mn(cp)) {
      no_mn.push_back(cp);
    }
  }
  std::string out;
  for (char32_t cp : no_mn) {
    const char32_t cl = russian_tolower_cp(cp);
    if (is_russian_lex_key_cp(cl)) {
      utf8_append_codepoint(out, cl);
    }
  }
  return out;
}

std::string utf8_russian_lowercase(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    if (is_combining_mark(cp)) {
      utf8_append_codepoint(out, cp);
      i += adv;
      continue;
    }
    utf8_append_codepoint(out, russian_tolower_cp(cp));
    i += adv;
  }
  return out;
}

bool surface_is_all_lowercase_russian(const std::string& surf) {
  return surf == utf8_russian_lowercase(surf);
}

void load_russian_lexicon_file(const std::filesystem::path& path,
                               std::unordered_map<std::string, std::string>& out_lex) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Russian G2P: cannot read lexicon " + path.generic_string());
  }
  std::unordered_map<std::string, std::pair<std::string, bool>> tmp;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string surf = trim_ascii_ws_copy(line.substr(0, tab));
    std::string ipa = trim_ascii_ws_copy(line.substr(tab + 1));
    const std::string k = normalize_lookup_key_utf8(surf);
    if (k.empty()) {
      continue;
    }
    const bool lower_ok = surface_is_all_lowercase_russian(surf);
    const auto it = tmp.find(k);
    if (it == tmp.end()) {
      tmp.emplace(k, std::make_pair(std::move(ipa), lower_ok));
    } else if (lower_ok && !it->second.second) {
      it->second = std::make_pair(std::move(ipa), true);
    }
  }
  out_lex.clear();
  for (auto& e : tmp) {
    out_lex.emplace(std::move(e.first), std::move(e.second.first));
  }
}

std::string filter_russian_graphemes_keep_stress(std::string_view raw) {
  std::string out;
  const std::string s = trim_ascii_ws_copy(raw);
  size_t i = 0;
  while (i < s.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, i, cp, adv);
    if (is_combining_mark(cp)) {
      utf8_append_codepoint(out, cp);
      i += adv;
      continue;
    }
    const char32_t cl = unicode_tolower_like_python(cp);
    if (is_russian_lex_key_cp(cl)) {
      utf8_append_codepoint(out, cl);
    }
    i += adv;
  }
  return out;
}

std::string strip_grapheme_diacritics_utf8(std::string_view sv) {
  const std::string trimmed = trim_ascii_ws_copy(sv);
  std::u32string u32;
  size_t i = 0;
  while (i < trimmed.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(trimmed, i, cp, adv);
    u32.push_back(unicode_tolower_like_python(cp));
    i += adv;
  }
  std::u32string expanded;
  for (char32_t cp : u32) {
    append_nfd_expansion(cp, expanded);
  }
  std::u32string no_mn;
  for (char32_t cp : expanded) {
    if (!is_unicode_mn(cp)) {
      no_mn.push_back(cp);
    }
  }
  return u32_to_utf8(no_mn);
}

std::optional<int> acute_stressed_vowel_ordinal(const std::string& w_nfc) {
  size_t i = 0;
  int v_ord = 0;
  const size_t n = w_nfc.size();
  while (i < n) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(w_nfc, i, cp, adv);
    if (is_russian_vowel_letter(cp)) {
      size_t j = i + adv;
      while (j < n) {
        char32_t c2 = 0;
        size_t a2 = 0;
        utf8_decode_at(w_nfc, j, c2, a2);
        if (!is_combining_mark(c2)) {
          break;
        }
        if (c2 == U'\u0301') {
          return v_ord;
        }
        j += a2;
      }
      ++v_ord;
      i = j;
    } else {
      i += adv;
    }
  }
  return std::nullopt;
}

int vowel_ordinal_to_syllable(const std::vector<std::string>& syls, int nucleus_ord) {
  int v = 0;
  for (size_t si = 0; si < syls.size(); ++si) {
    size_t p = 0;
    while (p < syls[si].size()) {
      char32_t c = 0;
      size_t a = 0;
      utf8_decode_at(syls[si], p, c, a);
      if (is_russian_vowel_letter(c)) {
        if (v == nucleus_ord) {
          return static_cast<int>(si);
        }
        ++v;
      }
      p += a;
    }
  }
  return 0;
}

std::vector<std::string> russian_orthographic_syllables_utf8(const std::string& word_lower) {
  std::string w;
  size_t i = 0;
  while (i < word_lower.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word_lower, i, cp, adv);
    if (is_russian_lex_key_cp(russian_tolower_cp(cp)) && !is_combining_mark(cp)) {
      utf8_append_codepoint(w, russian_tolower_cp(cp));
    }
    i += adv;
  }
  if (w.empty()) {
    return {};
  }
  if (w.find('-') != std::string::npos) {
    std::vector<std::string> acc;
    size_t start = 0;
    while (start < w.size()) {
      size_t dash = w.find('-', start);
      const size_t end = dash == std::string::npos ? w.size() : dash;
      if (end > start) {
        const std::string chunk = w.substr(start, end - start);
        auto part = russian_orthographic_syllables_utf8(chunk);
        acc.insert(acc.end(), part.begin(), part.end());
      }
      if (dash == std::string::npos) {
        break;
      }
      start = dash + 1;
    }
    return acc;
  }
  std::vector<std::pair<size_t, size_t>> spans;
  size_t pos = 0;
  while (pos < w.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(w, pos, cp, adv);
    if (is_russian_vowel_letter(cp)) {
      spans.push_back({pos, pos + adv});
    }
    pos += adv;
  }
  if (spans.empty()) {
    return {w};
  }
  std::vector<std::string> syllables;
  const size_t first_s = spans[0].first;
  std::string cur = w.substr(0, first_s);
  for (size_t idx = 0; idx < spans.size(); ++idx) {
    const size_t s = spans[idx].first;
    const size_t e = spans[idx].second;
    cur.append(w, s, e - s);
    if (idx + 1 < spans.size()) {
      const size_t next_s = spans[idx + 1].first;
      syllables.push_back(std::move(cur));
      cur = w.substr(e, next_s - e);
    } else {
      cur.append(w, e, std::string::npos);
      syllables.push_back(std::move(cur));
    }
  }
  syllables.erase(std::remove_if(syllables.begin(), syllables.end(),
                                 [](const std::string& x) { return x.empty(); }),
                  syllables.end());
  return syllables;
}

int stress_syllable_index(const std::vector<std::string>& syls, const std::string& stress_src) {
  if (syls.empty()) {
    return 0;
  }
  for (size_t si = 0; si < syls.size(); ++si) {
    size_t p = 0;
    while (p < syls[si].size()) {
      char32_t c = 0;
      size_t a = 0;
      utf8_decode_at(syls[si], p, c, a);
      if (c == U'\u0451' || c == U'\u0401') {
        return static_cast<int>(si);
      }
      p += a;
    }
  }
  const auto ord = acute_stressed_vowel_ordinal(stress_src);
  if (ord.has_value()) {
    return vowel_ordinal_to_syllable(syls, *ord);
  }
  return 0;
}

std::vector<int> syllable_index_per_codepoint(const std::string& w) {
  const auto syls = russian_orthographic_syllables_utf8(w);
  std::vector<int> out;
  for (size_t si = 0; si < syls.size(); ++si) {
    const std::vector<std::string> cps = utf8_split_codepoints(syls[si]);
    for (size_t k = 0; k < cps.size(); ++k) {
      out.push_back(static_cast<int>(si));
    }
  }
  return out;
}

bool palatalizable_cons(char32_t ch) {
  static const std::unordered_set<char32_t> s{
      U'\u0431', U'\u0432', U'\u0433', U'\u0434', U'\u0437', U'\u043A', U'\u043B', U'\u043C',
      U'\u043D', U'\u043F', U'\u0440', U'\u0441', U'\u0442', U'\u0444', U'\u0445'};
  return s.count(ch) != 0;
}

std::string emit_consonant(char32_t ch, bool palatal) {
  switch (ch) {
  case U'\u0447':
    return "tɕ";
  case U'\u0449':
    return "ɕː";
  case U'\u0446':
    return "ts";
  case U'\u0436':
    return "ʐ";
  case U'\u0431':
    return palatal ? "b" + kPalatal : "b";
  case U'\u0432':
    return palatal ? "v" + kPalatal : "v";
  case U'\u0433':
    return palatal ? "\xC9\xA1" + kPalatal : "\xC9\xA1";
  case U'\u0434':
    return palatal ? "d" + kPalatal : "d";
  case U'\u0437':
    return palatal ? "z" + kPalatal : "z";
  case U'\u0439':
    return "j";
  case U'\u043A':
    return palatal ? "k" + kPalatal : "k";
  case U'\u043B':
    return palatal ? "l" + kPalatal : "l";
  case U'\u043C':
    return palatal ? "m" + kPalatal : "m";
  case U'\u043D':
    return palatal ? "n" + kPalatal : "n";
  case U'\u043F':
    return palatal ? "p" + kPalatal : "p";
  case U'\u0440':
    return palatal ? "r" + kPalatal : "r";
  case U'\u0441':
    return palatal ? "s" + kPalatal : "s";
  case U'\u0442':
    return palatal ? "t" + kPalatal : "t";
  case U'\u0444':
    return palatal ? "f" + kPalatal : "f";
  case U'\u0445':
    return palatal ? "x" + kPalatal : "x";
  default:
    return "";
  }
}

bool ipa_piece_ends_with_palatal(const std::string& piece) {
  if (piece.size() >= kPalatal.size() &&
      piece.compare(piece.size() - kPalatal.size(), kPalatal.size(), kPalatal) == 0) {
    return true;
  }
  return false;
}

bool ipa_piece_last_is_vowel_letter(const std::string& piece) {
  const std::vector<std::string> cps = utf8_split_codepoints(piece);
  if (cps.empty()) {
    return false;
  }
  char32_t cp = 0;
  size_t adv = 0;
  utf8_decode_at(cps.back(), 0, cp, adv);
  return cp == U'a' || cp == U'e' || cp == U'i' || cp == U'o' || cp == U'u' ||
         cp == U'\u025B' || cp == U'\u0259' || cp == U'\u0268' || cp == U'\u026A' ||
         cp == U'\u028A';
}

bool ipa_piece_after_hard_consonant(const std::string& piece) {
  if (piece.empty()) {
    return false;
  }
  if (ipa_piece_ends_with_palatal(piece)) {
    return false;
  }
  if (ipa_piece_last_is_vowel_letter(piece)) {
    return false;
  }
  return true;
}

std::string vowel_ipa(char32_t ch, bool stressed, bool after_palatal, bool after_hard,
                      bool word_initial_jot) {
  const char32_t cl = russian_tolower_cp(ch);
  if (cl == U'\u0430') {
    return stressed ? "a" : "\xC9\x99";
  }
  if (cl == U'\u043E') {
    return stressed ? "o" : "\xC9\x99";
  }
  if (cl == U'\u0443') {
    return "u";
  }
  if (cl == U'\u044B') {
    return "\xC9\xA8";
  }
  if (cl == U'\u044D') {
    return "\xC9\x9B";
  }
  if (cl == U'\u0438') {
    return stressed ? "i" : "\xC9\xAA";
  }
  if (cl == U'\u0451') {
    return stressed ? "o" : "\xC9\x99";
  }
  if (cl == U'\u0435') {
    if (word_initial_jot) {
      return "e";
    }
    if (after_palatal) {
      return stressed ? "e" : "\xC9\xAA";
    }
    if (after_hard) {
      return stressed ? "\xC9\x9B" : "\xC9\xAA";
    }
    return stressed ? "je" : "j\xC9\xAA";
  }
  if (cl == U'\u044E') {
    if (word_initial_jot) {
      return "u";
    }
    if (after_palatal) {
      return "u";
    }
    return stressed ? "u" : "\xCA\x8A";
  }
  if (cl == U'\u044F') {
    if (word_initial_jot) {
      return stressed ? "a" : "\xC9\x99";
    }
    if (after_palatal) {
      return stressed ? "a" : "\xC9\x99";
    }
    if (after_hard) {
      return stressed ? "a" : "\xC9\x99";
    }
    return stressed ? "a" : "j\xC9\x99";
  }
  return "";
}

std::string letters_to_ipa_rules(const std::string& w_clean, int stress_syl) {
  std::string w;
  size_t z = 0;
  while (z < w_clean.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(w_clean, z, cp, adv);
    if (is_russian_lex_key_cp(russian_tolower_cp(cp)) && !is_combining_mark(cp)) {
      utf8_append_codepoint(w, russian_tolower_cp(cp));
    }
    z += adv;
  }
  if (w.empty()) {
    return "";
  }
  std::vector<int> syll_idx_per_cp = syllable_index_per_codepoint(w);
  const auto w_cps = utf8_split_codepoints(w);
  if (syll_idx_per_cp.size() != w_cps.size()) {
    syll_idx_per_cp.assign(w_cps.size(), 0);
  }

  std::vector<std::string> out;
  bool after_vowel = false;
  size_t byte_i = 0;
  const size_t n = w.size();
  int cp_index = 0;

  auto after_palatal = [&]() -> bool {
    if (out.empty()) {
      return false;
    }
    const std::string& last = out.back();
    if (last == "tɕ" || last == "ɕː" || last == "ts" || last == "ʐ") {
      return false;
    }
    return ipa_piece_ends_with_palatal(last);
  };

  auto after_hard = [&]() -> bool { return ipa_piece_after_hard_consonant(out.empty() ? "" : out.back()); };

  while (byte_i < n) {
    char32_t ch = 0;
    size_t adv = 0;
    utf8_decode_at(w, byte_i, ch, adv);
    const int syl_here =
        (cp_index >= 0 && static_cast<size_t>(cp_index) < syll_idx_per_cp.size())
            ? syll_idx_per_cp[static_cast<size_t>(cp_index)]
            : 0;
    const bool stressed = (syl_here == stress_syl);

    if (ch == U'\u044A' || ch == U'\u042A') {
      byte_i += adv;
      ++cp_index;
      continue;
    }
    if (ch == U'\u044C' || ch == U'\u042C') {
      byte_i += adv;
      ++cp_index;
      continue;
    }
    if (ch == U'\u0439' || ch == U'\u0419') {
      out.push_back("j");
      byte_i += adv;
      ++cp_index;
      after_vowel = false;
      continue;
    }
    // Python ``_letters_to_ipa_rules``: ``ш`` is not in ``_CONS_PHONE`` and not in ``цчщ``, so it is skipped.
    if (ch == U'\u0448' || ch == U'\u0428') {
      byte_i += adv;
      ++cp_index;
      after_vowel = false;
      continue;
    }

    if (is_russian_vowel_letter(ch)) {
      const bool jot = out.empty() || after_vowel;
      const bool ap = after_palatal();
      const bool ah = after_hard();
      const char32_t ve = russian_tolower_cp(ch);
      if (ve == U'\u0435' && jot) {
        out.push_back(stressed ? "je" : "j\xC9\xAA");
        byte_i += adv;
        ++cp_index;
        after_vowel = true;
        continue;
      }
      if (ve == U'\u044E' && jot) {
        out.push_back("ju");
        byte_i += adv;
        ++cp_index;
        after_vowel = true;
        continue;
      }
      if (ve == U'\u044F' && jot) {
        out.push_back(stressed ? "ja" : "j\xC9\x99");
        byte_i += adv;
        ++cp_index;
        after_vowel = true;
        continue;
      }
      out.push_back(vowel_ipa(ch, stressed, ap, ah, jot));
      byte_i += adv;
      ++cp_index;
      after_vowel = true;
      continue;
    }

    if (emit_consonant(ch, false).empty() && ch != U'\u0446' && ch != U'\u0447' && ch != U'\u0449') {
      byte_i += adv;
      ++cp_index;
      continue;
    }

    const size_t j = byte_i + adv;
    bool palatal = false;
    if (j < n) {
      char32_t nx = 0;
      size_t na = 0;
      utf8_decode_at(w, j, nx, na);
      if (nx == U'\u044C' || nx == U'\u042C') {
        palatal = palatalizable_cons(ch);
        out.push_back(emit_consonant(ch, palatal));
        byte_i = j + na;
        ++cp_index;
        after_vowel = false;
        continue;
      }
      if (nx == U'\u044A' || nx == U'\u042A') {
        out.push_back(emit_consonant(ch, false));
        byte_i = j + na;
        ++cp_index;
        after_vowel = false;
        continue;
      }
      if (is_russian_vowel_letter(nx)) {
        const char32_t v = nx;
        if ((ch == U'\u0436' || ch == U'\u0446' || ch == U'\u0448') && is_russian_vowel_letter(v)) {
          const char32_t vl = russian_tolower_cp(v);
          if (vl == U'\u0435' || vl == U'\u0451' || vl == U'\u0438' || vl == U'\u044E' ||
              vl == U'\u044F') {
            palatal = false;
          }
        } else if (ch == U'\u0447' || ch == U'\u0449') {
          palatal = false;
        } else {
          const char32_t vl = russian_tolower_cp(v);
          if (vl == U'\u0435' || vl == U'\u0451' || vl == U'\u0438' || vl == U'\u044E' ||
              vl == U'\u044F') {
            palatal = palatalizable_cons(ch);
          }
        }
        out.push_back(emit_consonant(ch, palatal));
        byte_i += adv;
        ++cp_index;
        after_vowel = false;
        continue;
      }
    }
    out.push_back(emit_consonant(ch, false));
    byte_i += adv;
    ++cp_index;
    after_vowel = false;
  }
  std::string ipa;
  for (const auto& p : out) {
    ipa += p;
  }
  return ipa;
}

std::string insert_primary_stress_before_vowel(std::string s) {
  erase_utf8_substr(s, kPri);
  erase_utf8_substr(s, kSec);
  size_t i = 0;
  while (i < s.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, i, cp, adv);
    if (cp == U'a' || cp == U'e' || cp == U'i' || cp == U'o' || cp == U'u' || cp == U'\u025B' ||
        cp == U'\u0259' || cp == U'\u0268' || cp == U'\u026A' || cp == U'\u028A' || cp == U'\u00F8' ||
        cp == U'\u0275') {
      s.insert(i, kPri);
      return s;
    }
    i += adv;
  }
  return kPri + s;
}

std::string rules_word_to_ipa_single(const std::string& w_clean, const std::string& stress_source,
                                     bool with_stress) {
  const auto syls = russian_orthographic_syllables_utf8(w_clean);
  const int stress_syl = stress_syllable_index(syls, stress_source);
  std::string body = letters_to_ipa_rules(w_clean, stress_syl);
  if (with_stress && !body.empty()) {
    body = insert_primary_stress_before_vowel(std::move(body));
  }
  return body;
}

std::string rules_word_to_ipa(const std::string& raw, bool with_stress) {
  const std::string stress_src = filter_russian_graphemes_keep_stress(raw);
  std::string stripped = strip_grapheme_diacritics_utf8(stress_src);
  std::string w;
  size_t q = 0;
  while (q < stripped.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(stripped, q, cp, adv);
    if (is_russian_lex_key_cp(cp)) {
      utf8_append_codepoint(w, cp);
    }
    q += adv;
  }
  if (w.empty()) {
    return "";
  }
  if (w.find('-') != std::string::npos) {
    std::vector<std::string> w_chunks;
    size_t start = 0;
    while (start < w.size()) {
      size_t dash = w.find('-', start);
      const size_t end = dash == std::string::npos ? w.size() : dash;
      if (end > start) {
        w_chunks.push_back(w.substr(start, end - start));
      }
      if (dash == std::string::npos) {
        break;
      }
      start = dash + 1;
    }
    std::vector<std::string> stress_chunks;
    {
      size_t st = 0;
      while (st < stress_src.size()) {
        size_t d = stress_src.find('-', st);
        const size_t en = d == std::string::npos ? stress_src.size() : d;
        if (en > st) {
          stress_chunks.push_back(stress_src.substr(st, en - st));
        }
        if (d == std::string::npos) {
          break;
        }
        st = d + 1;
      }
    }
    std::string merged;
    for (size_t c = 0; c < w_chunks.size(); ++c) {
      std::string part;
      if (stress_chunks.size() == w_chunks.size()) {
        part = rules_word_to_ipa_single(w_chunks[c], stress_chunks[c], with_stress);
      } else {
        part = rules_word_to_ipa_single(w_chunks[c], stress_src, with_stress);
      }
      if (c > 0) {
        merged += '-';
      }
      merged += part;
    }
    return merged;
  }
  return rules_word_to_ipa_single(w, stress_src, with_stress);
}

bool is_latin1_supplement_python_word_char(char32_t cp) {
  if (cp < U'\u00AA' || cp > U'\u00FF') {
    return false;
  }
  if (cp == U'\u00D7' || cp == U'\u00F7') {
    return false;
  }
  if (cp >= U'\u00C0' && cp <= U'\u00D6') {
    return true;
  }
  if (cp >= U'\u00D8' && cp <= U'\u00F6') {
    return true;
  }
  if (cp >= U'\u00F8' && cp <= U'\u00FF') {
    return true;
  }
  return cp == U'\u00AA' || cp == U'\u00B2' || cp == U'\u00B3' || cp == U'\u00B5' || cp == U'\u00B9' ||
         cp == U'\u00BA' || cp == U'\u00BC' || cp == U'\u00BD' || cp == U'\u00BE';
}

bool is_unicode_word_char_w(char32_t cp) {
  if (cp == U'-' || cp == U'_') {
    return true;
  }
  if (cp >= U'0' && cp <= U'9') {
    return true;
  }
  if (cp <= 0x7Fu) {
    return std::isalpha(static_cast<unsigned char>(cp)) != 0;
  }
  if (is_latin1_supplement_python_word_char(cp)) {
    return true;
  }
  if (cp >= U'\u0400' && cp <= U'\u04FF') {
    return true;
  }
  if (cp <= static_cast<char32_t>(std::numeric_limits<wint_t>::max())) {
    const wint_t w = static_cast<wint_t>(cp);
    if (std::iswalpha(w) != 0 || std::iswdigit(w) != 0) {
      return true;
    }
  }
  return false;
}

bool utf8_contains_cyrillic(const std::string& tok) {
  size_t i = 0;
  while (i < tok.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(tok, i, cp, adv);
    if (cp >= U'\u0400' && cp <= U'\u04FF') {
      return true;
    }
    i += adv;
  }
  return false;
}

bool try_consume_unicode_word(const std::string& text, size_t pos, size_t& out_end) {
  if (pos >= text.size()) {
    return false;
  }
  char32_t c0 = 0;
  size_t a0 = 0;
  utf8_decode_at(text, pos, c0, a0);
  if (!is_unicode_word_char_w(c0)) {
    return false;
  }
  size_t p = pos;
  while (p < text.size()) {
    char32_t c = 0;
    size_t a = 0;
    utf8_decode_at(text, p, c, a);
    if (is_unicode_word_char_w(c)) {
      p += a;
      continue;
    }
    break;
  }
  out_end = p;
  return out_end > pos;
}

}  // namespace

#include "russian-numbers.cpp"

RussianRuleG2p::RussianRuleG2p(std::filesystem::path dict_tsv)
    : RussianRuleG2p(std::move(dict_tsv), Options{}) {}

RussianRuleG2p::RussianRuleG2p(std::filesystem::path dict_tsv, Options options)
    : options_(options) {
  load_russian_lexicon_file(std::move(dict_tsv), lexicon_);
}

std::string RussianRuleG2p::finalize_ipa(std::string ipa) const {
  if (!options_.with_stress) {
    erase_utf8_substr(ipa, kPri);
    erase_utf8_substr(ipa, kSec);
    return ipa;
  }
  if (options_.vocoder_stress) {
    return GermanRuleG2p::normalize_ipa_stress_for_vocoder(std::move(ipa));
  }
  return ipa;
}

std::string RussianRuleG2p::lookup_or_rules(const std::string& raw_word) const {
  const std::string letters_only = normalize_lookup_key_utf8(raw_word);
  if (letters_only.empty()) {
    return "";
  }
  auto it = lexicon_.find(letters_only);
  if (it != lexicon_.end()) {
    return finalize_ipa(it->second);
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
  return finalize_ipa(rules_word_to_ipa(raw_word, options_.with_stress));
}

std::string RussianRuleG2p::word_to_ipa(const std::string& word) const {
  const std::string wraw = trim_ascii_ws_copy(word);
  if (wraw.empty()) {
    return "";
  }
  if (options_.expand_cardinal_digits && ru_ascii_all_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_russian_words(wraw);
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

std::string RussianRuleG2p::text_to_ipa_no_expand(const std::string& text,
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
    size_t wend = 0;
    if (try_consume_unicode_word(text, pos, wend)) {
      const std::string tok = text.substr(pos, wend - pos);
      if (utf8_contains_cyrillic(tok)) {
        const std::string k = normalize_lookup_key_utf8(tok);
        std::string wipa = word_to_ipa(tok);
        if (per_word_log != nullptr) {
          per_word_log->push_back(G2pWordLog{tok, k, G2pWordPath::kRuleBasedG2p, wipa});
          out += per_word_log->back().ipa;
        } else {
          out += wipa;
        }
      } else {
        out += tok;
      }
      pos = wend;
      continue;
    }
    const size_t start = pos;
    pos += adv;
    while (pos < n) {
      utf8_decode_at(text, pos, cp, adv);
      size_t tmp = 0;
      if (try_consume_unicode_word(text, pos, tmp) || cp == U' ' || cp == U'\t' || cp == U'\n' ||
          cp == U'\r') {
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

std::string RussianRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  if (options_.expand_cardinal_digits) {
    text = expand_russian_digit_tokens_in_text(std::move(text));
  }
  return text_to_ipa_no_expand(text, per_word_log);
}

bool dialect_resolves_to_russian_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "ru" || s == "ru-ru" || s == "russian";
}

std::vector<std::string> RussianRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first({"ru", "ru-RU", "ru_ru", "russian"});
}

std::filesystem::path resolve_russian_dict_path(const std::filesystem::path& model_root) {
  const std::filesystem::path under_data = model_root.parent_path() / "data" / "ru" / "dict.tsv";
  if (std::filesystem::is_regular_file(under_data)) {
    return under_data;
  }
  return model_root / "ru" / "dict.tsv";
}

}  // namespace moonshine_tts
