#include "ukrainian.h"

#include "g2p-word-log.h"
#include "utf8-utils.h"

extern "C" {
#include <utf8proc.h>
}

#include <cctype>
#include <cstdlib>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

namespace moonshine_tts {
namespace {

using moonshine_tts::trim_ascii_ws_copy;
using moonshine_tts::utf8_append_codepoint;
using moonshine_tts::utf8_decode_at;
using moonshine_tts::utf8_str_to_u32;

bool is_all_ascii_digits(std::string_view s) {
  if (s.empty()) {
    return false;
  }
  for (unsigned char c : s) {
    if (!std::isdigit(c)) {
      return false;
    }
  }
  return true;
}

// --- Stress stripping (NFD): remove common stress marks, keep U+0308 (ї vs і) --------------------

std::string utf8_nfc_utf8proc(const std::string& s) {
  utf8proc_uint8_t* p =
      utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(s.c_str()));
  if (p == nullptr) {
    return s;
  }
  std::string out(reinterpret_cast<char*>(p));
  std::free(p);
  return out;
}

std::string ukrainian_strip_stress_marks_utf8(std::string s) {
  utf8proc_uint8_t* nfd = utf8proc_NFD(reinterpret_cast<const utf8proc_uint8_t*>(s.c_str()));
  if (nfd == nullptr) {
    return s;
  }
  std::string stripped;
  for (const utf8proc_uint8_t* p = nfd; *p != 0;) {
    utf8proc_int32_t cp = 0;
    utf8proc_ssize_t adv = utf8proc_iterate(p, -1, &cp);
    if (adv <= 0) {
      break;
    }
    const auto cat = static_cast<utf8proc_category_t>(utf8proc_category(cp));
    // Drop all Mn except U+0308 (diaeresis) so NFD(ї) stays distinct from і.
    if (cat == UTF8PROC_CATEGORY_MN && cp != 0x308) {
      p += adv;
      continue;
    }
    utf8_append_codepoint(stripped, static_cast<char32_t>(cp));
    p += adv;
  }
  std::free(nfd);
  return utf8_nfc_utf8proc(stripped);
}

std::u32string utf8_to_u32_nfc(const std::string& s) {
  utf8proc_uint8_t* nfc = utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(s.c_str()));
  if (nfc == nullptr) {
    return utf8_str_to_u32(s);
  }
  std::string composed(reinterpret_cast<char*>(nfc));
  std::free(nfc);
  return utf8_str_to_u32(composed);
}

std::u32string ukrainian_lower_u32(const std::u32string& s) {
  std::u32string o;
  o.reserve(s.size());
  for (char32_t c : s) {
    const utf8proc_int32_t lo = utf8proc_tolower(static_cast<utf8proc_int32_t>(c));
    o.push_back(static_cast<char32_t>(lo));
  }
  return o;
}

// --- Ukrainian cardinals (UTF-8 words) --------------------------------------------------------

const char* kDigitWord[10] = {"нуль", "один", "два", "три", "чотири", "п'ять", "шість", "сім", "вісім", "дев'ять"};

const char* kTeens[10] = {"десять",      "одинадцять", "дванадцять", "тринадцять", "чотирнадцять",
                          "п'ятнадцять", "шістнадцять", "сімнадцять", "вісімнадцять", "дев'ятнадцять"};

const char* kTens[10] = {"",         "",          "двадцять", "тридцять", "сорок",      "п'ятдесят",
                         "шістдесят", "сімдесят", "вісімдесят", "дев'яносто"};

const char* kHundreds[10] = {"",        "сто",     "двісті", "триста", "чотириста", "п'ятсот",
                             "шістсот", "сімсот", "вісімсот", "дев'ятсот"};

std::string thousand_noun_utf8(int h) {
  if (h % 100 == 11 || h % 100 == 12 || h % 100 == 13 || h % 100 == 14) {
    return "тисяч";
  }
  const int m = h % 10;
  if (m == 1) {
    return "тисяча";
  }
  if (m == 2 || m == 3 || m == 4) {
    return "тисячі";
  }
  return "тисяч";
}

void append_under_100_thousand_mult(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 100) {
    return;
  }
  if (n == 1) {
    out.emplace_back("одна");
    return;
  }
  if (n == 2) {
    out.emplace_back("дві");
    return;
  }
  if (n == 3) {
    out.emplace_back("три");
    return;
  }
  if (n == 4) {
    out.emplace_back("чотири");
    return;
  }
  if (n < 20) {
    out.emplace_back(kTeens[n - 10]);
    return;
  }
  const int t = n / 10;
  const int u = n % 10;
  out.emplace_back(kTens[t]);
  if (u == 0) {
    return;
  }
  if (u == 1) {
    out.emplace_back("одна");
    return;
  }
  if (u == 2) {
    out.emplace_back("дві");
    return;
  }
  if (u == 3) {
    out.emplace_back("три");
    return;
  }
  if (u == 4) {
    out.emplace_back("чотири");
    return;
  }
  out.emplace_back(kDigitWord[u]);
}

void append_under_100_plain(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 100) {
    return;
  }
  if (n < 10) {
    out.emplace_back(kDigitWord[n]);
    return;
  }
  if (n < 20) {
    out.emplace_back(kTeens[n - 10]);
    return;
  }
  const int t = n / 10;
  const int u = n % 10;
  out.emplace_back(kTens[t]);
  if (u != 0) {
    out.emplace_back(kDigitWord[u]);
  }
}

void append_tokens_thousands_multiplier(int h, std::vector<std::string>& out) {
  if (h <= 0 || h > 999) {
    return;
  }
  const std::string tn = thousand_noun_utf8(h);
  if (h < 100) {
    append_under_100_thousand_mult(h, out);
    out.push_back(tn);
    return;
  }
  const int hundreds = h / 100;
  const int rem = h % 100;
  out.emplace_back(kHundreds[hundreds]);
  if (rem == 0) {
    out.push_back(tn);
    return;
  }
  append_under_100_thousand_mult(rem, out);
  out.push_back(tn);
}

void append_tokens_0_999(int n, std::vector<std::string>& out) {
  if (n < 0 || n > 999) {
    return;
  }
  if (n == 0) {
    out.emplace_back("нуль");
    return;
  }
  const int h = n / 100;
  const int r = n % 100;
  if (h > 0) {
    out.emplace_back(kHundreds[h]);
  }
  if (r == 0) {
    return;
  }
  append_under_100_plain(r, out);
}

void append_below_1_000_000(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 1'000'000) {
    return;
  }
  if (n < 1000) {
    append_tokens_0_999(n, out);
    return;
  }
  const int q = n / 1000;
  const int r = n % 1000;
  append_tokens_thousands_multiplier(q, out);
  if (r != 0) {
    append_tokens_0_999(r, out);
  }
}

std::string join_space(const std::vector<std::string>& p) {
  std::string o;
  for (size_t i = 0; i < p.size(); ++i) {
    if (i > 0) {
      o.push_back(' ');
    }
    o += p[i];
  }
  return o;
}

std::string expand_cardinal_digits_to_ukrainian_words(std::string_view s) {
  if (s.empty()) {
    return std::string(s);
  }
  for (unsigned char c : s) {
    if (!std::isdigit(c)) {
      return std::string(s);
    }
  }
  if (s.size() > 1 && s[0] == '0') {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
      if (i > 0) {
        o.push_back(' ');
      }
      o += kDigitWord[static_cast<size_t>(s[i] - '0')];
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
    return "нуль";
  }
  std::vector<std::string> parts;
  append_below_1_000_000(static_cast<int>(n), parts);
  return join_space(parts);
}

std::string expand_ukrainian_digit_tokens_in_text(std::string text) {
  static const std::regex range_re(R"(\b(\d+)-(\d+)\b)", std::regex::ECMAScript);
  static const std::regex dig_re(R"(\b\d+\b)", std::regex::ECMAScript);
  std::string out;
  std::sregex_iterator it(text.begin(), text.end(), range_re);
  std::sregex_iterator end;
  size_t pos = 0;
  for (; it != end; ++it) {
    const std::smatch& m = *it;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    out += expand_cardinal_digits_to_ukrainian_words(m.str(1));
    out += " - ";
    out += expand_cardinal_digits_to_ukrainian_words(m.str(2));
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
    out += expand_cardinal_digits_to_ukrainian_words(m.str());
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

// --- G2P ---------------------------------------------------------------------------------------

bool is_vowel_letter(char32_t c) {
  static const std::unordered_set<char32_t> k{U'а', U'е', U'є', U'и', U'і', U'ї', U'о', U'у', U'ю', U'я'};
  return k.count(c) != 0;
}

bool is_soft_vowel(char32_t c) {
  static const std::unordered_set<char32_t> k{U'є', U'і', U'ї', U'ю', U'я'};
  return k.count(c) != 0;
}

bool is_hard_no_pal(char32_t c) {
  static const std::unordered_set<char32_t> k{U'ж', U'ч', U'ш', U'щ'};
  return k.count(c) != 0;
}

bool is_palatalizable(char32_t c) {
  static const std::unordered_set<char32_t> k{U'б', U'в', U'г', U'ґ', U'д', U'з', U'к', U'л', U'м',
                                              U'н', U'п', U'р', U'с', U'т', U'ф', U'х', U'ц'};
  return k.count(c) != 0;
}

std::optional<size_t> next_letter_index(const std::u32string& w, size_t start) {
  for (size_t j = start; j < w.size(); ++j) {
    if (w[j] == U'\'') {
      continue;
    }
    return j;
  }
  return std::nullopt;
}

std::pair<bool, size_t> peek_apostrophe_soft_vowel(const std::u32string& w, size_t cons_end) {
  size_t j = cons_end + 1;
  if (j >= w.size() || w[j] != U'\'') {
    return {false, 0};
  }
  size_t k = j + 1;
  while (k < w.size() && w[k] == U'\'') {
    ++k;
  }
  if (k < w.size() && is_soft_vowel(w[k])) {
    return {true, k};
  }
  return {false, 0};
}

std::string v_allophone(const std::u32string& w, size_t i) {
  const auto nj = next_letter_index(w, i + 1);
  if (!nj.has_value()) {
    return "w";
  }
  const char32_t nxt = w[*nj];
  if (is_vowel_letter(nxt) || nxt == U'й') {
    return "ʋ";
  }
  if (nxt == U'ь') {
    return "w";
  }
  return "w";
}

bool ends_with_palatal_suffix(const std::string& p) {
  static const char kSuf[] = {'\xCA', '\xB2'};
  return p.size() >= 2 && p[p.size() - 2] == kSuf[0] && p[p.size() - 1] == kSuf[1];
}

bool is_vowel_ipa_piece(const std::string& p) {
  if (p.empty()) {
    return true;
  }
  const unsigned char c0 = static_cast<unsigned char>(p[0]);
  if (p == "j") {
    return true;
  }
  if (c0 == 'j' && p.size() > 1) {
    return true;
  }
  if (c0 == 'a' || c0 == 'e' || c0 == 'i' || c0 == 'o' || c0 == 'u') {
    return true;
  }
  // ɛ (U+025B) UTF-8 C9 9B, ɪ (U+026A) UTF-8 CA 8A
  if (p.size() >= 2 && c0 == 0xC9 && static_cast<unsigned char>(p[1]) == 0x9B) {
    return true;
  }
  if (p.size() >= 2 && c0 == 0xCA && static_cast<unsigned char>(p[1]) == 0x8A) {
    return true;
  }
  return false;
}

bool piece_ends_palatalized_consonant(const std::vector<std::string>& pieces) {
  for (size_t idx = pieces.size(); idx-- > 0;) {
    const std::string& p = pieces[idx];
    if (p.empty()) {
      continue;
    }
    if (is_vowel_ipa_piece(p)) {
      continue;
    }
    if (p == "dʒ" || p == "dz" || p == "tʃ" || p == "ts" || p == "ʃtʃ" || p == "ʒ" || p == "ʃ") {
      return false;
    }
    return ends_with_palatal_suffix(p);
  }
  return false;
}

void palatalize_last(std::vector<std::string>& pieces) {
  for (size_t idx = pieces.size(); idx-- > 0;) {
    std::string& p = pieces[idx];
    if (p.empty()) {
      continue;
    }
    if (p == "dʒ" || p == "dz" || p == "tʃ" || p == "ts" || p == "ʃtʃ" || p == "ʒ" || p == "ʃ") {
      return;
    }
    if (ends_with_palatal_suffix(p)) {
      return;
    }
    p += std::string{'\xCA', '\xB2'};  // U+02B2 modifier letter j (palatalization)
    return;
  }
}

std::string vowel_ipa(char32_t ch,
                      bool force_j,
                      bool after_vowel_letter,
                      bool word_onset,
                      const std::vector<std::string>& pieces) {
  if (force_j || word_onset || after_vowel_letter) {
    if (ch == U'я') {
      return "ja";
    }
    if (ch == U'ю') {
      return "ju";
    }
    if (ch == U'є') {
      return "jɛ";
    }
    if (ch == U'ї') {
      return "ji";
    }
  }
  if (ch == U'я') {
    return "a";
  }
  if (ch == U'ю') {
    return "u";
  }
  if (ch == U'є') {
    return "ɛ";
  }
  if (ch == U'ї') {
    return piece_ends_palatalized_consonant(pieces) ? "i" : "ji";
  }
  if (ch == U'а') {
    return "a";
  }
  if (ch == U'е') {
    return "ɛ";
  }
  if (ch == U'и') {
    return "ɪ";
  }
  if (ch == U'і') {
    return "i";
  }
  if (ch == U'о') {
    return "o";
  }
  if (ch == U'у') {
    return "u";
  }
  return "";
}

bool ipa_vowel_char(char32_t c) {
  return c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u' || c == U'\u025B' || c == U'\u026A';
}

std::string u32_to_utf8(const std::u32string& s) {
  std::string o;
  for (char32_t c : s) {
    utf8_append_codepoint(o, c);
  }
  return o;
}

std::string insert_primary_stress_penultimate(const std::string& ipa_utf8) {
  std::u32string u;
  for (char32_t c : utf8_str_to_u32(ipa_utf8)) {
    if (c != 0x02C8 && c != 0x02CC) {
      u.push_back(c);
    }
  }
  if (u.empty()) {
    return ipa_utf8;
  }
  std::vector<size_t> syll_starts;
  size_t i = 0;
  const size_t n = u.size();
  while (i < n) {
    if (u[i] == U'j' && i + 1 < n && ipa_vowel_char(u[i + 1])) {
      syll_starts.push_back(i);
      i += 2;
      continue;
    }
    if (ipa_vowel_char(u[i])) {
      syll_starts.push_back(i);
      ++i;
      continue;
    }
    ++i;
  }
  if (syll_starts.empty()) {
    return ipa_utf8;
  }
  const size_t stress_at = syll_starts.size() == 1 ? syll_starts[0] : syll_starts[syll_starts.size() - 2];
  std::u32string out;
  out.insert(out.end(), u.begin(), u.begin() + static_cast<std::ptrdiff_t>(stress_at));
  out.push_back(0x02C8);
  out.insert(out.end(), u.begin() + static_cast<std::ptrdiff_t>(stress_at), u.end());
  return u32_to_utf8(out);
}

std::string base_cons_ipa(char32_t c) {
  switch (c) {
  case U'б':
    return "b";
  case U'п':
    return "p";
  case U'м':
    return "m";
  case U'ф':
    return "f";
  case U'г':
    return "ɦ";
  case U'ґ':
    return "ɡ";
  case U'д':
    return "d";
  case U'т':
    return "t";
  case U'н':
    return "n";
  case U'л':
    return "l";
  case U'р':
    return "ɾ";
  case U'с':
    return "s";
  case U'з':
    return "z";
  case U'ж':
    return "ʒ";
  case U'ш':
    return "ʃ";
  case U'ч':
    return "tʃ";
  case U'щ':
    return "ʃtʃ";
  case U'ц':
    return "ts";
  case U'к':
    return "k";
  case U'х':
    return "x";
  default:
    return "";
  }
}

std::string word_to_ipa_inner(const std::u32string& w0, bool with_stress) {
  std::u32string w = w0;
  std::vector<std::string> pieces;
  size_t i = 0;
  bool prev_was_vowel_letter = false;
  bool word_onset = true;
  bool force_j_vowel = false;
  bool prev_hard_affricate = false;

  while (i < w.size()) {
    if (w[i] == U'\'') {
      ++i;
      continue;
    }
    if (i + 1 < w.size() && w[i] == U'д' && w[i + 1] == U'ж') {
      pieces.emplace_back("dʒ");
      i += 2;
      word_onset = false;
      prev_was_vowel_letter = false;
      prev_hard_affricate = true;
      continue;
    }
    if (i + 1 < w.size() && w[i] == U'д' && w[i + 1] == U'з') {
      pieces.emplace_back("dz");
      i += 2;
      word_onset = false;
      prev_was_vowel_letter = false;
      prev_hard_affricate = true;
      continue;
    }
    const char32_t ch = w[i];
    if (ch == U'ь') {
      palatalize_last(pieces);
      ++i;
      prev_hard_affricate = false;
      continue;
    }
    if (ch == U'й') {
      pieces.emplace_back("j");
      ++i;
      word_onset = false;
      prev_was_vowel_letter = false;
      prev_hard_affricate = false;
      continue;
    }
    if (is_vowel_letter(ch)) {
      const bool fj = force_j_vowel;
      if (force_j_vowel) {
        force_j_vowel = false;
      }
      pieces.emplace_back(vowel_ipa(ch, fj, prev_was_vowel_letter, word_onset, pieces));
      ++i;
      word_onset = false;
      prev_was_vowel_letter = true;
      prev_hard_affricate = false;
      continue;
    }
    const std::string bc = base_cons_ipa(ch);
    const bool is_v = (ch == U'в');
    if (bc.empty() && !is_v) {
      ++i;
      continue;
    }
    const auto [apostrophe_block, vowel_i] = peek_apostrophe_soft_vowel(w, i);
    const auto ni = next_letter_index(w, i + 1);
    const char32_t next_ch = ni.has_value() ? w[*ni] : U'\0';

    bool will_palatalize = false;
    if (!prev_hard_affricate && !apostrophe_block && is_palatalizable(ch) && !is_hard_no_pal(ch)) {
      if (is_soft_vowel(next_ch) || next_ch == U'і') {
        will_palatalize = true;
      }
    }
    if (is_v) {
      pieces.emplace_back(v_allophone(w, i));
    } else {
      pieces.emplace_back(bc);
    }
    if (will_palatalize) {
      palatalize_last(pieces);
    }
    if (apostrophe_block) {
      force_j_vowel = true;
      i = vowel_i;
      word_onset = false;
      prev_was_vowel_letter = false;
      prev_hard_affricate = false;
      continue;
    }
    ++i;
    word_onset = false;
    prev_was_vowel_letter = false;
    prev_hard_affricate = is_hard_no_pal(ch);
  }
  std::string ipa;
  for (const auto& p : pieces) {
    ipa += p;
  }
  if (with_stress) {
    ipa = insert_primary_stress_penultimate(ipa);
  }
  return ipa;
}

std::u32string filter_uk_word_chars(const std::u32string& w) {
  std::u32string o;
  for (char32_t c : w) {
    if (c == U'\'' || c == U'\u2019' || c == U'\u2018') {
      o.push_back(U'\'');
      continue;
    }
    const auto cat = static_cast<utf8proc_category_t>(utf8proc_category(static_cast<utf8proc_int32_t>(c)));
    if (cat == UTF8PROC_CATEGORY_LL || cat == UTF8PROC_CATEGORY_LU || cat == UTF8PROC_CATEGORY_LT) {
      o.push_back(c);
    }
  }
  return o;
}

bool is_ukrainian_word_char(char32_t cp) {
  if (cp == U'\'' || cp == U'\u2019' || cp == U'\u2018') {
    return true;
  }
  if (cp == U'_') {
    return true;
  }
  const auto cat = static_cast<utf8proc_category_t>(utf8proc_category(static_cast<utf8proc_int32_t>(cp)));
  if (cat == UTF8PROC_CATEGORY_ND) {
    return true;
  }
  return cat == UTF8PROC_CATEGORY_LL || cat == UTF8PROC_CATEGORY_LU || cat == UTF8PROC_CATEGORY_LT ||
         cat == UTF8PROC_CATEGORY_LM || cat == UTF8PROC_CATEGORY_LO;
}

bool is_space_cp(char32_t cp) {
  return cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r' || cp == U'\f' || cp == U'\v' ||
         utf8proc_category(static_cast<utf8proc_int32_t>(cp)) == UTF8PROC_CATEGORY_ZS;
}

std::string word_to_ipa_from_utf32_word(const std::u32string& letters, bool with_stress) {
  if (letters.empty()) {
    return "";
  }
  return word_to_ipa_inner(letters, with_stress);
}

std::string hyphen_join_word_ipas(const std::string& tok, bool with_stress) {
  std::string out;
  size_t a = 0;
  while (a < tok.size()) {
    const size_t dash = tok.find('-', a);
    const std::string part = tok.substr(a, dash == std::string::npos ? std::string::npos : dash - a);
    std::u32string w = utf8_to_u32_nfc(part);
    w = ukrainian_lower_u32(w);
    w = utf8_str_to_u32(ukrainian_strip_stress_marks_utf8(u32_to_utf8(w)));
    w = filter_uk_word_chars(w);
    if (!out.empty()) {
      out.push_back('-');
    }
    out += word_to_ipa_from_utf32_word(w, with_stress);
    if (dash == std::string::npos) {
      break;
    }
    a = dash + 1;
  }
  return out;
}

}  // namespace

UkrainianRuleG2p::UkrainianRuleG2p() = default;

UkrainianRuleG2p::UkrainianRuleG2p(Options options) : options_(options) {}

std::vector<std::string> UkrainianRuleG2p::dialect_ids() {
  return {"uk", "uk-UA", "ukrainian"};
}

std::string UkrainianRuleG2p::word_to_ipa(const std::string& word) const {
  const std::string wraw = trim_ascii_ws_copy(word);
  if (wraw.empty()) {
    return "";
  }
  if (options_.expand_cardinal_digits && is_all_ascii_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_ukrainian_words(wraw);
    if (phrase != wraw) {
      return UkrainianRuleG2p(Options{.with_stress = options_.with_stress, .expand_cardinal_digits = false})
          .text_to_ipa(phrase, nullptr);
    }
    return wraw;
  }
  if (!options_.expand_cardinal_digits) {
    static const std::regex dig_pass(R"(^[0-9]+$)", std::regex::ECMAScript);
    if (std::regex_match(wraw, dig_pass)) {
      return wraw;
    }
  }
  return hyphen_join_word_ipas(wraw, options_.with_stress);
}

std::string UkrainianRuleG2p::text_to_ipa_no_expand(const std::string& text,
                                                     std::vector<G2pWordLog>* per_word_log) const {
  std::string out;
  size_t pos = 0;
  const size_t n = text.size();
  while (pos < n) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(text, pos, cp, adv);
    if (is_space_cp(cp)) {
      out.push_back(' ');
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (!is_space_cp(cp)) {
          break;
        }
        pos += adv;
      }
      continue;
    }
    if (is_ukrainian_word_char(cp)) {
      const size_t start = pos;
      pos += adv;
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (!is_ukrainian_word_char(cp)) {
          break;
        }
        pos += adv;
      }
      const std::string tok = text.substr(start, pos - start);
      std::string wipa = UkrainianRuleG2p(Options{.with_stress = options_.with_stress, .expand_cardinal_digits = false})
                             .word_to_ipa(tok);
      if (per_word_log != nullptr) {
        per_word_log->push_back(G2pWordLog{tok, tok, G2pWordPath::kRuleBasedG2p, std::move(wipa)});
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
      if (is_ukrainian_word_char(cp) || is_space_cp(cp)) {
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

std::string UkrainianRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  if (options_.expand_cardinal_digits) {
    text = expand_ukrainian_digit_tokens_in_text(std::move(text));
  }
  return text_to_ipa_no_expand(text, per_word_log);
}

bool dialect_resolves_to_ukrainian_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  if (s == "uk" || s == "uk-ua" || s == "ukrainian") {
    return true;
  }
  return false;
}

std::string ukrainian_word_to_ipa(const std::string& word, bool with_stress, bool expand_cardinal_digits) {
  UkrainianRuleG2p::Options o;
  o.with_stress = with_stress;
  o.expand_cardinal_digits = expand_cardinal_digits;
  return UkrainianRuleG2p(std::move(o)).word_to_ipa(word);
}

std::string ukrainian_text_to_ipa(const std::string& text, bool with_stress,
                                  std::vector<G2pWordLog>* per_word_log, bool expand_cardinal_digits) {
  UkrainianRuleG2p::Options o;
  o.with_stress = with_stress;
  o.expand_cardinal_digits = expand_cardinal_digits;
  return UkrainianRuleG2p(std::move(o)).text_to_ipa(text, per_word_log);
}

}  // namespace moonshine_tts
