#include "turkish.h"

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
#include <vector>

namespace moonshine_tts {
namespace {

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

// --- Turkish cardinals (orthographic words, UTF-8) ---------------------------------------------

const char* kDigitWord[10] = {"sıfır", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"};

const char* kTens[10] = {"", "", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"};

void append_under_100(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 100) {
    return;
  }
  if (n < 10) {
    out.emplace_back(kDigitWord[n]);
    return;
  }
  if (n == 10) {
    out.emplace_back("on");
    return;
  }
  if (n < 20) {
    out.emplace_back("on");
    out.emplace_back(kDigitWord[n - 10]);
    return;
  }
  const int t = n / 10;
  const int u = n % 10;
  out.emplace_back(kTens[t]);
  if (u != 0) {
    out.emplace_back(kDigitWord[u]);
  }
}

void append_tokens_0_999(int n, std::vector<std::string>& out) {
  if (n < 0 || n > 999) {
    return;
  }
  if (n == 0) {
    out.emplace_back("sıfır");
    return;
  }
  const int h = n / 100;
  const int r = n % 100;
  if (h > 0) {
    if (h == 1) {
      out.emplace_back("yüz");
    } else {
      out.emplace_back(kDigitWord[h]);
      out.emplace_back("yüz");
    }
    if (r == 0) {
      return;
    }
  }
  append_under_100(r, out);
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
  if (q == 1) {
    out.emplace_back("bin");
  } else {
    append_tokens_0_999(q, out);
    out.emplace_back("bin");
  }
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

std::string expand_cardinal_digits_to_turkish_words(std::string_view s) {
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
    return "sıfır";
  }
  std::vector<std::string> parts;
  append_below_1_000_000(static_cast<int>(n), parts);
  return join_space(parts);
}

std::string expand_turkish_digit_tokens_in_text(std::string text) {
  static const std::regex range_re(R"(\b(\d+)-(\d+)\b)", std::regex::ECMAScript);
  static const std::regex dig_re(R"(\b\d+\b)", std::regex::ECMAScript);
  std::string out;
  std::sregex_iterator it(text.begin(), text.end(), range_re);
  std::sregex_iterator end;
  size_t pos = 0;
  for (; it != end; ++it) {
    const std::smatch& m = *it;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    out += expand_cardinal_digits_to_turkish_words(m.str(1));
    out += " - ";
    out += expand_cardinal_digits_to_turkish_words(m.str(2));
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
    out += expand_cardinal_digits_to_turkish_words(m.str());
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

// --- NFC + lower + letters --------------------------------------------------

std::u32string utf8_to_u32_nfc(const std::string& s) {
  utf8proc_uint8_t* nfc = utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(s.c_str()));
  if (nfc == nullptr) {
    return utf8_str_to_u32(s);
  }
  std::string composed(reinterpret_cast<char*>(nfc));
  std::free(nfc);
  return utf8_str_to_u32(composed);
}

char32_t turkish_tolower_cp(char32_t cp) {
  if (cp == U'İ') {
    return U'i';
  }
  if (cp == U'I') {
    return U'ı';
  }
  return static_cast<char32_t>(utf8proc_tolower(static_cast<utf8proc_int32_t>(cp)));
}

std::u32string turkish_lower_u32(const std::u32string& s) {
  std::u32string o;
  o.reserve(s.size());
  for (char32_t c : s) {
    o.push_back(turkish_tolower_cp(c));
  }
  return o;
}

bool is_tr_g2p_letter(char32_t c) {
  switch (c) {
  case U'a':
  case U'b':
  case U'c':
  case U'ç':
  case U'd':
  case U'e':
  case U'f':
  case U'g':
  case U'ğ':
  case U'h':
  case U'ı':
  case U'i':
  case U'j':
  case U'k':
  case U'l':
  case U'm':
  case U'n':
  case U'o':
  case U'ö':
  case U'p':
  case U'r':
  case U's':
  case U'ş':
  case U't':
  case U'u':
  case U'ü':
  case U'v':
  case U'y':
  case U'z':
  case U'q':
  case U'w':
  case U'x':
  case U'â':
  case U'ê':
  case U'î':
  case U'ô':
  case U'û':
    return true;
  default:
    return false;
  }
}

std::u32string letters_only_u32(const std::u32string& w) {
  std::u32string o;
  for (char32_t c : w) {
    if (c == U'\'') {
      continue;
    }
    if (is_tr_g2p_letter(c)) {
      o.push_back(c);
    }
  }
  return o;
}

bool is_vowel_orth(char32_t c) {
  return c == U'a' || c == U'e' || c == U'ı' || c == U'i' || c == U'o' || c == U'ö' || c == U'u' || c == U'ü' ||
         c == U'â' || c == U'ê' || c == U'î' || c == U'ô' || c == U'û';
}

bool is_front_vowel(char32_t c) {
  return c == U'e' || c == U'i' || c == U'ö' || c == U'ü' || c == U'ê' || c == U'î';
}

std::optional<size_t> prev_letter_index(const std::u32string& w, size_t i) {
  for (size_t j = i; j-- > 0;) {
    const char32_t c = w[j];
    if (is_tr_g2p_letter(c)) {
      return j;
    }
  }
  return std::nullopt;
}

std::optional<size_t> next_letter_index(const std::u32string& w, size_t i) {
  for (size_t j = i + 1; j < w.size(); ++j) {
    const char32_t c = w[j];
    if (is_tr_g2p_letter(c)) {
      return j;
    }
  }
  return std::nullopt;
}

std::optional<char32_t> next_vowel_from(const std::u32string& w, size_t start) {
  for (size_t j = start; j < w.size(); ++j) {
    if (is_vowel_orth(w[j])) {
      return w[j];
    }
  }
  return std::nullopt;
}

std::optional<char32_t> last_vowel_before(const std::u32string& w, size_t end) {
  for (size_t j = end; j-- > 0;) {
    if (is_vowel_orth(w[j])) {
      return w[j];
    }
  }
  return std::nullopt;
}

std::optional<char32_t> harmony_vowel_for_kg(const std::u32string& w, size_t cons_index) {
  auto n = next_vowel_from(w, cons_index + 1);
  if (n.has_value()) {
    return n;
  }
  return last_vowel_before(w, cons_index);
}

std::string map_k_or_g(char32_t ch, const std::u32string& w, size_t i) {
  const auto hv = harmony_vowel_for_kg(w, i);
  if (!hv.has_value()) {
    return ch == U'k' ? "k" : "ɡ";
  }
  const bool front = is_front_vowel(*hv);
  if (ch == U'k') {
    return front ? "c" : "k";
  }
  return front ? "ɟ" : "ɡ";
}

std::string map_simple_char(char32_t c) {
  switch (c) {
  case U'a':
    return "a";
  case U'b':
    return "b";
  case U'c':
    return "dʒ";
  case U'ç':
    return "tʃ";
  case U'd':
    return "d";
  case U'e':
    return "e";
  case U'f':
    return "f";
  case U'h':
    return "h";
  case U'ı':
    return "ɯ";
  case U'i':
    return "i";
  case U'j':
    return "ʒ";
  case U'l':
    return "l";
  case U'm':
    return "m";
  case U'n':
    return "n";
  case U'o':
    return "o";
  case U'ö':
    return "ø";
  case U'p':
    return "p";
  case U'r':
    return "ɾ";
  case U's':
    return "s";
  case U'ş':
    return "ʃ";
  case U't':
    return "t";
  case U'u':
    return "u";
  case U'ü':
    return "y";
  case U'v':
    return "v";
  case U'y':
    return "j";
  case U'z':
    return "z";
  case U'q':
    return "k";
  case U'w':
    return "v";
  case U'x':
    return "ks";
  case U'â':
    return "a";
  case U'ê':
    return "e";
  case U'î':
    return "i";
  case U'ô':
    return "o";
  case U'û':
    return "u";
  default:
    return "";
  }
}

bool is_vowel_ipa_char(char32_t c) {
  return c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u' || c == U'y' || c == U'\u026F' ||
         c == U'\u00F8';
}

std::u32string utf8_ipa_to_u32(std::string_view s) {
  std::string tmp(s);
  return utf8_str_to_u32(tmp);
}

std::string u32_to_utf8_ipa(const std::u32string& s) {
  std::string o;
  for (char32_t c : s) {
    utf8_append_codepoint(o, c);
  }
  return o;
}

std::string insert_primary_stress_final(const std::string& ipa_utf8) {
  std::u32string u = utf8_ipa_to_u32(ipa_utf8);
  if (u.empty()) {
    return ipa_utf8;
  }
  // Strip existing primary stress to re-place
  std::u32string v;
  v.reserve(u.size());
  for (char32_t c : u) {
    if (c != 0x02C8) {
      v.push_back(c);
    }
  }
  u = std::move(v);
  const char32_t long_mark = 0x02D0;
  int j = static_cast<int>(u.size()) - 1;
  while (j >= 0) {
    if (u[static_cast<size_t>(j)] == long_mark && j >= 1) {
      j -= 1;
      if (is_vowel_ipa_char(u[static_cast<size_t>(j)])) {
        const size_t pos = static_cast<size_t>(j);
        std::u32string out;
        out.insert(out.end(), u.begin(), u.begin() + static_cast<std::ptrdiff_t>(pos));
        out.push_back(0x02C8);
        out.push_back(u[pos]);
        out.push_back(long_mark);
        out.insert(out.end(), u.begin() + static_cast<std::ptrdiff_t>(pos) + 2, u.end());
        return u32_to_utf8_ipa(out);
      }
      j -= 1;
      continue;
    }
    if (is_vowel_ipa_char(u[static_cast<size_t>(j)])) {
      const size_t pos = static_cast<size_t>(j);
      std::u32string out;
      out.insert(out.end(), u.begin(), u.begin() + static_cast<std::ptrdiff_t>(pos));
      out.push_back(0x02C8);
      out.insert(out.end(), u.begin() + static_cast<std::ptrdiff_t>(pos), u.end());
      return u32_to_utf8_ipa(out);
    }
    j -= 1;
  }
  return ipa_utf8;
}

bool is_turkish_word_char(char32_t cp) {
  // Apostrophe separates proper nouns from suffixes (İstanbul'da); not part of a word token.
  // utf8proc may categorize U+0027 as a letter-like code point in some builds — force non-word.
  if (cp == U'\'' || cp == U'\u2019' || cp == U'\u2018') {
    return false;
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

}  // namespace

TurkishRuleG2p::TurkishRuleG2p() = default;

TurkishRuleG2p::TurkishRuleG2p(Options options) : options_(options) {}

std::vector<std::string> TurkishRuleG2p::dialect_ids() {
  return {"tr", "tr-TR", "turkish"};
}

std::string TurkishRuleG2p::word_to_ipa(const std::string& word) const {
  const std::string wraw = trim_ascii_ws_copy(word);
  if (wraw.empty()) {
    return "";
  }
  if (options_.expand_cardinal_digits && is_all_ascii_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_turkish_words(wraw);
    if (phrase != wraw) {
      return TurkishRuleG2p(Options{.with_stress = options_.with_stress, .expand_cardinal_digits = false})
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

  std::u32string w = utf8_to_u32_nfc(wraw);
  w = turkish_lower_u32(w);
  const std::u32string letters = letters_only_u32(w);
  if (letters.empty()) {
    return "";
  }

  std::vector<std::string> pieces;
  for (size_t i = 0; i < letters.size(); ++i) {
    const char32_t c = letters[i];
    if (c == U'ğ') {
      const auto pi = prev_letter_index(letters, i);
      const auto ni = next_letter_index(letters, i);
      const bool prev_v =
          pi.has_value() && is_vowel_orth(letters[*pi]);
      const bool next_v = ni.has_value() && is_vowel_orth(letters[*ni]);
      if (prev_v && next_v) {
        const char32_t pv = letters[*pi];
        pieces.push_back((pv == U'a' || pv == U'ı' || pv == U'o' || pv == U'u' || pv == U'â' || pv == U'ô' ||
                          pv == U'û')
                             ? "ɰ"
                             : "j");
      } else if (prev_v) {
        for (int k = static_cast<int>(pieces.size()) - 1; k >= 0; --k) {
          std::string& seg = pieces[static_cast<size_t>(k)];
          if (seg.empty()) {
            continue;
          }
          std::u32string su = utf8_str_to_u32(seg);
          if (su.empty()) {
            continue;
          }
          const char32_t last = su.back();
          if (last == 0x02D0) {
            continue;
          }
          if (is_vowel_ipa_char(last)) {
            utf8_append_codepoint(seg, 0x02D0);
            break;
          }
        }
      }
      continue;
    }
    if (c == U'k' || c == U'g') {
      pieces.push_back(map_k_or_g(c, letters, i));
      continue;
    }
    std::string frag = map_simple_char(c);
    if (!frag.empty()) {
      pieces.push_back(std::move(frag));
    }
  }

  std::string ipa;
  for (const auto& p : pieces) {
    ipa += p;
  }
  if (options_.with_stress) {
    ipa = insert_primary_stress_final(ipa);
  }
  return ipa;
}

std::string TurkishRuleG2p::text_to_ipa_no_expand(const std::string& text,
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
    if (is_turkish_word_char(cp)) {
      const size_t start = pos;
      pos += adv;
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (!is_turkish_word_char(cp)) {
          break;
        }
        pos += adv;
      }
      const std::string tok = text.substr(start, pos - start);
      std::string wipa = TurkishRuleG2p(Options{.with_stress = options_.with_stress, .expand_cardinal_digits = false})
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
      if (is_turkish_word_char(cp) || is_space_cp(cp)) {
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

std::string TurkishRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  if (options_.expand_cardinal_digits) {
    text = expand_turkish_digit_tokens_in_text(std::move(text));
  }
  return text_to_ipa_no_expand(text, per_word_log);
}

bool dialect_resolves_to_turkish_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  if (s == "tr" || s == "tr-tr" || s == "turkish") {
    return true;
  }
  return false;
}

std::string turkish_word_to_ipa(const std::string& word, bool with_stress, bool expand_cardinal_digits) {
  TurkishRuleG2p::Options o;
  o.with_stress = with_stress;
  o.expand_cardinal_digits = expand_cardinal_digits;
  return TurkishRuleG2p(std::move(o)).word_to_ipa(word);
}

std::string turkish_text_to_ipa(const std::string& text, bool with_stress,
                                std::vector<G2pWordLog>* per_word_log, bool expand_cardinal_digits) {
  TurkishRuleG2p::Options o;
  o.with_stress = with_stress;
  o.expand_cardinal_digits = expand_cardinal_digits;
  return TurkishRuleG2p(std::move(o)).text_to_ipa(text, per_word_log);
}

}  // namespace moonshine_tts
