#include "dutch.h"

#include "g2p-word-log.h"
#include "ipa-symbols.h"
#include "utf8-utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace moonshine_tts {
namespace {

using ipa::kPrimaryStressUtf8;
using ipa::kSecondaryStressUtf8;
using moonshine_tts::erase_utf8_substr;
using moonshine_tts::trim_ascii_ws_copy;
using moonshine_tts::utf8_decode_at;

char32_t dutch_unicode_tolower_cp(char32_t c) {
  switch (c) {
  case U'À':
    return U'à';
  case U'Á':
    return U'á';
  case U'Â':
    return U'â';
  case U'Ã':
    return U'ã';
  case U'Ä':
    return U'ä';
  case U'Å':
    return U'å';
  case U'É':
    return U'é';
  case U'È':
    return U'è';
  case U'Ê':
    return U'ê';
  case U'Ë':
    return U'ë';
  case U'Í':
    return U'í';
  case U'Ì':
    return U'ì';
  case U'Î':
    return U'î';
  case U'Ï':
    return U'ï';
  case U'Ó':
    return U'ó';
  case U'Ò':
    return U'ò';
  case U'Ô':
    return U'ô';
  case U'Õ':
    return U'õ';
  case U'Ö':
    return U'ö';
  case U'Ú':
    return U'ú';
  case U'Ù':
    return U'ù';
  case U'Û':
    return U'û';
  case U'Ü':
    return U'ü';
  case U'Ý':
    return U'ý';
  case U'Ÿ':
    return U'ÿ';
  case U'Ç':
    return U'ç';
  case U'Ñ':
    return U'ñ';
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

/// Fold to ``a-z`` + hyphen for TSV keys (mirrors Python ``normalize_lexicon_key`` intent).
void append_lexicon_folded(std::string& out, char32_t cl) {
  if (cl == U'-') {
    out.push_back('-');
    return;
  }
  if (cl >= U'a' && cl <= U'z') {
    utf8_append_codepoint(out, cl);
    return;
  }
  switch (cl) {
  case U'à':
  case U'á':
  case U'â':
  case U'ã':
  case U'ä':
  case U'å':
    out.push_back('a');
    return;
  case U'è':
  case U'é':
  case U'ê':
  case U'ë':
    out.push_back('e');
    return;
  case U'ì':
  case U'í':
  case U'î':
  case U'ï':
    out.push_back('i');
    return;
  case U'ò':
  case U'ó':
  case U'ô':
  case U'õ':
  case U'ö':
    out.push_back('o');
    return;
  case U'ù':
  case U'ú':
  case U'û':
  case U'ü':
    out.push_back('u');
    return;
  case U'ý':
  case U'ÿ':
    out.push_back('y');
    return;
  case U'ç':
    out.push_back('c');
    return;
  case U'ñ':
    out.push_back('n');
    return;
  default:
    return;
  }
}

std::string normalize_lexicon_key_utf8(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    const char32_t cl = dutch_unicode_tolower_cp(cp);
    if (cl == U'\u0133' || cl == U'\u0132') {
      out += "ij";
    } else {
      append_lexicon_folded(out, cl);
    }
    i += adv;
  }
  return out;
}

bool is_grapheme_char(char32_t cl) {
  if (cl == U'-') {
    return true;
  }
  if (cl >= U'a' && cl <= U'z') {
    return true;
  }
  return cl == U'á' || cl == U'é' || cl == U'í' || cl == U'ó' || cl == U'ú' || cl == U'à' ||
         cl == U'è' || cl == U'ê' || cl == U'ë' || cl == U'ï' || cl == U'ö' || cl == U'ü';
}

std::u32string normalize_grapheme_key_u32(const std::string& word) {
  std::u32string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    const char32_t cl = dutch_unicode_tolower_cp(cp);
    if (cl == U'\u0133' || cl == U'\u0132') {
      out.push_back(U'i');
      out.push_back(U'j');
    } else if (is_grapheme_char(cl)) {
      out.push_back(cl);
    }
    i += adv;
  }
  return out;
}

static const char* kDigitWord[] = {"nul",  "een",  "twee", "drie", "vier",
                                   "vijf", "zes",  "zeven", "acht", "negen"};

static const char* kTeenWord(int n) {
  static const char* w[] = {"dertien", "veertien", "vijftien", "zestien", "zeventien", "achttien",
                            "negentien"};
  if (n >= 13 && n <= 19) {
    return w[n - 13];
  }
  return "";
}

std::string join_unit_tens(int u, std::string_view tens_word) {
  if (u == 0) {
    return std::string(tens_word);
  }
  static const char* stem[] = {"een", "twee", "drie", "vier", "vijf", "zes", "zeven", "acht", "negen"};
  std::string r = stem[u - 1];
  r += "en";
  r += tens_word;
  return r;
}

static const char* kTens[] = {"", "", "twintig", "dertig", "veertig", "vijftig",
                              "zestig", "zeventig", "tachtig", "negentig"};

std::string below_100(int n) {
  if (n < 0 || n >= 100) {
    throw std::out_of_range("below_100");
  }
  if (n < 10) {
    return kDigitWord[n];
  }
  if (n < 13) {
    static const char* w[] = {"tien", "elf", "twaalf"};
    return w[n - 10];
  }
  if (n < 20) {
    return kTeenWord(n);
  }
  const int t = n / 10;
  const int u = n % 10;
  return join_unit_tens(u, kTens[t]);
}

std::string below_1000_spaced(int n) {
  if (n < 0 || n >= 1000) {
    throw std::out_of_range("below_1000_spaced");
  }
  if (n < 100) {
    return below_100(n);
  }
  const int h = n / 100;
  const int r = n % 100;
  std::string head = (h == 1) ? "honderd" : (std::string(kDigitWord[h]) + "honderd");
  if (r == 0) {
    return head;
  }
  return head + " " + below_100(r);
}

// For 1100–1999: n/100 is 11..19 (``elfhonderd`` … ``negentienhonderd``).
static const char* kTeenHundred[] = {"elf",     "twaalf",  "dertien", "veertien", "vijftien",
                                     "zestien", "zeventien", "achttien", "negentien"};

std::string from_1000_to_9999(int n) {
  if (n < 1000 || n > 9999) {
    throw std::out_of_range("from_1000_to_9999");
  }
  if (n < 1100) {
    if (n == 1000) {
      return "duizend";
    }
    return "duizend " + below_100(n - 1000);
  }
  if (n < 2000) {
    const int c = n / 100;
    const int r = n % 100;
    const char* th = (c >= 11 && c <= 19) ? kTeenHundred[c - 11] : nullptr;
    if (!th) {
      throw std::out_of_range("teen hundred");
    }
    std::string head = std::string(th) + "honderd";
    if (r == 0) {
      return head;
    }
    return head + " " + below_100(r);
  }
  const int q = n / 1000;
  const int r = n % 1000;
  std::string left = (q == 1) ? "duizend" : (below_100(q) + "duizend");
  if (r == 0) {
    return left;
  }
  return left + " " + below_1000_spaced(r);
}

std::string fix_thousands_compound(int q) {
  if (q == 1) {
    return "duizend";
  }
  if (q < 10) {
    return std::string(kDigitWord[q]) + "duizend";
  }
  if (q < 100) {
    return below_100(q) + "duizend";
  }
  return below_1000_spaced(q) + " duizend";
}

std::string below_1_000_000_v2(int n) {
  if (n < 10'000) {
    if (n < 1000) {
      return below_1000_spaced(n);
    }
    return from_1000_to_9999(n);
  }
  const int q = n / 1000;
  const int r = n % 1000;
  std::string left = (q == 1) ? "duizend" : fix_thousands_compound(q);
  if (r == 0) {
    return left;
  }
  return left + " " + below_1000_spaced(r);
}

std::string expand_cardinal_digits_to_dutch_words(std::string_view sv) {
  std::string s(sv);
  bool all_digit = !s.empty();
  for (unsigned char c : s) {
    if (!std::isdigit(c)) {
      all_digit = false;
      break;
    }
  }
  if (!all_digit) {
    return s;
  }
  if (s.size() > 1 && s[0] == '0') {
    std::string out;
    for (size_t k = 0; k < s.size(); ++k) {
      if (k > 0) {
        out.push_back(' ');
      }
      out += kDigitWord[s[k] - '0'];
    }
    return out;
  }
  const long long n = std::stoll(s);
  if (n > 999'999) {
    return s;
  }
  if (n == 0) {
    return "nul";
  }
  return below_1_000_000_v2(static_cast<int>(n));
}

bool is_ascii_digit(char c) {
  return c >= '0' && c <= '9';
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

constexpr std::array<char32_t, 46> kLetterlikeWordChars = {
    U'\u2102', U'\u2107', U'\u210A', U'\u210B', U'\u210C', U'\u210D', U'\u210E', U'\u210F', U'\u2110', U'\u2111',
    U'\u2112', U'\u2113', U'\u2115', U'\u2119', U'\u211A', U'\u211B', U'\u211C', U'\u211D', U'\u2124', U'\u2126',
    U'\u2128', U'\u212A', U'\u212B', U'\u212C', U'\u212D', U'\u212F', U'\u2130', U'\u2131', U'\u2132', U'\u2133',
    U'\u2134', U'\u2135', U'\u2136', U'\u2137', U'\u2138', U'\u2139', U'\u213C', U'\u213D', U'\u213E', U'\u213F',
    U'\u2145', U'\u2146', U'\u2147', U'\u2148', U'\u2149', U'\u214E'};

bool is_letterlike_math_word_char(char32_t cp) {
  return std::binary_search(kLetterlikeWordChars.begin(), kLetterlikeWordChars.end(), cp);
}

bool is_dutch_word_char(char32_t cp) {
  // Match Python 3 ``re.UNICODE`` ``\\w``: ASCII apostrophe is not a word char (so ``nazi's`` → nazi, ', s).
  if (cp == U'-') {
    return true;
  }
  if (cp >= U'0' && cp <= U'9') {
    return true;
  }
  if (is_latin1_supplement_python_word_char(cp)) {
    return true;
  }
  if (cp >= U'\u0100' && cp <= U'\u024F') {
    return true;
  }
  if (is_letterlike_math_word_char(cp)) {
    return true;
  }
  const char32_t cl = dutch_unicode_tolower_cp(cp);
  if (cl >= U'a' && cl <= U'z') {
    return true;
  }
  return cl == U'à' || cl == U'â' || cl == U'ä' || cl == U'é' || cl == U'è' || cl == U'ê' || cl == U'ë' ||
         cl == U'ï' || cl == U'î' || cl == U'ô' || cl == U'ù' || cl == U'û' || cl == U'ü' || cl == U'ÿ' ||
         cl == U'ç' || cl == U'œ' || cl == U'æ' || cl == U'á' || cl == U'í' || cl == U'ó' || cl == U'ú' ||
         cl == U'ñ' || cl == U'ý';
}

size_t prev_utf8_index(const std::string& t, size_t byte_i) {
  if (byte_i == 0) {
    return 0;
  }
  size_t j = byte_i;
  do {
    --j;
  } while (j > 0 && (static_cast<unsigned char>(t[j]) & 0xC0) == 0x80);
  return j;
}

bool word_boundary_before(const std::string& t, size_t byte_i) {
  if (byte_i == 0) {
    return true;
  }
  char32_t cp = 0;
  size_t adv = 0;
  const size_t pi = prev_utf8_index(t, byte_i);
  utf8_decode_at(t, pi, cp, adv);
  return !is_dutch_word_char(cp);
}

bool word_boundary_after(const std::string& t, size_t byte_i) {
  if (byte_i >= t.size()) {
    return true;
  }
  char32_t cp = 0;
  size_t adv = 0;
  utf8_decode_at(t, byte_i, cp, adv);
  return !is_dutch_word_char(cp);
}

std::string expand_digit_tokens_in_text(std::string_view text_sv) {
  std::string text(text_sv);
  std::string out;
  size_t i = 0;
  const size_t n = text.size();
  while (i < n) {
    if (!is_ascii_digit(static_cast<unsigned char>(text[i]))) {
      char32_t cp = 0;
      size_t adv = 0;
      utf8_decode_at(text, i, cp, adv);
      utf8_append_codepoint(out, cp);
      i += adv;
      continue;
    }
    if (!word_boundary_before(text, i)) {
      out.push_back(text[i]);
      ++i;
      continue;
    }
    size_t j = i;
    while (j < n && is_ascii_digit(static_cast<unsigned char>(text[j]))) {
      ++j;
    }
    if (j < n && text[j] == '-' && j + 1 < n && is_ascii_digit(static_cast<unsigned char>(text[j + 1]))) {
      size_t mid = j;
      size_t k = j + 1;
      while (k < n && is_ascii_digit(static_cast<unsigned char>(text[k]))) {
        ++k;
      }
      if (word_boundary_after(text, k)) {
        const std::string a = text.substr(i, mid - i);
        const std::string b = text.substr(mid + 1, k - (mid + 1));
        out += expand_cardinal_digits_to_dutch_words(a);
        out += " tot ";
        out += expand_cardinal_digits_to_dutch_words(b);
        i = k;
        continue;
      }
    }
    if (word_boundary_after(text, j)) {
      out += expand_cardinal_digits_to_dutch_words(text.substr(i, j - i));
      i = j;
      continue;
    }
    out.push_back(text[i]);
    ++i;
  }
  return out;
}

bool digit_pass_through_pattern(std::string_view raw) {
  if (raw.empty()) {
    return false;
  }
  size_t p = 0;
  while (p < raw.size()) {
    if (!is_ascii_digit(static_cast<unsigned char>(raw[p]))) {
      return false;
    }
    while (p < raw.size() && is_ascii_digit(static_cast<unsigned char>(raw[p]))) {
      ++p;
    }
    if (p >= raw.size()) {
      return true;
    }
    if (raw[p] != '-') {
      return false;
    }
    ++p;
  }
  return false;
}

bool all_ascii_digits_string(const std::string& s) {
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

std::string apply_lexicon_ipa_postprocess(std::string ipa, std::string_view word_key) {
  static const std::string kArch{"\xC9\x91r.\xCA\x92i\xCB\x90"};  // ɑr.ʒiː
  static const std::string kArchRep{"\xC9\x91r.xi"};               // ɑr.xi
  for (;;) {
    const size_t p = ipa.find(kArch);
    if (p == std::string::npos) {
      break;
    }
    ipa.replace(p, kArch.size(), kArchRep);
  }
  if (!word_key.empty() && word_key[0] == 'g') {
    static const std::string kGh{"\xC9\xA3"};  // ɣ
    if (ipa.size() >= 1 && ipa[0] == 'x') {
      ipa.replace(0, 1, kGh);
    } else if (ipa.size() >= kPrimaryStressUtf8.size() + 1 &&
               ipa.compare(0, kPrimaryStressUtf8.size(), kPrimaryStressUtf8) == 0 &&
               ipa[kPrimaryStressUtf8.size()] == 'x') {
      ipa.replace(kPrimaryStressUtf8.size(), 1, kGh);
    } else if (ipa.size() >= kSecondaryStressUtf8.size() + 1 &&
               ipa.compare(0, kSecondaryStressUtf8.size(), kSecondaryStressUtf8) == 0 &&
               ipa[kSecondaryStressUtf8.size()] == 'x') {
      ipa.replace(kSecondaryStressUtf8.size(), 1, kGh);
    }
  }
  return ipa;
}

const std::unordered_map<std::string, std::string>& oov_supplement_ipa() {
  static const std::unordered_map<std::string, std::string> m = {
      {"architecten", "\xC9\x91r.xi\xCB\x88t\xC9\x9Bkt\xC9\x99n"},
      {"rijksarchitect", "\xCB\x88r\xC9\x9Biks.\xC9\x91r.xi.\xCB\x88t\xC9\x9Bkt"},
      {"naziheerschappij",
       "\xCB\x88"
       "na\xCB\x90"
       ".tsi\xCB\x90"
       ".he\xCB\x90"
       "r.sx\xC9\x91"
       ".\xCB\x88"
       "p\xC9\x9Bi"},
      {"stedenbouwkundige",
       "st\xCB\x88"
       "e\xCB\x90"
       "d\xC9\x99"
       "nb\xCA\x8C\xCA\x8A"
       "k\xCB\x88"
       "\xC9\xB5"
       "nd\xC9\xAA\xC9\xA3\xC9\x99"},
  };
  return m;
}

const std::unordered_map<std::string, std::string>& function_word_ipa() {
  static const std::unordered_map<std::string, std::string> m = {
      {"de", "d\xC9\x99"},
      {"het", "\xC9\xA6\xC9\x99t"},
      {"een", "\xC9\x99n"},
      {"te", "t\xC9\x99"},
      {"je", "j\xC9\x99"},
      {"ze", "z\xC9\x99"},
      {"we", "\xCA\x8B\xC9\x99"},
      {"me", "m\xC9\x99"},
      {"mijn", "m\xC9\x9Bin"},
      {"zijn", "z\xC9\x9Bin"},
      {"hij", "\xC9\xA6\xC9\x9Bi"},
      {"wij", "\xCA\x8B\xC9\x9Bi"},
      {"jij", "j\xC9\x9Bi"},
  };
  return m;
}

void load_dutch_lexicon_file(const std::filesystem::path& path,
                             std::unordered_map<std::string, std::string>& out_map) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Dutch lexicon: cannot open " + path.generic_string());
  }
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
    const std::string key = normalize_lexicon_key_utf8(surf);
    if (key.empty()) {
      continue;
    }
    bool lower_lemma = true;
    size_t si = 0;
    while (si < surf.size()) {
      char32_t cp = 0;
      size_t adv = 0;
      utf8_decode_at(surf, si, cp, adv);
      if (dutch_unicode_tolower_cp(cp) != cp) {
        lower_lemma = false;
        break;
      }
      si += adv;
    }
    auto it = tmp.find(key);
    if (it == tmp.end()) {
      tmp[key] = Slot{std::move(ipa), lower_lemma};
    } else if (lower_lemma && !it->second.from_lower) {
      it->second.ipa = std::move(ipa);
      it->second.from_lower = true;
    }
  }
  out_map.clear();
  out_map.reserve(tmp.size());
  for (auto& e : tmp) {
    out_map.emplace(std::move(e.first), std::move(e.second.ipa));
  }
}

bool is_vowel_letter32(char32_t c) {
  if (c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u' || c == U'y') {
    return true;
  }
  return c == U'á' || c == U'é' || c == U'í' || c == U'ó' || c == U'ú' || c == U'à' || c == U'è' ||
         c == U'ê' || c == U'ë' || c == U'ï' || c == U'ö' || c == U'ü';
}

char32_t strip_to_plain_vowel(char32_t c) {
  switch (c) {
  case U'á':
  case U'à':
    return U'a';
  case U'é':
  case U'è':
  case U'ê':
  case U'ë':
    return U'e';
  case U'í':
  case U'ï':
    return U'i';
  case U'ó':
  case U'ö':
    return U'o';
  case U'ú':
  case U'ü':
    return U'u';
  default:
    return c;
  }
}

bool word_has_written_stress_u32(const std::u32string& w) {
  for (char32_t c : w) {
    if (c == U'á' || c == U'é' || c == U'í' || c == U'ó' || c == U'ú') {
      return true;
    }
  }
  return false;
}

std::optional<size_t> stressed_syllable_from_acute(const std::vector<std::u32string>& syllables,
                                                   const std::u32string& w) {
  (void)w;
  for (size_t i = 0; i < syllables.size(); ++i) {
    for (char32_t c : syllables[i]) {
      if (c == U'á' || c == U'é' || c == U'í' || c == U'ó' || c == U'ú') {
        return i;
      }
    }
  }
  return std::nullopt;
}

std::vector<std::pair<size_t, size_t>> dutch_vowel_nucleus_spans(const std::u32string& w) {
  static const std::pair<const char32_t*, int> kPat[] = {
      {U"aai", 3}, {U"eeu", 3}, {U"oei", 3}, {U"ieu", 3}, {U"ij", 2}, {U"ei", 2},
      {U"au", 2},  {U"ou", 2},  {U"ui", 2},  {U"eu", 2},  {U"aa", 2}, {U"ee", 2},
      {U"oo", 2},  {U"uu", 2},  {U"oe", 2},  {U"ai", 2},  {U"ie", 2},
  };
  std::vector<std::pair<size_t, size_t>> spans;
  size_t i = 0;
  const size_t n = w.size();
  while (i < n) {
    if (w[i] == U'-') {
      ++i;
      continue;
    }
    if (!is_vowel_letter32(w[i])) {
      ++i;
      continue;
    }
    bool matched = false;
    for (const auto& pr : kPat) {
      const int ln = pr.second;
      if (i + static_cast<size_t>(ln) <= n) {
        bool ok = true;
        for (int k = 0; k < ln; ++k) {
          if (w[i + static_cast<size_t>(k)] != pr.first[k]) {
            ok = false;
            break;
          }
        }
        if (ok) {
          spans.emplace_back(i, i + static_cast<size_t>(ln));
          i += static_cast<size_t>(ln);
          matched = true;
          break;
        }
      }
    }
    if (matched) {
      continue;
    }
    spans.emplace_back(i, i + 1);
    ++i;
  }
  return spans;
}

std::vector<std::u32string> dutch_orthographic_syllables_u32(const std::u32string& word) {
  std::u32string w = word;
  while (!w.empty() && w.front() == U'-') {
    w.erase(w.begin());
  }
  while (!w.empty() && w.back() == U'-') {
    w.pop_back();
  }
  for (size_t z = 0; z < w.size();) {
    if (w[z] == U'-' && z + 1 < w.size() && w[z + 1] == U'-') {
      w.erase(z, 1);
    } else {
      ++z;
    }
  }
  if (w.empty()) {
    return {};
  }
  if (w.find(U'-') != std::u32string::npos) {
    std::vector<std::u32string> parts;
    size_t a = 0;
    while (a <= w.size()) {
      size_t d = w.find(U'-', a);
      const size_t end = d == std::u32string::npos ? w.size() : d;
      if (end > a) {
        std::u32string chunk = w.substr(a, end - a);
        auto sub = dutch_orthographic_syllables_u32(chunk);
        parts.insert(parts.end(), sub.begin(), sub.end());
      }
      if (d == std::u32string::npos) {
        break;
      }
      a = d + 1;
    }
    return parts;
  }
  const auto spans = dutch_vowel_nucleus_spans(w);
  if (spans.empty()) {
    return {w};
  }
  std::vector<std::u32string> syllables;
  const size_t first_s = spans[0].first;
  std::u32string cur = w.substr(0, first_s);
  for (size_t idx = 0; idx < spans.size(); ++idx) {
    const size_t s = spans[idx].first;
    const size_t e = spans[idx].second;
    cur.append(w, s, e - s);
    if (idx + 1 < spans.size()) {
      const size_t next_s = spans[idx + 1].first;
      syllables.push_back(std::move(cur));
      cur = w.substr(e, next_s - e);
    } else {
      cur.append(w, e, w.size() - e);
      syllables.push_back(std::move(cur));
    }
  }
  syllables.erase(std::remove_if(syllables.begin(), syllables.end(),
                                 [](const std::u32string& sy) { return sy.empty(); }),
                  syllables.end());
  return syllables;
}

size_t unstressed_prefix_len_u32(const std::u32string& wl) {
  static const std::u32string prefs[] = {
      U"tegen", U"tussen", U"door", U"voor", U"ver", U"her", U"ont", U"in", U"op", U"af",
      U"uit",   U"aan",    U"be",   U"ge",   U"er",  U"te",
  };
  for (const auto& p : prefs) {
    if (wl.size() > p.size() && wl.compare(0, p.size(), p) == 0 && wl[p.size()] != U'-') {
      return p.size();
    }
  }
  return 0;
}

size_t default_stress_syllable_index(const std::vector<std::u32string>& syls, const std::u32string& w) {
  if (syls.empty()) {
    return 0;
  }
  const size_t n = syls.size();
  if (n == 1) {
    return 0;
  }
  if (word_has_written_stress_u32(w)) {
    if (auto ax = stressed_syllable_from_acute(syls, w)) {
      return *ax;
    }
  }
  std::u32string wl32;
  for (char32_t c : w) {
    if (c != U'-') {
      wl32.push_back(c);
    }
  }
  auto ends_with = [&](std::u32string_view suf) {
    return wl32.size() > suf.size() + 1 && wl32.size() >= suf.size() &&
           wl32.compare(wl32.size() - suf.size(), suf.size(), suf) == 0;
  };
  if (ends_with(U"atie") || ends_with(U"iteit") || ends_with(U"isme") || ends_with(U"eerd") ||
      ends_with(U"eren")) {
    return n - 1;
  }
  const size_t plen = unstressed_prefix_len_u32(wl32);
  if (plen > 0) {
    if (!syls[0].empty() && syls[0][0] == U'g' && syls[0].size() > 2 && syls[0][1] == U'e') {
      return 0;
    }
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

std::string insert_primary_stress_before_vowel_dutch(std::string s) {
  erase_utf8_substr(s, kPrimaryStressUtf8);
  static const char* const kPat[] = {
      "\xC9\x9Bi",       // ɛi
      "\xCA\x8Cu",       // ʌu
      "\xCA\x8Cy",       // ʌy
      "\xC3\xB8\xCB\x90",  // øː
      "a\xC9\xAA\xCC\xAF",  // aɪ̯
      "i\xCB\x90",       // iː
      "e\xCB\x90",       // eː
      "a\xCB\x90",       // aː
      "o\xCB\x90",       // oː
      "u\xCB\x90",       // uː
      "y\xCB\x90",       // yː
      "\xC9\xAA",        // ɪ
      "\xCA\x8F",        // ʏ
      "y",
      "\xC3\xB8",        // ø
      "a",
      "\xC9\x9B",        // ɛ
      "\xC9\x99",        // ə
      "i",
      "o",
      "\xC9\x94",        // ɔ
      "u",
      "\xC9\x91",        // ɑ
  };
  size_t pos = 0;
  while (pos < s.size()) {
    for (const char* pat : kPat) {
      const size_t plen = std::strlen(pat);
      if (s.size() - pos >= plen && s.compare(pos, plen, pat) == 0) {
        return s.substr(0, pos) + kPrimaryStressUtf8 + s.substr(pos);
      }
    }
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, pos, cp, adv);
    pos += adv;
  }
  return kPrimaryStressUtf8 + s;
}

bool ipa_starts_with_nucleus_dutch(std::string_view rest) {
  static const std::string_view kNuc[] = {
      "\xC9\x9Bi",       // ɛi
      "\xCA\x8Cu",       // ʌu
      "\xCA\x8Cy",       // ʌy
      "a\xC9\xAA\xCC\xAF",
      "i\xCB\x90",
      "e\xCB\x90",
      "a\xCB\x90",
      "o\xCB\x90",
      "u\xCB\x90",
      "y\xCB\x90",
      "\xC3\xB8\xCB\x90",
      "\xC5\x8B\xCC\xA9",  // ŋ̩
      "n\xCC\xA9",
      "m\xCC\xA9",
      "l\xCC\xA9",
      "\xC9\x99",
      "\xC9\x9B",
      "\xC9\xAA",
      "\xCA\x8F",
      "y",
      "\xC3\xB8",
      "\xC9\x94",
      "\xC9\x91",
      "a",
      "i",
      "e",
      "o",
      "u",
  };
  for (std::string_view p : kNuc) {
    if (rest.size() >= p.size() && rest.substr(0, p.size()) == p) {
      return true;
    }
  }
  return false;
}

bool ipa_at_stress_mark_dutch(const std::string& ipa, size_t j) {
  if (ipa.size() - j >= kPrimaryStressUtf8.size() &&
      ipa.compare(j, kPrimaryStressUtf8.size(), kPrimaryStressUtf8) == 0) {
    return true;
  }
  return ipa.size() - j >= kSecondaryStressUtf8.size() &&
         ipa.compare(j, kSecondaryStressUtf8.size(), kSecondaryStressUtf8) == 0;
}

size_t ipa_skip_pre_nucleus_dutch(std::string_view s, size_t j) {
  if (j >= s.size()) {
    return j;
  }
  if (s.size() - j >= kPrimaryStressUtf8.size() &&
      s.compare(j, kPrimaryStressUtf8.size(), kPrimaryStressUtf8) == 0) {
    return j;
  }
  if (s.size() - j >= kSecondaryStressUtf8.size() &&
      s.compare(j, kSecondaryStressUtf8.size(), kSecondaryStressUtf8) == 0) {
    return j;
  }
  if (ipa_starts_with_nucleus_dutch(s.substr(j))) {
    return j;
  }
  static const std::string_view kCons[] = {"t\xCD\xA1\xCA\x83", "t\xCA\x83", "t\xCD\xA1s"};
  for (std::string_view p : kCons) {
    if (s.size() - j >= p.size() && s.substr(j, p.size()) == p) {
      return j + p.size();
    }
  }
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

std::string final_devoice_obstruents(std::string ipa) {
  if (ipa.empty()) {
    return ipa;
  }
  const size_t pos = prev_utf8_index(ipa, ipa.size());
  char32_t cp = 0;
  size_t adv = 0;
  utf8_decode_at(ipa, pos, cp, adv);
  std::string rep;
  switch (cp) {
  case U'b':
    rep = "p";
    break;
  case U'd':
    rep = "t";
    break;
  case U'\u0261':
    rep = "k";
    break;
  case U'v':
    rep = "f";
    break;
  case U'z':
    rep = "s";
    break;
  case U'\u0263':
    rep = "x";
    break;
  case U'\u0292':
    rep = "\xCA\x83";
    break;
  default:
    return ipa;
  }
  ipa.replace(pos, adv, rep);
  return ipa;
}

std::string letters_to_ipa_no_stress(const std::u32string& syl_in, const std::u32string& full_word_nh,
                                     const std::u32string& hyphen_word, size_t span_start) {
  std::u32string s = syl_in;
  for (char32_t& c : s) {
    c = dutch_unicode_tolower_cp(c);
  }
  const size_t n = s.size();
  size_t i = 0;
  std::string out;
  auto append = [&](const char* u8) { out += u8; };

  while (i < n) {
    if (s[i] == U'-') {
      ++i;
      continue;
    }
    const char32_t ch = s[i];

    if (i + 3 <= n && s[i] == U's' && s[i + 1] == U'c' && s[i + 2] == U'h') {
      append("sx");
      i += 3;
      continue;
    }
    if (i + 2 <= n && s[i] == U'c' && s[i + 1] == U'h') {
      append("x");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'n' && s[i + 1] == U'g') {
      append("\xC5\x8B");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'n' && s[i + 1] == U'k') {
      append("\xC5\x8Bk");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U's' && s[i + 1] == U'j') {
      append("\xCA\x83");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U't' && s[i + 1] == U'j') {
      append("t\xCA\x83");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'i' && s[i + 1] == U'j') {
      append("\xC9\x9Bi");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'e' && s[i + 1] == U'i') {
      append("\xC9\x9Bi");
      i += 2;
      continue;
    }
    if (i + 3 <= n && s[i] == U'a' && s[i + 1] == U'a' && s[i + 2] == U'i') {
      append("a\xCB\x90i");
      i += 3;
      continue;
    }
    if (i + 3 <= n && s[i] == U'e' && s[i + 1] == U'e' && s[i + 2] == U'u') {
      append("e\xCB\x90\xCA\x8F");
      i += 3;
      continue;
    }
    if (i + 3 <= n && s[i] == U'o' && s[i + 1] == U'e' && s[i + 2] == U'i') {
      append("\xCA\x8Ci");
      i += 3;
      continue;
    }
    if (i + 3 <= n && s[i] == U'i' && s[i + 1] == U'e' && s[i + 2] == U'u') {
      append("\xCA\x8Cu");
      i += 3;
      continue;
    }
    if (i + 2 <= n && ((s[i] == U'a' && s[i + 1] == U'u') || (s[i] == U'o' && s[i + 1] == U'u'))) {
      append("\xCA\x8Cu");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'u' && s[i + 1] == U'i') {
      append("\xCA\x8Cy");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'e' && s[i + 1] == U'u') {
      append("\xC3\xB8\xCB\x90");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'o' && s[i + 1] == U'e') {
      append("u");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'a' && s[i + 1] == U'i') {
      append("a\xC9\xAA\xCC\xAF");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'a' && s[i + 1] == U'a') {
      append("a\xCB\x90");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'e' && s[i + 1] == U'e') {
      append("e\xCB\x90");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'o' && s[i + 1] == U'o') {
      append("o\xCB\x90");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'u' && s[i + 1] == U'u') {
      append("y");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'i' && s[i + 1] == U'e') {
      const bool past_ie = (i + 2 >= n);
      const char32_t nxt = past_ie ? U'\0' : s[i + 2];
      // Python: ``nxt in "tsd"`` is True when *nxt* is ``""`` (empty substring quirk).
      const bool nxt_in_tsd =
          past_ie || nxt == U't' || nxt == U's' || nxt == U'd';
      const bool cond2 = (i + 3 >= n) || !is_vowel_letter32(s[i + 3]);
      if (nxt_in_tsd && cond2) {
        append("i");
      } else {
        append("i\xCB\x90");
      }
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U'q' && s[i + 1] == U'u') {
      append("kv");
      i += 2;
      continue;
    }
    if (ch == U'h') {
      append("\xC9\xA6");
      ++i;
      continue;
    }
    if (ch == U'x') {
      append("ks");
      ++i;
      continue;
    }
    if (ch == U'c' && i + 1 < n) {
      const char32_t v = s[i + 1];
      if (v == U'e' || v == U'i' || v == U'é' || v == U'è' || v == U'ê' || v == U'ë') {
        append("s");
        ++i;
        continue;
      }
    }
    if (ch == U'c') {
      append("k");
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
    if (ch == U'y') {
      const bool pv = (i > 0) && is_vowel_letter32(s[i - 1]);
      const bool nv = (i + 1 < n) && is_vowel_letter32(s[i + 1]);
      append((!pv && nv) ? "j" : "i");
      ++i;
      continue;
    }
    if (ch == U'w') {
      append("\xCA\x8B");
      ++i;
      continue;
    }
    if (ch == U'v') {
      append("v");
      ++i;
      continue;
    }
    if (ch == U'z') {
      append("z");
      ++i;
      continue;
    }
    if (ch == U'g') {
      append("\xC9\xA3");
      ++i;
      continue;
    }
    if (is_vowel_letter32(ch)) {
      const char32_t plain = strip_to_plain_vowel(ch);
      if (ch == U'é') {
        append("e\xCB\x90");
      } else if (ch == U'è' || ch == U'ê') {
        append("\xC9\x9B");
      } else if (ch == U'ë') {
        append("\xC9\x99");
      } else if (ch == U'ï' || ch == U'ü') {
        append("y");
      } else if (ch == U'ö') {
        append("\xC3\xB8");
      } else if (plain == U'a') {
        append("\xC9\x91");
      } else if (plain == U'e') {
        append((i == n - 1) ? "\xC9\x99" : "\xC9\x9B");
      } else if (plain == U'i') {
        append("\xC9\xAA");
      } else if (plain == U'o') {
        append("\xC9\x94");
      } else if (plain == U'u') {
        append("\xCA\x8F");
      } else {
        append("i");
      }
      ++i;
      continue;
    }
    if (ch == U'r') {
      append("r");
      ++i;
      continue;
    }
    if (i + 2 <= n && s[i] == U's' && s[i + 1] == U's') {
      append("s");
      i += 2;
      continue;
    }
    if (ch == U's') {
      const bool prev_v = (i > 0) && is_vowel_letter32(s[i - 1]);
      const bool next_v = (i + 1 < n) && is_vowel_letter32(s[i + 1]);
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
    if (ch == U'b' || ch == U'd' || ch == U'f' || ch == U'k' || ch == U'l' || ch == U'm' || ch == U'n' ||
        ch == U'p' || ch == U't') {
      ++i;
      continue;
    }
    if (i + 2 <= n && s[i] == U'p' && s[i + 1] == U'h') {
      append("f");
      i += 2;
      continue;
    }
    if (i + 2 <= n && s[i] == U't' && s[i + 1] == U'h') {
      append("t");
      i += 2;
      continue;
    }
    ++i;
  }
  (void)full_word_nh;
  (void)hyphen_word;
  (void)span_start;

  std::u32string stem = syl_in;
  for (char32_t& c : stem) {
    c = dutch_unicode_tolower_cp(c);
  }
  while (!stem.empty() && stem.back() == U'-') {
    stem.pop_back();
  }
  static const std::string kGhSuffix{"\xC9\xA3"};
  static const std::string kScriptGSuffix{"\xC9\xA1"};
  if (stem.size() >= 3) {
    const bool lijk = stem.size() >= 5 && stem.substr(stem.size() - 4) == U"lijk";
    if (!lijk && stem[stem.size() - 2] == U'i' && stem.back() == U'g') {
      if (out.size() >= kGhSuffix.size() && out.compare(out.size() - kGhSuffix.size(), kGhSuffix.size(), kGhSuffix) == 0) {
        out.replace(out.size() - kGhSuffix.size(), kGhSuffix.size(), "x");
      } else if (out.size() >= kScriptGSuffix.size() &&
                 out.compare(out.size() - kScriptGSuffix.size(), kScriptGSuffix.size(), kScriptGSuffix) == 0) {
        out.replace(out.size() - kScriptGSuffix.size(), kScriptGSuffix.size(), "x");
      }
    }
  }
  return final_devoice_obstruents(std::move(out));
}

std::u32string strip_hyphens_u32(const std::u32string& w) {
  std::u32string o;
  for (char32_t c : w) {
    if (c != U'-') {
      o.push_back(c);
    }
  }
  return o;
}

std::string rules_word_to_ipa_utf8(const std::string& raw_word, bool with_stress) {
  const std::u32string w = normalize_grapheme_key_u32(raw_word);
  if (w.empty()) {
    return "";
  }
  const std::u32string wl_nh = strip_hyphens_u32(w);
  const auto syls = dutch_orthographic_syllables_u32(w);
  if (syls.empty()) {
    return "";
  }
  const size_t stress_idx =
      with_stress ? default_stress_syllable_index(syls, w) : static_cast<size_t>(-1);
  size_t offset = 0;
  std::string ipa;
  for (size_t idx = 0; idx < syls.size(); ++idx) {
    std::u32string sy_lower = syls[idx];
    for (char32_t& c : sy_lower) {
      c = dutch_unicode_tolower_cp(c);
    }
    std::string chunk = letters_to_ipa_no_stress(sy_lower, wl_nh, w, offset);
    if (with_stress && idx == stress_idx && !chunk.empty()) {
      chunk = insert_primary_stress_before_vowel_dutch(std::move(chunk));
    }
    ipa += chunk;
    offset += syls[idx].size();
  }
  return ipa;
}

}  // namespace

std::filesystem::path resolve_dutch_dict_path(const std::filesystem::path& model_root) {
  const std::filesystem::path under_model = model_root / "nl" / "dict.tsv";
  if (std::filesystem::is_regular_file(under_model)) {
    return under_model;
  }
  return model_root.parent_path() / "data" / "nl" / "dict.tsv";
}

bool dialect_resolves_to_dutch_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "nl" || s == "nl-nl" || s == "dutch";
}

std::vector<std::string> DutchRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first({"nl", "nl-NL", "nl_nl", "dutch"});
}

std::string DutchRuleG2p::normalize_ipa_stress_for_vocoder(std::string ipa) {
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
    while (j < ipa.size() && !ipa_at_stress_mark_dutch(ipa, j) &&
           !ipa_starts_with_nucleus_dutch(std::string_view(ipa).substr(j))) {
      const size_t j2 = ipa_skip_pre_nucleus_dutch(ipa, j);
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

DutchRuleG2p::DutchRuleG2p(std::filesystem::path dict_tsv) : DutchRuleG2p(std::move(dict_tsv), Options{}) {}

DutchRuleG2p::DutchRuleG2p(std::filesystem::path dict_tsv, Options options) : options_(options) {
  load_dutch_lexicon_file(std::move(dict_tsv), lexicon_);
}

std::string DutchRuleG2p::finalize_ipa(std::string ipa, bool from_lexicon) const {
  if (!options_.with_stress) {
    erase_utf8_substr(ipa, kPrimaryStressUtf8);
    erase_utf8_substr(ipa, kSecondaryStressUtf8);
    return ipa;
  }
  if (options_.vocoder_stress && !from_lexicon) {
    return normalize_ipa_stress_for_vocoder(std::move(ipa));
  }
  return ipa;
}

std::string DutchRuleG2p::lookup_or_rules(const std::string& raw_word) const {
  const std::string letters_only = normalize_lexicon_key_utf8(raw_word);
  if (letters_only.empty()) {
    return "";
  }
  auto it = lexicon_.find(letters_only);
  if (it != lexicon_.end()) {
    std::string ipa = apply_lexicon_ipa_postprocess(it->second, letters_only);
    return finalize_ipa(std::move(ipa), true);
  }
  const auto& sup = oov_supplement_ipa();
  auto si = sup.find(letters_only);
  if (si != sup.end()) {
    std::string ipa = apply_lexicon_ipa_postprocess(si->second, letters_only);
    return finalize_ipa(std::move(ipa), true);
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
      merged += apply_lexicon_ipa_postprocess(hc->second, chunks[c]);
    }
    if (ok) {
      return finalize_ipa(std::move(merged), true);
    }
  }
  const auto& fw = function_word_ipa();
  auto fi = fw.find(letters_only);
  if (fi != fw.end()) {
    return finalize_ipa(fi->second, false);
  }
  std::string ipa = rules_word_to_ipa_utf8(raw_word, options_.with_stress);
  return finalize_ipa(std::move(ipa), false);
}

std::string DutchRuleG2p::word_to_ipa(const std::string& word) const {
  const std::string wraw = trim_ascii_ws_copy(word);
  if (wraw.empty()) {
    return "";
  }
  if (options_.expand_cardinal_digits && all_ascii_digits_string(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_dutch_words(wraw);
    if (phrase != wraw) {
      return text_to_ipa_no_expand(phrase, nullptr);
    }
    return wraw;
  }
  if (!options_.expand_cardinal_digits && digit_pass_through_pattern(wraw)) {
    return wraw;
  }
  if (wraw == "Speer") {
    return finalize_ipa("sp\xCB\x88\xC9\xAA\xCB\x90r", false);
  }
  return lookup_or_rules(wraw);
}

std::string DutchRuleG2p::text_to_ipa_no_expand(const std::string& text,
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
    if (is_dutch_word_char(cp)) {
      const size_t start = pos;
      pos += adv;
      while (pos < n) {
        utf8_decode_at(text, pos, cp, adv);
        if (!is_dutch_word_char(cp)) {
          break;
        }
        pos += adv;
      }
      const std::string tok = text.substr(start, pos - start);
      const std::string k = normalize_lexicon_key_utf8(tok);
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
      if (is_dutch_word_char(cp) || cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
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

std::string DutchRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  if (options_.expand_cardinal_digits) {
    return text_to_ipa_no_expand(expand_digit_tokens_in_text(std::move(text)), per_word_log);
  }
  return text_to_ipa_no_expand(text, per_word_log);
}

}  // namespace moonshine_tts
