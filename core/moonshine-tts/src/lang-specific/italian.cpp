#include "italian.h"

#include "g2p-word-log.h"
#include "ipa-symbols.h"
#include "german.h"
#include "utf8-utils.h"

#include <algorithm>
#include <clocale>
#include <cctype>
#include <cstdint>
#include <cwctype>
#include <limits>
#include <fstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace moonshine_tts {
namespace {

const std::string& kPri = ipa::kPrimaryStressUtf8;
const std::string& kSec = ipa::kSecondaryStressUtf8;

using moonshine_tts::erase_utf8_substr;
using moonshine_tts::trim_ascii_ws_copy;
using moonshine_tts::utf8_decode_at;

char32_t italian_tolower_cp(char32_t c) {
  switch (c) {
  case U'À':
    return U'à';
  case U'È':
    return U'è';
  case U'É':
    return U'é';
  case U'Ì':
    return U'ì';
  case U'Í':
    return U'í';
  case U'Î':
    return U'î';
  case U'Ò':
    return U'ò';
  case U'Ó':
    return U'ó';
  case U'Ù':
    return U'ù';
  case U'Ú':
    return U'ú';
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

bool is_italian_lexicon_key_cp(char32_t c) {
  c = italian_tolower_cp(c);
  if (c == U'\u2019') {
    c = U'\'';
  }
  if (c >= U'a' && c <= U'z') {
    return true;
  }
  if (c == U'\'' || c == U'-') {
    return true;
  }
  return c == U'à' || c == U'è' || c == U'é' || c == U'ì' || c == U'í' || c == U'î' || c == U'ò' ||
         c == U'ó' || c == U'ù' || c == U'ú';
}

std::string normalize_lookup_key_utf8(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    if (cp == U'\u2019') {
      cp = U'\'';
    }
    const char32_t cl = italian_tolower_cp(cp);
    if (is_italian_lexicon_key_cp(cl)) {
      utf8_append_codepoint(out, cl);
    }
    i += adv;
  }
  return out;
}

std::string utf8_lowercase_italian(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    utf8_append_codepoint(out, italian_tolower_cp(cp));
    i += adv;
  }
  return out;
}

void load_italian_lexicon_file(const std::filesystem::path& path,
                               std::unordered_map<std::string, std::string>& out_lex) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Italian G2P: cannot read lexicon " + path.generic_string());
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
    const bool lower_ok = (surf == utf8_lowercase_italian(surf));
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

// --- Italian numbers (italian_numbers.py) ---------------------------------

bool is_all_ascii_digits(std::string_view s) {
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

static const char* kDigitWord[10] = {"zero", "uno", "due", "tre", "quattro", "cinque",
                                     "sei", "sette", "otto", "nove"};

std::string under_100(int n) {
  static const char* teens[] = {nullptr,
                                "undici",
                                "dodici",
                                "tredici",
                                "quattordici",
                                "quindici",
                                "sedici",
                                "diciassette",
                                "diciotto",
                                "diciannove"};
  static const char* tens[] = {"", "", "venti", "trenta", "quaranta", "cinquanta",
                               "sessanta", "settanta", "ottanta", "novanta"};
  if (n < 0 || n >= 100) {
    throw std::out_of_range("under_100");
  }
  if (n < 10) {
    return kDigitWord[n];
  }
  if (n == 10) {
    return "dieci";
  }
  if (n < 20) {
    return teens[n - 10];
  }
  const int t = n / 10;
  const int u = n % 10;
  const char* tn = tens[t];
  if (u == 0) {
    return tn;
  }
  std::string stem(tn);
  stem.pop_back();
  if (u == 1) {
    return stem + "uno";
  }
  if (u == 8) {
    return stem + "otto";
  }
  if (u == 3) {
    if (std::string_view(tn).back() == 'i') {
      return stem + "itré";
    }
    return stem + "atré";
  }
  if (std::string_view(tn).back() == 'i') {
    if (u == 6) {
      return stem + "isei";
    }
    if (u == 7) {
      return stem + "isette";
    }
    return stem + "i" + kDigitWord[u];
  }
  if (u == 6) {
    return stem + "asei";
  }
  if (u == 7) {
    return stem + "asette";
  }
  return stem + "a" + kDigitWord[u];
}

std::string hundred_head(int h) {
  static const char* stems[] = {"due", "tre", "quattro", "cinque", "sei", "sette", "otto", "nove"};
  if (h == 1) {
    return "cento";
  }
  if (h < 2 || h > 9) {
    throw std::out_of_range("hundred_head");
  }
  return std::string(stems[h - 2]) + "cento";
}

void append_tokens_0_999(int n, std::vector<std::string>& out) {
  if (n < 0 || n > 999) {
    throw std::out_of_range("tokens_0_999");
  }
  if (n == 0) {
    out.emplace_back("zero");
    return;
  }
  if (n < 100) {
    out.push_back(under_100(n));
    return;
  }
  const int h = n / 100;
  const int r = n % 100;
  if (r == 0) {
    out.push_back(h == 1 ? "cento" : hundred_head(h));
    return;
  }
  if (h == 1) {
    out.emplace_back("cento");
    out.push_back(under_100(r));
    return;
  }
  out.push_back(hundred_head(h));
  out.push_back(under_100(r));
}

std::string spell_1_999_fused(int n) {
  if (n < 1 || n > 999) {
    throw std::out_of_range("spell_1_999_fused");
  }
  if (n < 100) {
    return under_100(n);
  }
  const int h = n / 100;
  const int r = n % 100;
  if (h == 1) {
    if (r == 0) {
      return "cento";
    }
    return "cento" + under_100(r);
  }
  const std::string head = hundred_head(h);
  if (r == 0) {
    return head;
  }
  return head + under_100(r);
}

void append_thousands_multiplier(int q, std::vector<std::string>& out) {
  if (q < 1 || q > 999) {
    throw std::out_of_range("thousands_multiplier");
  }
  if (q == 1) {
    out.emplace_back("mille");
    return;
  }
  if (q < 10) {
    static const char* fused[] = {"duemila", "tremila", "quattromila", "cinquemila", "seimila",
                                    "settemila", "ottomila", "novemila"};
    out.emplace_back(fused[q - 2]);
    return;
  }
  out.push_back(spell_1_999_fused(q) + "mila");
}

void below_1_000_000_tokens(int n, std::vector<std::string>& out) {
  if (n < 0 || n >= 1'000'000) {
    throw std::out_of_range("below_1_000_000");
  }
  if (n < 1000) {
    append_tokens_0_999(n, out);
    return;
  }
  const int q = n / 1000;
  const int r = n % 1000;
  append_thousands_multiplier(q, out);
  if (r != 0) {
    append_tokens_0_999(r, out);
  }
}

std::string expand_cardinal_digits_to_italian_words(std::string_view s) {
  if (!is_all_ascii_digits(s)) {
    return std::string(s);
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
    return "zero";
  }
  std::vector<std::string> parts;
  below_1_000_000_tokens(static_cast<int>(n), parts);
  std::string o;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      o.push_back(' ');
    }
    o += parts[i];
  }
  return o;
}

std::string expand_digit_tokens_in_text(std::string text) {
  static const std::regex range_re(R"(\b(\d+)-(\d+)\b)", std::regex::ECMAScript);
  static const std::regex dig_re(R"(\b\d+\b)", std::regex::ECMAScript);
  std::string out;
  std::sregex_iterator it(text.begin(), text.end(), range_re);
  std::sregex_iterator end;
  size_t pos = 0;
  for (; it != end; ++it) {
    const std::smatch& m = *it;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    out += expand_cardinal_digits_to_italian_words(m.str(1));
    out += " - ";
    out += expand_cardinal_digits_to_italian_words(m.str(2));
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
    out += expand_cardinal_digits_to_italian_words(m.str());
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

// --- Rules (UTF-32 syllable pass) -----------------------------------------

std::u32string utf8_to_u32(const std::string& s) {
  std::u32string o;
  size_t i = 0;
  while (i < s.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, i, cp, adv);
    o.push_back(cp);
    i += adv;
  }
  return o;
}

std::string u32_to_utf8(const std::u32string& s) {
  std::string o;
  for (char32_t c : s) {
    utf8_append_codepoint(o, c);
  }
  return o;
}

bool is_vowel_ch(char32_t c) {
  c = italian_tolower_cp(c);
  return c == U'a' || c == U'e' || c == U'i' || c == U'o' || c == U'u' || c == U'à' ||
         c == U'è' || c == U'é' || c == U'ì' || c == U'í' || c == U'î' || c == U'ò' || c == U'ó' ||
         c == U'ù' || c == U'ú';
}

char32_t strip_accent_letter(char32_t c) {
  c = italian_tolower_cp(c);
  switch (c) {
  case U'à':
  case U'á':
    return U'a';
  case U'è':
  case U'é':
  case U'ê':
    return U'e';
  case U'ì':
  case U'í':
  case U'î':
    return U'i';
  case U'ò':
  case U'ó':
    return U'o';
  case U'ù':
  case U'ú':
    return U'u';
  default:
    return c;
  }
}

bool should_hiatus_it(char32_t a, char32_t b) {
  const char32_t al = italian_tolower_cp(a);
  const char32_t bl = italian_tolower_cp(b);
  if (al == U'í' || al == U'ì' || bl == U'í' || bl == U'ì') {
    return true;
  }
  if (al == U'ú' || al == U'ù' || bl == U'ú' || bl == U'ù') {
    return true;
  }
  const char32_t ba = strip_accent_letter(al);
  const char32_t bb = strip_accent_letter(bl);
  if ((ba == U'a' || ba == U'e' || ba == U'o') && (bb == U'a' || bb == U'e' || bb == U'o')) {
    return true;
  }
  if ((ba == U'i' || ba == U'u') && (bb == U'a' || bb == U'e' || bb == U'o')) {
    return false;
  }
  if ((ba == U'a' || ba == U'e' || ba == U'o') && (bb == U'i' || bb == U'u')) {
    return false;
  }
  if (ba == bb) {
    return true;
  }
  if ((ba == U'i' || ba == U'u') && (bb == U'i' || bb == U'u')) {
    return false;
  }
  return true;
}

void vowel_nucleus_spans(const std::u32string& w, std::vector<std::pair<size_t, size_t>>& spans) {
  spans.clear();
  size_t i = 0;
  const size_t n = w.size();
  while (i < n) {
    if (!is_vowel_ch(w[i])) {
      ++i;
      continue;
    }
    if (i + 1 < n && is_vowel_ch(w[i + 1])) {
      if (should_hiatus_it(w[i], w[i + 1])) {
        spans.emplace_back(i, i + 1);
        ++i;
      } else {
        spans.emplace_back(i, i + 2);
        i += 2;
      }
    } else {
      spans.emplace_back(i, i + 1);
      ++i;
    }
  }
}

bool valid_onset2(char a, char b) {
  static const char* pairs[] = {"bl", "br", "cl", "cr", "dr", "fl", "fr", "gl", "gr",
                                "pl", "pr", "tr", "ch"};
  const char buf[3] = {static_cast<char>(std::tolower(static_cast<unsigned char>(a))),
                       static_cast<char>(std::tolower(static_cast<unsigned char>(b))), 0};
  for (const char* p : pairs) {
    if (buf[0] == p[0] && buf[1] == p[1]) {
      return true;
    }
  }
  return false;
}

void split_intervocalic_cluster(const std::string& cluster, std::string& coda, std::string& onset) {
  coda.clear();
  onset.clear();
  if (cluster.empty()) {
    return;
  }
  if (cluster.size() >= 2 &&
      valid_onset2(cluster[cluster.size() - 2], cluster[cluster.size() - 1])) {
    if (cluster.size() == 2) {
      onset = cluster;
      return;
    }
    coda = cluster.substr(0, cluster.size() - 2);
    onset = cluster.substr(cluster.size() - 2);
    return;
  }
  coda = cluster.substr(0, cluster.size() - 1);
  onset = cluster.substr(cluster.size() - 1);
}

std::vector<std::u32string> italian_orthographic_syllables_u32(std::u32string w) {
  for (char32_t& c : w) {
    c = italian_tolower_cp(c);
  }
  std::u32string filt;
  for (char32_t c : w) {
    if (c == U'-' || (c >= U'a' && c <= U'z') || c == U'à' || c == U'è' || c == U'é' ||
        c == U'ì' || c == U'í' || c == U'î' || c == U'ò' || c == U'ó' || c == U'ù' || c == U'ú') {
      filt.push_back(c);
    }
  }
  w = std::move(filt);
  if (w.empty()) {
    return {};
  }
  if (w.find(U'-') != std::u32string::npos) {
    std::vector<std::u32string> acc;
    size_t st = 0;
    while (st <= w.size()) {
      size_t d = w.find(U'-', st);
      if (d == std::u32string::npos) {
        if (st < w.size()) {
          auto sub = w.substr(st);
          auto part = italian_orthographic_syllables_u32(sub);
          acc.insert(acc.end(), part.begin(), part.end());
        }
        break;
      }
      if (d > st) {
        auto part = italian_orthographic_syllables_u32(w.substr(st, d - st));
        acc.insert(acc.end(), part.begin(), part.end());
      }
      st = d + 1;
    }
    return acc;
  }
  std::vector<std::pair<size_t, size_t>> spans;
  vowel_nucleus_spans(w, spans);
  if (spans.empty()) {
    return {w};
  }
  std::vector<std::u32string> syllables;
  const size_t first_s = spans[0].first;
  std::u32string cur = w.substr(0, first_s);
  for (size_t idx = 0; idx < spans.size(); ++idx) {
    const size_t s = spans[idx].first;
    const size_t e = spans[idx].second;
    cur.append(w.substr(s, e - s));
    if (idx + 1 < spans.size()) {
      const size_t next_s = spans[idx + 1].first;
      const std::u32string cluster_u = w.substr(e, next_s - e);
      const std::string cluster = u32_to_utf8(cluster_u);
      std::string coda;
      std::string onset;
      split_intervocalic_cluster(cluster, coda, onset);
      cur.append(utf8_to_u32(coda));
      syllables.push_back(cur);
      cur = utf8_to_u32(onset);
    } else {
      cur.append(w.substr(e));
      syllables.push_back(cur);
    }
  }
  std::vector<std::u32string> non_empty;
  for (auto& sy : syllables) {
    if (!sy.empty()) {
      non_empty.push_back(std::move(sy));
    }
  }
  return non_empty;
}

bool accented_vowel_in_u32(char32_t c) {
  c = italian_tolower_cp(c);
  return c == U'à' || c == U'è' || c == U'é' || c == U'ì' || c == U'í' || c == U'ò' || c == U'ó' ||
         c == U'ù' || c == U'ú' || c == U'î';
}

size_t default_stressed_syllable_index(const std::vector<std::u32string>& syls,
                                       const std::u32string& word_lower) {
  if (syls.empty()) {
    return 0;
  }
  std::u32string w;
  for (char32_t c : word_lower) {
    c = italian_tolower_cp(c);
    if (c == U'-' || (c >= U'a' && c <= U'z') || accented_vowel_in_u32(c)) {
      w.push_back(c);
    }
  }
  for (char32_t c : w) {
    if (accented_vowel_in_u32(c)) {
      for (size_t i = 0; i < syls.size(); ++i) {
        for (char32_t sc : syls[i]) {
          if (accented_vowel_in_u32(sc)) {
            return i;
          }
        }
      }
    }
  }
  const size_t n = syls.size();
  if (n == 1) {
    return 0;
  }
  std::u32string tail = w;
  while (!tail.empty() && tail.back() == U'-') {
    tail.pop_back();
  }
  if (tail.empty()) {
    return 0;
  }
  const char32_t last = strip_accent_letter(tail.back());
  if (last == U'a' || last == U'e' || last == U'i' || last == U'o' || last == U'u') {
    return n >= 2 ? n - 2 : 0;
  }
  return n - 1;
}

std::string insert_primary_stress_before_vowel(std::string ipa) {
  size_t p = 0;
  while (p < ipa.size()) {
    if (ipa.compare(p, kPri.size(), kPri) == 0) {
      ipa.erase(p, kPri.size());
      continue;
    }
    if (ipa.compare(p, kSec.size(), kSec) == 0) {
      ipa.erase(p, kSec.size());
      continue;
    }
    ++p;
  }
  for (size_t i = 0; i < ipa.size();) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(ipa, i, cp, adv);
    if (cp == U'a' || cp == U'e' || cp == U'i' || cp == U'o' || cp == U'u' || cp == U'ɛ' ||
        cp == U'ɔ') {
      ipa.insert(i, kPri);
      return ipa;
    }
    i += adv;
  }
  return kPri + ipa;
}

bool next_is_vowel_u32(const std::u32string& s, size_t j) {
  size_t k = j;
  while (k < s.size()) {
    if (s[k] == U'h') {
      ++k;
      continue;
    }
    return is_vowel_ch(s[k]);
  }
  return false;
}

bool ei_e_accent(char32_t c) {
  c = italian_tolower_cp(c);
  return c == U'e' || c == U'é' || c == U'è';
}

// After c/g (and related digraphs): letters that palatalize, matching Python ``in "eiéè"``
// (the character set is e, i, é, è — e.g. syllable "ci" → /t͡ʃ/).
bool italian_cg_palatal_letter(char32_t c) {
  c = italian_tolower_cp(c);
  return c == U'e' || c == U'é' || c == U'è' || c == U'i' || c == U'ì' || c == U'í' || c == U'î';
}

std::string letters_to_ipa_no_stress(const std::u32string& su) {
  std::string out;
  const size_t n = su.size();
  size_t i = 0;
  while (i < n) {
    if (su[i] == U'-') {
      ++i;
      continue;
    }
    if (i + 1 < n && su[i] == U'z' && su[i + 1] == U'z') {
      out.append("tt");
      out.append("\xCD\xA1");  // U+0361 COMBINING DOUBLE INVERTED BREVE
      out.push_back('s');
      i += 2;
      continue;
    }
    if (i + 2 < n && su[i] == U'c' && su[i + 1] == U'c' && italian_cg_palatal_letter(su[i + 2])) {
      out.append("tt\xCD\xA1\xCA\x83");  // tt͡ʃ
      i += 3;
      continue;
    }
    if (i + 2 < n && su[i] == U'g' && su[i + 1] == U'g' && italian_cg_palatal_letter(su[i + 2])) {
      out.append("dd\xCD\xA1\xCA\x92");  // dd͡ʒ
      i += 3;
      continue;
    }
    if (i + 1 < n && su[i] == U'g' && su[i + 1] == U'n') {
      out += "\xC9\xB2\xC9\xB2";  // ɲɲ
      i += 2;
      continue;
    }
    if (i + 2 < n && su[i] == U'g' && su[i + 1] == U'l' && italian_tolower_cp(su[i + 2]) == U'i') {
      const char32_t nxt = (i + 3 < n) ? italian_tolower_cp(su[i + 3]) : U'\0';
      const std::u32string vow_after = U"aeiouàèéìòóù";
      if (nxt == U'\0' || vow_after.find(nxt) != std::u32string::npos) {
        out += "\xCA\x8E";  // ʎ
        i += 3;
        continue;
      }
      if (nxt == U'i' && (i + 4 >= n || !is_vowel_ch(su[i + 4]))) {
        out += "\xCA\x8E";
        i += 3;
        continue;
      }
    }
    if (i + 1 < n && su[i] == U'c' && su[i + 1] == U'h') {
      out.push_back('k');
      i += 2;
      continue;
    }
    if (i + 2 < n && su[i] == U'g' && su[i + 1] == U'h' && italian_cg_palatal_letter(su[i + 2])) {
      out += "\xC9\xA1";  // ɡ
      i += 3;
      continue;
    }
    if (i + 2 < n && su[i] == U's' && su[i + 1] == U'c' && italian_cg_palatal_letter(su[i + 2])) {
      out += "\xCA\x83";  // ʃ
      i += 3;
      continue;
    }
    if (i + 2 < n && su[i] == U's' && su[i + 1] == U'c') {
      const char32_t t = italian_tolower_cp(su[i + 2]);
      if (t == U'a' || t == U'o' || t == U'u' || t == U'à' || t == U'ò' || t == U'ù') {
        out += "sk";
        i += 3;
        continue;
      }
    }
    if (i + 1 < n && su[i] == U'q' && su[i + 1] == U'u') {
      out += "kw";
      i += 2;
      continue;
    }
    if (i + 2 < n && su[i] == U'g' && su[i + 1] == U'u' && italian_cg_palatal_letter(su[i + 2])) {
      out += "\xC9\xA1";
      i += 2;
      continue;
    }
    if (i + 2 < n && su[i] == U'c' && su[i + 1] == U'i' && is_vowel_ch(su[i + 2])) {
      out.append("t\xCD\xA1\xCA\x83");
      i += 2;
      continue;
    }
    if (i + 2 < n && su[i] == U'g' && su[i + 1] == U'i' && is_vowel_ch(su[i + 2])) {
      out.append("d\xCD\xA1\xCA\x92");
      i += 2;
      continue;
    }
    if (su[i] == U'c' && i + 1 < n && italian_cg_palatal_letter(su[i + 1])) {
      out.append("t\xCD\xA1\xCA\x83");
      i += 2;
      continue;
    }
    if (su[i] == U'g' && i + 1 < n && italian_cg_palatal_letter(su[i + 1])) {
      out.append("d\xCD\xA1\xCA\x92");
      i += 2;
      continue;
    }
    char32_t ch = italian_tolower_cp(su[i]);
    if (ch == U'h') {
      ++i;
      continue;
    }
    if (i + 1 < n && ch == italian_tolower_cp(su[i + 1]) && ch != U'a' && ch != U'e' && ch != U'i' &&
        ch != U'o' && ch != U'u' && ch != U'à' && ch != U'è' && ch != U'é' && ch != U'ì' &&
        ch != U'í' && ch != U'î' && ch != U'ò' && ch != U'ó' && ch != U'ù' && ch != U'ú') {
      const char32_t cl = ch;
      if (cl == U'b' || cl == U'c' || cl == U'd' || cl == U'f' || cl == U'g' || cl == U'l' ||
          cl == U'm' || cl == U'n' || cl == U'p' || cl == U's' || cl == U't' || cl == U'v') {
        std::string bit;
        utf8_append_codepoint(bit, cl);
        out += bit;
        out += bit;
      } else {
        std::string bit;
        utf8_append_codepoint(bit, cl);
        out += bit;
      }
      i += 2;
      continue;
    }
    if (ch == U'c') {
      out.push_back('k');
      ++i;
      continue;
    }
    if (ch == U'g') {
      out += "\xC9\xA1";
      ++i;
      continue;
    }
    if (ch == U'q') {
      if (i + 1 < n && su[i + 1] == U'u' && next_is_vowel_u32(su, i + 2)) {
        out.push_back('k');
        i += 2;
        continue;
      }
      out.push_back('k');
      ++i;
      continue;
    }
    if (ch == U's') {
      const bool prev_v = (i > 0 && is_vowel_ch(su[i - 1]));
      const bool next_v = (i + 1 < n && next_is_vowel_u32(su, i + 1));
      out.push_back((prev_v && next_v) ? 'z' : 's');
      ++i;
      continue;
    }
    if (ch == U'z') {
      const bool prev_v = (i > 0 && is_vowel_ch(su[i - 1]));
      const bool next_v = (i + 1 < n && next_is_vowel_u32(su, i + 1));
      if (prev_v && next_v) {
        out.append("d\xCD\xA1\xCA\x92");
      } else {
        out.append("t\xCD\xA1s");
      }
      ++i;
      continue;
    }
    if (ch == U'x') {
      out += "ks";
      ++i;
      continue;
    }
    if (ch == U'j') {
      out.push_back('j');
      ++i;
      continue;
    }
    if (ch == U'w') {
      out.push_back('w');
      ++i;
      continue;
    }
    if (ch == U'k') {
      out.push_back('k');
      ++i;
      continue;
    }
    if (is_vowel_ch(su[i])) {
      const char32_t cl = italian_tolower_cp(su[i]);
      if (i + 1 < n && is_vowel_ch(su[i + 1])) {
        const char32_t nxt = italian_tolower_cp(su[i + 1]);
        if ((cl == U'a' || cl == U'à') && (nxt == U'u' || nxt == U'ù' || nxt == U'ú')) {
          out += "aw";
          i += 2;
          continue;
        }
        if ((cl == U'a' || cl == U'à') && (nxt == U'i' || nxt == U'ì' || nxt == U'í')) {
          out += "aj";
          i += 2;
          continue;
        }
        if (ei_e_accent(su[i]) && (nxt == U'i' || nxt == U'ì' || nxt == U'í')) {
          out += "ej";
          i += 2;
          continue;
        }
        if ((cl == U'o' || cl == U'ó' || cl == U'ò') && (nxt == U'i' || nxt == U'ì' || nxt == U'í')) {
          out += "oj";
          i += 2;
          continue;
        }
        if (ei_e_accent(su[i]) && (nxt == U'u' || nxt == U'ù' || nxt == U'ú')) {
          out += "\xC9\x9Bw";  // ɛw
          i += 2;
          continue;
        }
        if ((cl == U'o' || cl == U'ó' || cl == U'ò') && (nxt == U'u' || nxt == U'ù' || nxt == U'ú')) {
          out += "ow";
          i += 2;
          continue;
        }
      }
      if (cl == U'a' || cl == U'à') {
        out.push_back('a');
      } else if (cl == U'e' || cl == U'é') {
        out.push_back('e');
      } else if (cl == U'è' || cl == U'ê') {
        out += "\xC9\x9B";
      } else if (cl == U'i' || cl == U'í' || cl == U'ì' || cl == U'î') {
        out.push_back('i');
      } else if (cl == U'o' || cl == U'ó') {
        out.push_back('o');
      } else if (cl == U'ò') {
        out += "\xC9\x94";
      } else if (cl == U'u' || cl == U'ù' || cl == U'ú') {
        out.push_back('u');
      } else {
        out.push_back('a');
      }
      ++i;
      continue;
    }
    if (ch == U'b' || ch == U'd' || ch == U'f' || ch == U'l' || ch == U'm' || ch == U'n' ||
        ch == U'p' || ch == U'r' || ch == U't' || ch == U'v') {
      std::string bit;
      utf8_append_codepoint(bit, ch);
      out += bit;
      ++i;
      continue;
    }
    ++i;
  }
  return out;
}

std::string rules_word_to_ipa_utf8(const std::string& raw, bool with_stress) {
  std::u32string w;
  size_t ii = 0;
  while (ii < raw.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(raw, ii, cp, adv);
    const char32_t cl = italian_tolower_cp(cp);
    if (cl == U'-' || (cl >= U'a' && cl <= U'z') || cl == U'à' || cl == U'è' || cl == U'é' ||
        cl == U'ì' || cl == U'í' || cl == U'î' || cl == U'ò' || cl == U'ó' || cl == U'ù' ||
        cl == U'ú' || (cl >= U'A' && cl <= U'Z')) {
      w.push_back(cl);
    }
    ii += adv;
  }
  if (w.empty()) {
    return "";
  }
  auto syls = italian_orthographic_syllables_u32(w);
  if (syls.empty()) {
    return "";
  }
  const size_t stress_idx =
      with_stress ? default_stressed_syllable_index(syls, w) : static_cast<size_t>(-1);
  std::string acc;
  for (size_t idx = 0; idx < syls.size(); ++idx) {
    std::string chunk = letters_to_ipa_no_stress(syls[idx]);
    if (with_stress && idx == stress_idx && !chunk.empty()) {
      chunk = insert_primary_stress_before_vowel(std::move(chunk));
    }
    acc += chunk;
  }
  return acc;
}

const std::unordered_map<std::string, std::string>& function_word_ipa() {
  static const std::unordered_map<std::string, std::string> m = {
      {"e", "e"},       {"ed", "ed"},     {"o", "o"},       {"a", "a"},
      {"i", "i"},       {"il", "il"},     {"lo", "lo"},     {"la", "la"},
      {"le", "le"},     {"gli", "\xCA\x8Ei"},  {"un", "un"},
      {"uno", kPri + "uno"}, {"una", kPri + "una"}, {"di", "di"}, {"da", "da"},
      {"in", "in"},     {"su", "su"},     {"per", "per"},   {"tra", "tra"},
      {"fra", "fra"},   {"del", "del"},   {"della", kPri + "d\xC9\x9Blla"},
      {"delle", kPri + "d\xC9\x9Blle"}, {"dei", kPri + "dei"},
      {"degli", kPri + "de\xCA\x8E\xCA\x8Ei"}, {"al", "al"},
      {"allo", kPri + "allo"}, {"alla", kPri + "alla"}, {"ai", "ai"},
      {"agli", kPri + "a\xCA\x8E\xCA\x8Ei"}, {"alle", kPri + "alle"},
      {"nel", "nel"},   {"nello", kPri + "n\xC9\x9Bllo"}, {"nella", kPri + "n\xC9\x9Blla"},
      {"nell", "n\xC9\x9Bll"}, {"sul", "sul"}, {"sullo", kPri + "sullo"},
      {"sulla", kPri + "sulla"}, {"col", "kol"}, {"coi", kPri + "koi"},
      {"ci", "t\xCD\xA1\xCA\x83i"}, {"vi", "vi"}, {"si", "si"}, {"ti", "ti"},
      {"mi", "mi"},     {"non", "non"},   {"che", "ke"}};
  return m;
}

bool is_italian_word_char(char32_t cp) {
  if (cp == U'-' || cp == U'\'' || cp == U'\u2019') {
    return true;
  }
  if (cp >= U'0' && cp <= U'9') {
    return true;
  }
  if (cp == U'_') {
    return true;
  }
  if (cp <= 0x7Fu) {
    return std::isalpha(static_cast<unsigned char>(cp)) != 0;
  }
  // Match Python ``_ITALIAN_WORD_PATTERN`` (``[\w\-]+`` with re.UNICODE): letters + numbers.
  static const bool utf8_ctype = []() {
    return std::setlocale(LC_CTYPE, "C.UTF-8") != nullptr ||
           std::setlocale(LC_CTYPE, "en_US.UTF-8") != nullptr;
  }();
  if (utf8_ctype && cp <= static_cast<char32_t>(std::numeric_limits<wint_t>::max())) {
    const wint_t w = static_cast<wint_t>(cp);
    if (std::iswalpha(w) != 0 || std::iswdigit(w) != 0) {
      return true;
    }
  }
  cp = italian_tolower_cp(cp);
  if (cp >= U'a' && cp <= U'z') {
    return true;
  }
  return cp == U'à' || cp == U'è' || cp == U'é' || cp == U'ì' || cp == U'í' || cp == U'î' ||
         cp == U'ò' || cp == U'ó' || cp == U'ù' || cp == U'ú';
}

bool try_consume_italian_word(const std::string& text, size_t pos, size_t& out_end) {
  if (pos >= text.size()) {
    return false;
  }
  char32_t c0 = 0;
  size_t a0 = 0;
  utf8_decode_at(text, pos, c0, a0);
  if (c0 == U'\'' || c0 == U'\u2019') {
    if (pos + a0 >= text.size()) {
      return false;
    }
    char32_t c1 = 0;
    size_t a1 = 0;
    utf8_decode_at(text, pos + a0, c1, a1);
    if (!is_italian_word_char(c1) || c1 == U'\'' || c1 == U'\u2019') {
      return false;
    }
  } else if (!is_italian_word_char(c0) || c0 == U'\'' || c0 == U'\u2019') {
    return false;
  }
  size_t p = pos;
  while (p < text.size()) {
    char32_t c = 0;
    size_t a = 0;
    utf8_decode_at(text, p, c, a);
    if (is_italian_word_char(c)) {
      p += a;
      continue;
    }
    if (c == U'\'' || c == U'\u2019') {
      if (p + a >= text.size()) {
        p += a;
        break;
      }
      char32_t c2 = 0;
      size_t a2 = 0;
      utf8_decode_at(text, p + a, c2, a2);
      if (is_italian_word_char(c2) && c2 != U'\'' && c2 != U'\u2019') {
        p += a + a2;
        continue;
      }
      p += a;
      break;
    }
    break;
  }
  out_end = p;
  return out_end > pos;
}

}  // namespace

ItalianRuleG2p::ItalianRuleG2p(std::filesystem::path dict_tsv)
    : ItalianRuleG2p(std::move(dict_tsv), Options{}) {}

ItalianRuleG2p::ItalianRuleG2p(std::filesystem::path dict_tsv, Options options)
    : options_(options) {
  load_italian_lexicon_file(std::move(dict_tsv), lexicon_);
}

std::string ItalianRuleG2p::finalize_ipa(std::string ipa, bool from_lexicon) const {
  if (!options_.with_stress) {
    erase_utf8_substr(ipa, kPri);
    erase_utf8_substr(ipa, kSec);
    return ipa;
  }
  if (options_.vocoder_stress && !from_lexicon) {
    return GermanRuleG2p::normalize_ipa_stress_for_vocoder(std::move(ipa));
  }
  return ipa;
}

std::string ItalianRuleG2p::lookup_or_rules(const std::string& raw_word) const {
  const std::string letters_only = normalize_lookup_key_utf8(raw_word);
  if (letters_only.empty()) {
    return "";
  }
  auto it = lexicon_.find(letters_only);
  if (it != lexicon_.end()) {
    return finalize_ipa(it->second, true);
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
      return finalize_ipa(std::move(merged), true);
    }
  }
  const auto& fw = function_word_ipa();
  auto fi = fw.find(letters_only);
  if (fi != fw.end()) {
    return finalize_ipa(fi->second, false);
  }
  return finalize_ipa(rules_word_to_ipa_utf8(raw_word, options_.with_stress), false);
}

std::string ItalianRuleG2p::word_to_ipa(const std::string& word) const {
  const std::string wraw = trim_ascii_ws_copy(word);
  if (wraw.empty()) {
    return "";
  }
  if (options_.expand_cardinal_digits && is_all_ascii_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_italian_words(wraw);
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

std::string ItalianRuleG2p::text_to_ipa_no_expand(const std::string& text,
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
    if (try_consume_italian_word(text, pos, wend)) {
      const std::string tok = text.substr(pos, wend - pos);
      const std::string k = normalize_lookup_key_utf8(tok);
      std::string wipa = word_to_ipa(tok);
      if (per_word_log != nullptr) {
        per_word_log->push_back(G2pWordLog{tok, k, G2pWordPath::kRuleBasedG2p, wipa});
        out += per_word_log->back().ipa;
      } else {
        out += wipa;
      }
      pos = wend;
      continue;
    }
    const size_t start = pos;
    pos += adv;
    while (pos < n) {
      utf8_decode_at(text, pos, cp, adv);
      size_t tmp = 0;
      if (try_consume_italian_word(text, pos, tmp) || cp == U' ' || cp == U'\t' || cp == U'\n' ||
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

std::string ItalianRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  if (options_.expand_cardinal_digits) {
    text = expand_digit_tokens_in_text(std::move(text));
  }
  return text_to_ipa_no_expand(text, per_word_log);
}

bool dialect_resolves_to_italian_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "it" || s == "it-it" || s == "italian";
}

std::vector<std::string> ItalianRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first({"it", "it-IT", "it_it", "italian"});
}

std::filesystem::path resolve_italian_dict_path(const std::filesystem::path& model_root) {
  const std::filesystem::path under_data = model_root.parent_path() / "data" / "it" / "dict.tsv";
  if (std::filesystem::is_regular_file(under_data)) {
    return under_data;
  }
  return model_root / "it" / "dict.tsv";
}

}  // namespace moonshine_tts
