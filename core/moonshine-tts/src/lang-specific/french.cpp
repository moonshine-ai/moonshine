#include "french.h"
#include "french-compound-map.h"

#include "french-internal.h"
#include "g2p-word-log.h"
#include "ipa-symbols.h"
#include "utf8-utils.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <fstream>
#include <istream>
#include <sstream>
#include <regex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moonshine_tts {
namespace {

using ipa::kPrimaryStressUtf8;
using ipa::kSecondaryStressUtf8;
using moonshine_tts::trim_ascii_ws_copy;
using moonshine_tts::utf8_decode_at;

char32_t french_tolower_cp(char32_t c) {
  switch (c) {
  case U'À':
    return U'à';
  case U'Â':
    return U'â';
  case U'Ä':
    return U'ä';
  case U'É':
    return U'é';
  case U'È':
    return U'è';
  case U'Ê':
    return U'ê';
  case U'Ë':
    return U'ë';
  case U'Î':
    return U'î';
  case U'Ï':
    return U'ï';
  case U'Ô':
    return U'ô';
  case U'Ö':
    return U'ö';
  case U'Ù':
    return U'ù';
  case U'Û':
    return U'û';
  case U'Ü':
    return U'ü';
  case U'Ÿ':
    return U'ÿ';
  case U'Ç':
    return U'ç';
  case U'Œ':
    return U'œ';
  case U'Æ':
    return U'æ';
  default:
    break;
  }
  if (c >= U'A' && c <= U'Z') {
    return c + 32;
  }
  // Latin-1 uppercase letters (Python ``str.lower``); skip × (U+00D7) and ß (lowercase only).
  if ((c >= U'\u00C0' && c <= U'\u00D6') || (c >= U'\u00D8' && c <= U'\u00DE')) {
    return c + 32;
  }
  return c;
}

bool is_french_key_cp(char32_t c) {
  c = french_tolower_cp(c);
  if (c >= U'a' && c <= U'z') {
    return true;
  }
  return c == U'à' || c == U'â' || c == U'ä' || c == U'é' || c == U'è' || c == U'ê' || c == U'ë' ||
         c == U'ï' || c == U'î' || c == U'ô' || c == U'ù' || c == U'û' || c == U'ü' || c == U'ÿ' ||
         c == U'ç' || c == U'œ' || c == U'æ' || c == U'\'' || c == U'-';
}

std::string normalize_lookup_key_utf8(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    const char32_t cl = french_tolower_cp(cp);
    if (is_french_key_cp(cl)) {
      utf8_append_codepoint(out, cl);
    }
    i += adv;
  }
  return out;
}

// Python ``re.UNICODE`` word chars in U+00AA..U+00FF (excludes × U+00D7 and ÷ U+00F7).
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

// Letterlike math symbols that are ``\w`` in Python 3 (e.g. U+2102 ℂ) but never appear in French lookup keys.
constexpr std::array<char32_t, 46> kLetterlikeWordChars = {
    U'\u2102', U'\u2107', U'\u210A', U'\u210B', U'\u210C', U'\u210D', U'\u210E', U'\u210F', U'\u2110', U'\u2111',
    U'\u2112', U'\u2113', U'\u2115', U'\u2119', U'\u211A', U'\u211B', U'\u211C', U'\u211D', U'\u2124', U'\u2126',
    U'\u2128', U'\u212A', U'\u212B', U'\u212C', U'\u212D', U'\u212F', U'\u2130', U'\u2131', U'\u2132', U'\u2133',
    U'\u2134', U'\u2135', U'\u2136', U'\u2137', U'\u2138', U'\u2139', U'\u213C', U'\u213D', U'\u213E', U'\u213F',
    U'\u2145', U'\u2146', U'\u2147', U'\u2148', U'\u2149', U'\u214E'};

bool is_letterlike_math_word_char(char32_t cp) {
  return std::binary_search(kLetterlikeWordChars.begin(), kLetterlikeWordChars.end(), cp);
}

bool is_french_word_char(char32_t cp) {
  // Match Python ``[\w'-]+``: only ASCII apostrophe is in-word; U+2019 is punctuation (typographic ’).
  if (cp == U'-' || cp == U'\'') {
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
  const char32_t cl = french_tolower_cp(cp);
  if (cl >= U'a' && cl <= U'z') {
    return true;
  }
  return cl == U'à' || cl == U'â' || cl == U'ä' || cl == U'é' || cl == U'è' || cl == U'ê' || cl == U'ë' ||
         cl == U'ï' || cl == U'î' || cl == U'ô' || cl == U'ù' || cl == U'û' || cl == U'ü' || cl == U'ÿ' ||
         cl == U'ç' || cl == U'œ' || cl == U'æ';
}

std::string to_lower_ascii(std::string_view w) {
  std::string s(w);
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string to_lower_pos_inventory_utf8(const std::string& word) {
  std::string out;
  size_t i = 0;
  while (i < word.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(word, i, cp, adv);
    utf8_append_codepoint(out, french_tolower_cp(cp));
    i += adv;
  }
  return out;
}

void load_french_lexicon_stream(std::istream& in, std::unordered_map<std::string, std::string>& out_lex) {
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
    const bool lower_lemma = (surf == to_lower_ascii(surf));
    const auto it = tmp.find(k);
    if (it == tmp.end()) {
      tmp.emplace(k, std::make_pair(std::move(ipa), lower_lemma));
    } else if (lower_lemma && !it->second.second) {
      it->second = std::make_pair(std::move(ipa), true);
    }
  }
  out_lex.clear();
  for (auto& e : tmp) {
    out_lex.emplace(std::move(e.first), std::move(e.second.first));
  }
}

void load_french_lexicon_file(const std::filesystem::path& path,
                              std::unordered_map<std::string, std::string>& out_lex) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("French G2P: cannot read lexicon " + path.generic_string());
  }
  load_french_lexicon_stream(in, out_lex);
}

std::string parse_first_csv_field(std::string_view line) {
  if (line.empty()) {
    return {};
  }
  if (line[0] == '\xEF' && line.size() >= 3 && line[1] == '\xBB' && line[2] == '\xBF') {
    line.remove_prefix(3);
  }
  if (!line.empty() && line[0] == '"') {
    std::string out;
    for (size_t i = 1; i < line.size(); ++i) {
      const char c = line[i];
      if (c == '"') {
        if (i + 1 < line.size() && line[i + 1] == '"') {
          out.push_back('"');
          ++i;
          continue;
        }
        break;
      }
      out.push_back(c);
    }
    return trim_ascii_ws_copy(out);
  }
  const size_t comma = line.find(',');
  return trim_ascii_ws_copy(line.substr(0, comma == std::string::npos ? line.size() : comma));
}

void load_french_pos_csv_stream(std::istream& in, const std::string& cat_upper,
                                std::unordered_map<std::string, std::unordered_set<std::string>>& out) {
  std::string header;
  if (!std::getline(in, header)) {
    return;
  }
  if (header.find("form") == std::string::npos) {
    return;
  }
  std::unordered_set<std::string> forms;
  std::string line;
  while (std::getline(in, line)) {
    const std::string form = parse_first_csv_field(line);
    if (form.empty() || form == "-" || form.find(' ') != std::string::npos) {
      continue;
    }
    forms.insert(to_lower_ascii(form));
  }
  if (!forms.empty()) {
    out[cat_upper] = std::move(forms);
  }
}

void load_french_pos_dir(const std::filesystem::path& dir,
                         std::unordered_map<std::string, std::unordered_set<std::string>>& out) {
  out.clear();
  if (!std::filesystem::is_directory(dir)) {
    return;
  }
  std::vector<std::filesystem::path> paths;
  for (const auto& ent : std::filesystem::directory_iterator(dir)) {
    if (!ent.is_regular_file()) {
      continue;
    }
    if (ent.path().extension() == ".csv") {
      paths.push_back(ent.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  for (const auto& p : paths) {
    std::string cat = p.stem().string();
    for (char& c : cat) {
      c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    std::ifstream in(p);
    if (!in) {
      continue;
    }
    load_french_pos_csv_stream(in, cat, out);
  }
}

void load_french_pos_from_csv_utf8_map(
    const std::unordered_map<std::string, std::string>& cat_upper_to_utf8,
    std::unordered_map<std::string, std::unordered_set<std::string>>& out) {
  out.clear();
  std::vector<std::pair<std::string, const std::string*>> sorted;
  sorted.reserve(cat_upper_to_utf8.size());
  for (const auto& pr : cat_upper_to_utf8) {
    sorted.emplace_back(pr.first, &pr.second);
  }
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  for (const auto& pr : sorted) {
    std::istringstream in(*pr.second);
    load_french_pos_csv_stream(in, pr.first, out);
  }
}

const std::array<const char*, 10> kDigitWord = {"zéro", "un",   "deux",  "trois", "quatre",
                                                "cinq", "six",  "sept",  "huit",  "neuf"};

const std::array<const char*, 17> kUnits = {"zéro",     "un",        "deux",      "trois",     "quatre",
                                            "cinq",     "six",       "sept",      "huit",      "neuf",
                                            "dix",      "onze",      "douze",     "treize",    "quatorze",
                                            "quinze",   "seize"};

std::vector<std::string> below_100(int n) {
  if (n < 0 || n >= 100) {
    throw std::out_of_range("below_100");
  }
  if (n < 17) {
    return {kUnits[static_cast<size_t>(n)]};
  }
  if (n < 20) {
    return {std::string("dix-") + kUnits[static_cast<size_t>(n - 10)]};
  }
  if (n < 60) {
    const int tens = (n / 10) * 10;
    const int u = n % 10;
    const char* tens_w = nullptr;
    if (tens == 20) {
      tens_w = "vingt";
    } else if (tens == 30) {
      tens_w = "trente";
    } else if (tens == 40) {
      tens_w = "quarante";
    } else if (tens == 50) {
      tens_w = "cinquante";
    } else {
      tens_w = "vingt";
    }
    if (u == 0) {
      return {tens_w};
    }
    if (u == 1) {
      return {std::string(tens_w) + "-et-un"};
    }
    return {std::string(tens_w) + "-" + kUnits[static_cast<size_t>(u)]};
  }
  if (n < 70) {
    return {std::string("soixante-") + kUnits[static_cast<size_t>(n - 60)]};
  }
  if (n < 80) {
    const int u = n - 70;
    if (u == 1) {
      return {"soixante-et-onze"};
    }
    if (u <= 6) {
      return {std::string("soixante-") + kUnits[static_cast<size_t>(10 + u)]};
    }
    return {std::string("soixante-dix-") + kUnits[static_cast<size_t>(u)]};
  }
  const int u = n - 80;
  if (u == 0) {
    return {"quatre-vingts"};
  }
  if (u == 10) {
    return {"quatre-vingt-dix"};
  }
  if (u < 10) {
    return {std::string("quatre-vingt-") + kUnits[static_cast<size_t>(u)]};
  }
  if (u < 17) {
    return {std::string("quatre-vingt-") + kUnits[static_cast<size_t>(u)]};
  }
  return {std::string("quatre-vingt-dix-") + kUnits[static_cast<size_t>(u - 10)]};
}

std::vector<std::string> below_1000(int n) {
  if (n < 0 || n >= 1000) {
    throw std::out_of_range("below_1000");
  }
  if (n == 0) {
    return {};
  }
  const int h = n / 100;
  const int r = n % 100;
  if (h == 0) {
    return below_100(r);
  }
  std::vector<std::string> parts;
  if (h == 1) {
    parts.emplace_back("cent");
    if (r == 0) {
      return parts;
    }
    auto tail = below_100(r);
    parts.insert(parts.end(), tail.begin(), tail.end());
    return parts;
  }
  if (r == 0) {
    parts.emplace_back(kUnits[static_cast<size_t>(h)]);
    parts.emplace_back("cents");
    return parts;
  }
  parts.emplace_back(kUnits[static_cast<size_t>(h)]);
  parts.emplace_back("cent");
  auto tail = below_100(r);
  parts.insert(parts.end(), tail.begin(), tail.end());
  return parts;
}

std::vector<std::string> below_1_000_000(int n) {
  if (n < 0 || n >= 1'000'000) {
    throw std::out_of_range("below_1_000_000");
  }
  if (n < 1000) {
    return below_1000(n);
  }
  const int q = n / 1000;
  const int r = n % 1000;
  std::vector<std::string> parts;
  if (q == 1) {
    parts.emplace_back("mille");
  } else {
    auto head = below_1000(q);
    parts.insert(parts.end(), head.begin(), head.end());
    parts.emplace_back("mille");
  }
  if (r != 0) {
    auto tail = below_1000(r);
    parts.insert(parts.end(), tail.begin(), tail.end());
  }
  return parts;
}

std::string join_space(const std::vector<std::string>& v) {
  std::string out;
  for (size_t i = 0; i < v.size(); ++i) {
    if (i > 0) {
      out.push_back(' ');
    }
    out += v[i];
  }
  return out;
}

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

std::string expand_cardinal_digits_to_french_words(std::string_view s) {
  if (!is_all_ascii_digits(s)) {
    return std::string(s);
  }
  if (s.size() > 1 && s[0] == '0') {
    std::vector<std::string> bits;
    for (char c : s) {
      bits.emplace_back(kDigitWord[static_cast<size_t>(c - '0')]);
    }
    return join_space(bits);
  }
  long long n = 0;
  for (char c : s) {
    n = n * 10 + (c - '0');
  }
  if (n > 999'999) {
    return std::string(s);
  }
  if (n == 0) {
    return "zéro";
  }
  return join_space(below_1_000_000(static_cast<int>(n)));
}

std::string expand_digit_tokens_in_text(const std::string& text) {
  static const std::regex re(R"(\b\d+\b)", std::regex::ECMAScript);
  std::string out;
  std::sregex_iterator it(text.begin(), text.end(), re);
  std::sregex_iterator end;
  size_t pos = 0;
  for (; it != end; ++it) {
    const std::smatch& m = *it;
    out.append(text, pos, static_cast<size_t>(m.position() - static_cast<long>(pos)));
    const std::string rep = expand_cardinal_digits_to_french_words(m.str());
    out += rep;
    pos = static_cast<size_t>(m.position() + static_cast<long>(m.length()));
  }
  out.append(text, pos, std::string::npos);
  return out;
}

const std::unordered_map<std::string, std::string>& cardinal_compound_ipa_map() {
  return french_compound_map::cardinal_compound_ipa_entries();
}

const std::unordered_set<std::string>& h_aspire_set() {
  static const std::unordered_set<std::string> kSet = {
      "hareng",     "harpagon",   "harpe",        "hargneux",     "hargneusement", "hautain",
      "haut",       "hâte",       "haïr",         "haï",          "haïe",          "haïes",
      "haïs",       "héros",      "héroïne",      "hérisson",     "hérésie",       "hiérarchie",
      "hollande",   "honte",      "honteux",      "huit",         "huitième",      "humble",
      "humour",     "hurler",     "hutte"};
  return kSet;
}

const std::unordered_set<std::string>& closed_liaison_determiners() {
  static const std::unordered_set<std::string> kSet = {
      "les",    "des",     "ces",      "mes",     "tes",      "ses",     "nos",      "vos",
      "leurs",  "aux",     "quelques", "plusieurs", "certains", "certaines"};
  return kSet;
}

const std::vector<std::string>& pos_scan_order() {
  static const std::vector<std::string> kOrder = {"DET", "PRON", "PREP", "CONJ", "ADJ", "ADV", "VERB",
                                                  "NOUN"};
  return kOrder;
}

std::vector<std::string> categories_for_form(
    const std::string& word, const std::unordered_map<std::string, std::unordered_set<std::string>>& inv) {
  const std::string k = to_lower_pos_inventory_utf8(word);
  std::vector<std::string> found;
  for (const std::string& cat : pos_scan_order()) {
    const auto it = inv.find(cat);
    if (it == inv.end()) {
      continue;
    }
    if (it->second.count(k) != 0u) {
      found.push_back(cat);
    }
  }
  return found;
}

std::optional<std::string> classify_pos(
    const std::string& word,
    const std::unordered_map<std::string, std::unordered_set<std::string>>& inv,
    const std::optional<std::string>& prev_pos) {
  const std::vector<std::string> cands = categories_for_form(word, inv);
  if (cands.empty()) {
    return std::nullopt;
  }
  if (cands.size() == 1) {
    return cands[0];
  }
  if (prev_pos == "DET") {
    for (const std::string& c : cands) {
      if (c == "ADJ") {
        return c;
      }
    }
    for (const std::string& c : cands) {
      if (c == "NOUN") {
        return c;
      }
    }
  }
  if (prev_pos == "PRON") {
    for (const std::string& c : cands) {
      if (c == "VERB") {
        return c;
      }
    }
  }
  for (const std::string& cat : pos_scan_order()) {
    for (const std::string& c : cands) {
      if (c == cat) {
        return c;
      }
    }
  }
  return cands[0];
}

std::string strip_stress(std::string_view ipa) {
  std::string s(ipa);
  for (;;) {
    const size_t p = s.find(kPrimaryStressUtf8);
    if (p == std::string::npos) {
      break;
    }
    s.erase(p, kPrimaryStressUtf8.size());
  }
  for (;;) {
    const size_t p = s.find(kSecondaryStressUtf8);
    if (p == std::string::npos) {
      break;
    }
    s.erase(p, kSecondaryStressUtf8.size());
  }
  return s;
}

// Longest-first nucleus prefixes (UTF-8 byte strings).
const std::vector<std::string>& french_nucleus_prefixes() {
  static const std::vector<std::string> kPrefixes = {
      "ɑ̃", "ɛ̃", "ɔ̃", "œ̃", "ə", "ɛ", "œ", "ø", "ɔ", "ɑ", "æ", "ɜ",
      "a",  "e",  "i",  "o",  "u", "y", "ɪ", "ʊ"};
  return kPrefixes;
}

std::vector<std::pair<size_t, size_t>> french_nucleus_spans(const std::string& ipa_no_stress) {
  std::vector<std::pair<size_t, size_t>> spans;
  size_t i = 0;
  const size_t n = ipa_no_stress.size();
  while (i < n) {
    bool matched = false;
    for (const std::string& p : french_nucleus_prefixes()) {
      if (ipa_no_stress.compare(i, p.size(), p) == 0) {
        spans.emplace_back(i, i + p.size());
        i += p.size();
        matched = true;
        break;
      }
    }
    if (!matched) {
      ++i;
    }
  }
  return spans;
}

std::string replace_suffix_once(std::string ipa, std::string_view old_s, std::string_view new_s) {
  const size_t pos = ipa.rfind(old_s);
  if (pos == std::string::npos) {
    return ipa;
  }
  return ipa.substr(0, pos) + std::string(new_s) + ipa.substr(pos + old_s.size());
}

std::optional<std::string> nasal_liaison_transform(const std::string& word, const std::string& ipa) {
  const std::string w = to_lower_ascii(word);
  const std::string s = strip_stress(ipa);
  if ((w == "mon" || w == "ton" || w == "son" || w == "bon") && s.size() >= 4 &&
      s.compare(s.size() - 4, 4, "ɔ̃") == 0) {
    return replace_suffix_once(ipa, "ɔ̃", "ɔn");
  }
  if (w == "un" && s.size() >= 4 && s.compare(s.size() - 4, 4, "œ̃") == 0) {
    return replace_suffix_once(ipa, "œ̃", "œn");
  }
  if ((w == "aucun" || w == "aucune") && s.size() >= 4 && s.compare(s.size() - 4, 4, "œ̃") == 0) {
    return replace_suffix_once(ipa, "œ̃", "œn");
  }
  if (w == "en" && s.size() >= 4 && s.compare(s.size() - 4, 4, "ɑ̃") == 0) {
    return replace_suffix_once(ipa, "ɑ̃", "ɑn");
  }
  return std::nullopt;
}

std::string ortho_for_liaison(std::string_view word) {
  std::string out;
  const std::string w(word);
  size_t i = 0;
  while (i < w.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(w, i, cp, adv);
    const char32_t cl = french_tolower_cp(cp);
    if ((cl >= U'a' && cl <= U'z') || cl == U'-' || cl == U'à' || cl == U'â' || cl == U'ä' || cl == U'é' ||
        cl == U'è' || cl == U'ê' || cl == U'ë' || cl == U'ï' || cl == U'î' || cl == U'ô' || cl == U'ù' ||
        cl == U'û' || cl == U'ü' || cl == U'ÿ' || cl == U'ç' || cl == U'œ' || cl == U'æ') {
      utf8_append_codepoint(out, cl);
    }
    i += adv;
  }
  return out;
}

bool utf8_last_cp(const std::string& s, char32_t& out_cp) {
  if (s.empty()) {
    return false;
  }
  size_t p = s.size();
  while (p > 0) {
    --p;
    if ((static_cast<unsigned char>(s[p]) & 0xC0) != 0x80) {
      size_t adv = 0;
      utf8_decode_at(s, p, out_cp, adv);
      return true;
    }
  }
  return false;
}

std::optional<std::string> orthographic_liaison_consonant(std::string_view word) {
  std::string w = ortho_for_liaison(word);
  if (w.empty()) {
    return std::nullopt;
  }
  if (w.size() >= 3 && (w.ends_with("ent") || w.ends_with("ont"))) {
    return std::string("t");
  }
  if (w.size() > 1 && w.back() == 'e') {
    w.pop_back();
  }
  if (w.empty()) {
    return std::nullopt;
  }
  const char last = w.back();
  static const std::unordered_map<char, const char*> cmap = {
      {'s', "z"}, {'x', "z"}, {'z', "z"}, {'d', "t"}, {'t', "t"}, {'n', "n"}, {'r', "ʁ"}, {'l', "l"},
      {'f', "v"}, {'c', "k"}, {'p', "p"}, {'g', "ɡ"}, {'m', "m"}, {'b', "b"},
  };
  const auto it = cmap.find(last);
  if (it == cmap.end()) {
    return std::nullopt;
  }
  return std::string(it->second);
}

bool ipa_starts_with_vowel_sound(std::string_view ipa_sv) {
  const std::string s = strip_stress(ipa_sv);
  if (s.empty()) {
    return false;
  }
  char32_t cp0 = 0;
  size_t adv0 = 0;
  utf8_decode_at(s, 0, cp0, adv0);
  if (cp0 == U'ɥ' || cp0 == U'w' || cp0 == U'j') {
    if (adv0 < s.size()) {
      char32_t cp1 = 0;
      size_t adv1 = 0;
      utf8_decode_at(s, adv0, cp1, adv1);
      if (cp1 == U'ə') {
        return true;
      }
      switch (cp1) {
      case U'a':
      case U'e':
      case U'i':
      case U'o':
      case U'u':
      case U'y':
      case U'ø':
      case U'œ':
      case U'ɔ':
      case U'ɑ':
      case U'ɛ':
      case U'ɜ':
      case U'ɪ':
      case U'ʊ':
        return true;
      default:
        break;
      }
    }
    return false;
  }
  switch (cp0) {
  case U'a':
  case U'e':
  case U'i':
  case U'o':
  case U'u':
  case U'y':
  case U'ø':
  case U'œ':
  case U'ɔ':
  case U'ɑ':
  case U'ɛ':
  case U'ə':
  case U'ɜ':
  case U'ɪ':
  case U'ʊ':
  case U'ɶ':
    return true;
  default:
    break;
  }
  if (adv0 < s.size()) {
    char32_t cp1 = 0;
    size_t adv1 = 0;
    utf8_decode_at(s, adv0, cp1, adv1);
    if (cp1 == U'\u0303') {
      switch (cp0) {
      case U'a':
      case U'ɔ':
      case U'o':
      case U'œ':
      case U'e':
      case U'ɛ':
      case U'i':
      case U'ɑ':
      case U'u':
        return true;
      default:
        break;
      }
    }
  }
  return false;
}

bool ipa_ends_with_audible_consonant(std::string_view ipa_sv) {
  const std::string s = strip_stress(ipa_sv);
  if (s.empty()) {
    return false;
  }
  if (s.size() >= 2 && static_cast<unsigned char>(s[s.size() - 2]) == 0xCC &&
      static_cast<unsigned char>(s.back()) == 0x83) {
    return false;
  }
  char32_t cp = 0;
  if (!utf8_last_cp(s, cp)) {
    return false;
  }
  if (cp == U'\u0303') {
    return false;
  }
  switch (cp) {
  case U'a':
  case U'e':
  case U'i':
  case U'o':
  case U'u':
  case U'y':
  case U'ø':
  case U'œ':
  case U'ɔ':
  case U'ɑ':
  case U'ɛ':
  case U'ə':
  case U'ɜ':
  case U'ɪ':
  case U'ʊ':
  case U'ɶ':
    return false;
  default:
    break;
  }
  switch (cp) {
  case U'b':
  case U'd':
  case U'f':
  case U'ɡ':
  case U'ɟ':
  case U'h':
  case U'j':
  case U'k':
  case U'l':
  case U'm':
  case U'n':
  case U'p':
  case U'q':
  case U'ʁ':
  case U'r':
  case U's':
  case U't':
  case U'v':
  case U'z':
  case U'ʃ':
  case U'ʒ':
  case U'ɲ':
  case U'ŋ':
  case U'w':
  case U'ɥ':
  case U'ç':
  case U'c':
    return true;
  default:
    return false;
  }
}

enum class LiaisonStrength { kNone, kOptional, kObligatory };

LiaisonStrength liaison_strength_fn(const std::optional<std::string>& pos_left,
                                    const std::optional<std::string>& pos_right,
                                    const std::string& wleft, const std::string& /*wright*/,
                                    bool optional_register_formal) {
  const std::string wl = to_lower_ascii(wleft);
  if (pos_left == "CONJ" && wl == "et") {
    return LiaisonStrength::kObligatory;
  }
  if (!pos_left.has_value() && closed_liaison_determiners().count(wl) != 0u &&
      (pos_right == "NOUN" || pos_right == "ADJ")) {
    return LiaisonStrength::kObligatory;
  }
  if (!pos_right.has_value() && (pos_left == "PRON" || pos_left == "DET")) {
    return LiaisonStrength::kObligatory;
  }
  if (!pos_left.has_value() && !pos_right.has_value()) {
    return LiaisonStrength::kNone;
  }
  if (!pos_left.has_value() || !pos_right.has_value()) {
    return LiaisonStrength::kNone;
  }
  if (*pos_left == "NOUN" && *pos_right == "VERB") {
    return LiaisonStrength::kNone;
  }
  if (*pos_left == "VERB" && *pos_right == "NOUN") {
    return LiaisonStrength::kNone;
  }
  if (*pos_left == "PRON" && *pos_right == "VERB") {
    return LiaisonStrength::kObligatory;
  }
  if (*pos_left == "PRON" && *pos_right == "NOUN") {
    return LiaisonStrength::kObligatory;
  }
  if (*pos_left == "DET" && (*pos_right == "NOUN" || *pos_right == "ADJ")) {
    return LiaisonStrength::kObligatory;
  }
  if (*pos_left == "DET" && *pos_right == "ADV") {
    return optional_register_formal ? LiaisonStrength::kOptional : LiaisonStrength::kNone;
  }
  if (*pos_left == "PREP") {
    return optional_register_formal ? LiaisonStrength::kOptional : LiaisonStrength::kNone;
  }
  if (*pos_left == "ADJ" && *pos_right == "NOUN") {
    return optional_register_formal ? LiaisonStrength::kOptional : LiaisonStrength::kNone;
  }
  if (*pos_left == "CONJ") {
    return LiaisonStrength::kNone;
  }
  return LiaisonStrength::kNone;
}

std::pair<std::string, std::string> apply_liaison_pair(std::string ipa_left, const std::string& ipa_right,
                                                       const std::string& wleft, const std::string& wright,
                                                       LiaisonStrength strength) {
  if (strength == LiaisonStrength::kNone) {
    return {ipa_left, ipa_right};
  }
  if (trim_ascii_ws_copy(ipa_left).empty()) {
    return {ipa_left, ipa_right};
  }
  if (h_aspire_set().count(to_lower_ascii(wright)) != 0u) {
    return {ipa_left, ipa_right};
  }
  if (!ipa_starts_with_vowel_sound(ipa_right)) {
    return {ipa_left, ipa_right};
  }
  const auto nasal = nasal_liaison_transform(wleft, ipa_left);
  if (nasal.has_value()) {
    return {*nasal, ipa_right};
  }
  if (ipa_ends_with_audible_consonant(ipa_left)) {
    return {ipa_left, ipa_right};
  }
  const auto c = orthographic_liaison_consonant(wleft);
  if (!c.has_value()) {
    return {ipa_left, ipa_right};
  }
  std::string tail = ipa_left;
  while (!tail.empty() && (tail.back() == ' ' || tail.back() == '\t')) {
    tail.pop_back();
  }
  if (tail.size() >= c->size() && tail.compare(tail.size() - c->size(), c->size(), *c) == 0) {
    return {ipa_left, ipa_right};
  }
  return {ipa_left + *c, ipa_right};
}

std::optional<std::string> lookup_lexicon(const std::unordered_map<std::string, std::string>& lex,
                                         const std::string& key) {
  const auto it = lex.find(key);
  if (it != lex.end()) {
    return it->second;
  }
  if (key.size() > 1 && key.back() == '\'') {
    const std::string k2 = key.substr(0, key.size() - 1);
    const auto it2 = lex.find(k2);
    if (it2 != lex.end()) {
      return it2->second;
    }
  }
  return std::nullopt;
}

const std::unordered_map<std::string, std::string>& heteronym_default_ipa() {
  static const std::unordered_map<std::string, std::string> kMap = {{"est", "ɛ"}, {"a", "a"}};
  return kMap;
}

size_t count_primary_stress_marks(const std::string& s) {
  size_t c = 0;
  size_t p = 0;
  while ((p = s.find(kPrimaryStressUtf8, p)) != std::string::npos) {
    ++c;
    p += kPrimaryStressUtf8.size();
  }
  return c;
}

}  // namespace

std::string FrenchRuleG2p::ensure_french_nuclear_stress(std::string ipa) {
  if (ipa.empty() || trim_ascii_ws_copy(ipa).empty()) {
    return ipa;
  }
  if (ipa.find('-') != std::string::npos) {
    std::string out;
    size_t start = 0;
    while (start < ipa.size()) {
      const size_t dash = ipa.find('-', start);
      const size_t end = dash == std::string::npos ? ipa.size() : dash;
      if (end > start) {
        std::string chunk = ipa.substr(start, end - start);
        if (!out.empty()) {
          out.push_back('-');
        }
        out += ensure_french_nuclear_stress(std::move(chunk));
      }
      if (dash == std::string::npos) {
        break;
      }
      start = dash + 1;
    }
    return out;
  }
  std::string s = strip_stress(ipa);
  if (s.empty()) {
    return ipa;
  }
  const auto spans = french_nucleus_spans(s);
  if (spans.empty()) {
    return kPrimaryStressUtf8 + s;
  }
  const size_t start = spans.back().first;
  return s.substr(0, start) + kPrimaryStressUtf8 + s.substr(start);
}

FrenchRuleG2p::FrenchRuleG2p(std::filesystem::path dict_tsv, std::filesystem::path csv_dir)
    : FrenchRuleG2p(std::move(dict_tsv), std::move(csv_dir), Options{}) {}

FrenchRuleG2p::FrenchRuleG2p(std::filesystem::path dict_tsv, std::filesystem::path csv_dir, Options options)
    : options_(options), csv_dir_(std::move(csv_dir)) {
  load_french_lexicon_file(dict_tsv, lexicon_);
  load_french_pos_dir(csv_dir_, pos_by_cat_);
}

FrenchRuleG2p::FrenchRuleG2p(std::string dict_tsv_utf8, std::filesystem::path csv_dir, Options options)
    : options_(options), csv_dir_(std::move(csv_dir)) {
  std::istringstream in(std::move(dict_tsv_utf8));
  load_french_lexicon_stream(in, lexicon_);
  load_french_pos_dir(csv_dir_, pos_by_cat_);
}

FrenchRuleG2p::FrenchRuleG2p(std::string dict_tsv_utf8,
                             std::unordered_map<std::string, std::string> pos_csv_utf8_by_cat_upper,
                             Options options)
    : options_(options), csv_dir_{} {
  std::istringstream in(std::move(dict_tsv_utf8));
  load_french_lexicon_stream(in, lexicon_);
  load_french_pos_from_csv_utf8_map(pos_csv_utf8_by_cat_upper, pos_by_cat_);
}

std::string FrenchRuleG2p::finalize_word_ipa(std::string ipa, bool from_compound) const {
  if (!options_.with_stress) {
    return strip_stress(ipa);
  }
  if (!from_compound) {
    return ensure_french_nuclear_stress(std::move(ipa));
  }
  return ipa;
}

std::string FrenchRuleG2p::word_to_ipa_impl(const std::string& raw_word, bool expand_digits) const {
  const std::string wraw = trim_ascii_ws_copy(raw_word);
  if (wraw.empty()) {
    return "";
  }
  if (expand_digits && options_.expand_cardinal_digits && is_all_ascii_digits(wraw)) {
    const std::string phrase = expand_cardinal_digits_to_french_words(wraw);
    if (phrase != wraw) {
      return text_to_ipa_impl(phrase, false, nullptr);
    }
  }
  const std::string key = normalize_lookup_key_utf8(wraw);
  if (key.empty()) {
    return "";
  }
  std::optional<std::string> ipa = lookup_lexicon(lexicon_, key);
  if (!ipa.has_value()) {
    const auto hit = heteronym_default_ipa().find(key);
    if (hit != heteronym_default_ipa().end()) {
      ipa = hit->second;
    }
  }
  bool from_compound = false;
  if (!ipa.has_value()) {
    const std::string low = to_lower_ascii(wraw);
    const auto& cmap = cardinal_compound_ipa_map();
    const auto it = cmap.find(low);
    if (it != cmap.end()) {
      ipa = it->second;
      from_compound = true;
    }
  }
  if (!ipa.has_value() && options_.oov_rules) {
    ipa = french_detail::oov_word_to_ipa(wraw, options_.with_stress);
    from_compound = false;
  }
  if (!ipa.has_value() || ipa->empty()) {
    return "";
  }
  return finalize_word_ipa(std::move(*ipa), from_compound);
}

std::string FrenchRuleG2p::word_to_ipa(const std::string& word) const {
  return word_to_ipa_impl(word, options_.expand_cardinal_digits);
}

std::string FrenchRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  return text_to_ipa_impl(text, options_.expand_cardinal_digits, per_word_log);
}

std::string FrenchRuleG2p::text_to_ipa_impl(const std::string& text, bool expand_digits,
                                            std::vector<G2pWordLog>* per_word_log) const {
  std::string work = text;
  if (expand_digits && options_.expand_cardinal_digits) {
    work = expand_digit_tokens_in_text(work);
  }

  struct Tok {
    std::string s;
    bool is_word = false;
  };
  std::vector<Tok> tokens;
  std::vector<std::string> words;

  size_t pos = 0;
  const size_t n = work.size();
  while (pos < n) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(work, pos, cp, adv);
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
      const size_t start = pos;
      pos += adv;
      while (pos < n) {
        utf8_decode_at(work, pos, cp, adv);
        if (cp != U' ' && cp != U'\t' && cp != U'\n' && cp != U'\r') {
          break;
        }
        pos += adv;
      }
      tokens.push_back({work.substr(start, pos - start), false});
      continue;
    }
    if (is_french_word_char(cp)) {
      const size_t start = pos;
      pos += adv;
      while (pos < n) {
        utf8_decode_at(work, pos, cp, adv);
        if (!is_french_word_char(cp)) {
          break;
        }
        pos += adv;
      }
      const std::string tok = work.substr(start, pos - start);
      tokens.push_back({tok, true});
      words.push_back(tok);
      continue;
    }
    const size_t start = pos;
    pos += adv;
    while (pos < n) {
      utf8_decode_at(work, pos, cp, adv);
      if (is_french_word_char(cp) || cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
        break;
      }
      pos += adv;
    }
    tokens.push_back({work.substr(start, pos - start), false});
  }

  std::vector<std::optional<std::string>> ipas;
  std::vector<std::optional<std::string>> poses;
  std::optional<std::string> prev_pos;
  for (const std::string& w : words) {
    const std::string k = normalize_lookup_key_utf8(w);
    std::string ipa = word_to_ipa_impl(w, false);
    if (ipa.empty() && !k.empty()) {
      ipas.push_back(std::nullopt);
    } else if (ipa.empty()) {
      ipas.push_back(std::nullopt);
    } else {
      ipas.push_back(std::move(ipa));
    }
    auto pos_tag = classify_pos(w, pos_by_cat_, prev_pos);
    poses.push_back(pos_tag);
    if (pos_tag.has_value()) {
      prev_pos = *pos_tag;
    } else {
      const std::string wl = to_lower_ascii(w);
      const auto dit = pos_by_cat_.find("DET");
      const bool in_det = dit != pos_by_cat_.end() && dit->second.count(wl) != 0u;
      if (closed_liaison_determiners().count(wl) != 0u || in_det) {
        prev_pos = "DET";
      } else {
        prev_pos = pos_tag;
      }
    }
  }

  std::vector<std::string> ipa_by_idx;
  if (!options_.liaison) {
    ipa_by_idx.reserve(words.size());
    for (const std::string& w : words) {
      ipa_by_idx.push_back(word_to_ipa_impl(w, false));
    }
  } else {
    ipa_by_idx.resize(words.size());
    for (size_t i = 0; i < words.size(); ++i) {
      std::string left = ipas[i].value_or("");
      if (i + 1 < words.size()) {
        const std::string& right_ipa = ipas[i + 1].value_or("");
        LiaisonStrength st = liaison_strength_fn(poses[i], poses[i + 1], words[i], words[i + 1],
                                                 options_.liaison_optional);
        if (st == LiaisonStrength::kNone && to_lower_ascii(words[i]) == "et" && !poses[i].has_value()) {
          st = LiaisonStrength::kObligatory;
        }
        if (st == LiaisonStrength::kOptional && !options_.liaison_optional) {
          st = LiaisonStrength::kNone;
        }
        if (st != LiaisonStrength::kNone) {
          auto pr = apply_liaison_pair(std::move(left), right_ipa, words[i], words[i + 1], st);
          left = std::move(pr.first);
        }
      }
      if (options_.with_stress && count_primary_stress_marks(left) <= 1) {
        left = ensure_french_nuclear_stress(std::move(left));
      }
      ipa_by_idx[i] = std::move(left);
    }
  }

  std::string out;
  size_t wi = 0;
  bool prev_was_word = false;
  for (const Tok& t : tokens) {
    if (t.is_word) {
      const std::string ipa_w = wi < ipa_by_idx.size() ? ipa_by_idx[wi] : std::string{};
      const std::string kw = wi < words.size() ? normalize_lookup_key_utf8(words[wi]) : std::string{};
      if (per_word_log != nullptr) {
        per_word_log->push_back(
            G2pWordLog{t.s, kw, G2pWordPath::kRuleBasedG2p, ipa_w});
      }
      if (prev_was_word) {
        out.push_back(' ');
      }
      out += ipa_w;
      ++wi;
      prev_was_word = true;
    } else if (!t.s.empty() &&
               std::all_of(t.s.begin(), t.s.end(), [](unsigned char c) { return std::isspace(c) != 0; })) {
      out.push_back(' ');
      prev_was_word = false;
    } else {
      out += t.s;
      prev_was_word = false;
    }
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

bool dialect_resolves_to_french_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "fr" || s == "fr-fr" || s == "french";
}

std::vector<std::string> FrenchRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first({"fr", "fr-FR", "fr_fr", "french"});
}

}  // namespace moonshine_tts
