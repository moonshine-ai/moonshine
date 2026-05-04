#include "english-hand-oov.h"

#include "text-normalize.h"

#include <cctype>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace moonshine_tts {
namespace {

// U+02C8 / U+02CC as UTF-8 (avoid comparing multi-byte IPA to a single char).
constexpr std::string_view kIpaPrimaryStress{"\xCB\x88", 2};
constexpr std::string_view kIpaSecondaryStress{"\xCB\x8C", 2};

bool utf8_starts_with(const std::string& s, std::string_view p) {
  return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

std::string_view last_utf8_char(std::string_view s) {
  if (s.empty()) {
    return {};
  }
  size_t i = s.size();
  while (i > 0) {
    --i;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if ((c & 0xC0U) != 0x80U) {
      return s.substr(i);
    }
  }
  return {};
}

bool last_ipa_unit_is_vowel(std::string_view prev) {
  const std::string_view last = last_utf8_char(prev);
  if (last.size() == 1) {
    const char ch = last[0];
    return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' || ch == 'y';
  }
  return last == "æ" || last == "ɛ" || last == "ɪ" || last == "ɔ" || last == "ʊ" || last == "ɑ" ||
         last == "ɒ" || last == "ə" || last == "ɚ" || last == "ɝ" || last == "ɨ" || last == "ʉ";
}

constexpr bool is_vowel(char c) {
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

constexpr bool is_consonant(char c) {
  return c >= 'a' && c <= 'z' && !is_vowel(c);
}

int next_vowel_index(std::string_view w, int start) {
  for (int j = start; j < static_cast<int>(w.size()); ++j) {
    if (is_vowel(w[static_cast<size_t>(j)])) {
      return j;
    }
  }
  return -1;
}

bool magic_e_lengthens(std::string_view w, int vowel_i) {
  if (vowel_i < 0 || vowel_i >= static_cast<int>(w.size())) {
    return false;
  }
  if (w.empty() || w.back() != 'e' || w.size() < static_cast<size_t>(vowel_i + 3)) {
    return false;
  }
  const int j = vowel_i + 1;
  if (j >= static_cast<int>(w.size()) - 1) {
    return false;
  }
  if (!(w[static_cast<size_t>(w.size() - 2)] >= 'a' && w[static_cast<size_t>(w.size() - 2)] <= 'z' &&
        !is_vowel(w[static_cast<size_t>(w.size() - 2)]))) {
    return false;
  }
  const std::string_view mid = w.substr(static_cast<size_t>(j), w.size() - 1 - static_cast<size_t>(j));
  if (mid.empty()) {
    return false;
  }
  for (char c : mid) {
    if (is_vowel(c)) {
      return false;
    }
  }
  return mid.size() == 1;
}

std::pair<std::string, int> r_controlled(std::string_view w, int i) {
  if (i + 1 >= static_cast<int>(w.size()) || w[static_cast<size_t>(i + 1)] != 'r') {
    return {"", 0};
  }
  const char v = w[static_cast<size_t>(i)];
  if (v == 'a') {
    return {"ɑɹ", 2};
  }
  if (v == 'e') {
    return {"ɛɹ", 2};
  }
  if (v == 'i') {
    return {"ɪɹ", 2};
  }
  if (v == 'o') {
    return {"ɔɹ", 2};
  }
  if (v == 'u') {
    return {"ʊɹ", 2};
  }
  if (v == 'y') {
    return {"aɪɹ", 2};
  }
  return {"", 0};
}

struct Literal {
  std::string_view grapheme;
  std::string_view ipa;
};

// Longest graphemes first (same set as ``english_rule_g2p._OOV_LITERALS``).
const Literal kLiterals[] = {
    {"tch", "tʃ"},   {"dge", "dʒ"},   {"tion", "ʃən"}, {"sion", "ʒən"}, {"sure", "ʒɚ"},
    {"ture", "tʃɚ"}, {"ough", "oʊ"},  {"augh", "ɔː"},  {"eigh", "eɪ"},  {"igh", "aɪ"},
    {"oar", "ɔɹ"},   {"our", "aʊɹ"},  {"oor", "ɔɹ"},   {"ear", "ɪɹ"},   {"eer", "ɪɹ"},
    {"ier", "ɪɹ"},   {"air", "ɛɹ"},   {"are", "ɛɹ"},   {"ire", "aɪɹ"},  {"ure", "jʊɹ"},
    {"ai", "eɪ"},    {"ay", "eɪ"},    {"au", "ɔː"},   {"aw", "ɔː"},    {"ea", "iː"},
    {"ee", "iː"},    {"ei", "eɪ"},    {"ey", "eɪ"},    {"eu", "juː"},   {"ew", "juː"},
    {"ie", "iː"},    {"oa", "oʊ"},    {"oe", "oʊ"},    {"oi", "ɔɪ"},    {"oy", "ɔɪ"},
    {"oo", "uː"},    {"ou", "aʊ"},    {"ow", "oʊ"},    {"ph", "f"},     {"gh", ""},
    {"ng", "ŋ"},     {"ch", "tʃ"},    {"sh", "ʃ"},     {"th", "θ"},     {"wh", "w"},
    {"qu", "kw"},   {"ck", "k"},     {"sch", "sk"},   {"ss", "s"},     {"ll", "l"},
    {"mm", "m"},    {"nn", "n"},     {"ff", "f"},     {"pp", "p"},     {"tt", "t"},
    {"zz", "z"},    {"rr", "ɹ"},     {"dd", "d"},     {"bb", "b"},     {"gg", "ɡ"},
};

const std::unordered_map<std::string, std::string> kFunctionWords = {
    {"the", "ðə"},   {"a", "ə"},      {"an", "æn"},    {"to", "tə"},    {"of", "əv"},
    {"and", "ænd"},  {"or", "ɔɹ"},    {"are", "ɑɹ"},   {"was", "wəz"},  {"were", "wɝ"},
    {"from", "fɹʌm"}, {"have", "hæv"}, {"has", "hæz"}, {"been", "bɪn"}, {"do", "du"},
    {"does", "dʌz"}, {"your", "jɔɹ"}, {"you", "ju"}, {"they", "ðeɪ"}, {"their", "ðɛɹ"},
    {"there", "ðɛɹ"},
};

bool th_voiced_word(std::string_view w) {
  return w == "the" || w == "this" || w == "that" || w == "they" || w == "then" || w == "than" ||
         w == "there" || w == "these" || w == "those";
}

std::string oov_single_consonant(char c, std::string_view w, int i) {
  if (c == 'c') {
    const char nxt = (i + 1 < static_cast<int>(w.size())) ? w[static_cast<size_t>(i + 1)] : '\0';
    if (nxt == 'e' || nxt == 'i' || nxt == 'y') {
      return "s";
    }
    return "k";
  }
  if (c == 'g') {
    const char nxt = (i + 1 < static_cast<int>(w.size())) ? w[static_cast<size_t>(i + 1)] : '\0';
    if (nxt == 'e' || nxt == 'i' || nxt == 'y') {
      return "dʒ";
    }
    return "ɡ";
  }
  if (c == 'j') {
    return "dʒ";
  }
  if (c == 'q') {
    return "k";
  }
  if (c == 'x') {
    return "ks";
  }
  if (c == 'y') {
    if (i == 0 && next_vowel_index(w, 1) >= 0) {
      return "j";
    }
    return "aɪ";
  }
  if (c == 'r') {
    return "ɹ";
  }
  if (c == 'h') {
    return "h";
  }
  switch (c) {
    case 'b':
      return "b";
    case 'd':
      return "d";
    case 'f':
      return "f";
    case 'k':
      return "k";
    case 'l':
      return "l";
    case 'm':
      return "m";
    case 'n':
      return "n";
    case 'p':
      return "p";
    case 's':
      return "s";
    case 't':
      return "t";
    case 'v':
      return "v";
    case 'w':
      return "w";
    case 'z':
      return "z";
    default:
      return std::string(1, c);
  }
}

std::pair<std::string, int> oov_vowel(std::string_view w, int i) {
  const char v = w[static_cast<size_t>(i)];
  auto rc = r_controlled(w, i);
  if (rc.second > 0) {
    return rc;
  }
  const bool magic = magic_e_lengthens(w, i);
  const int nxt_c = next_vowel_index(w, i + 1);
  bool closed = false;
  if (nxt_c >= 0) {
    const std::string_view between = w.substr(static_cast<size_t>(i + 1), static_cast<size_t>(nxt_c - i - 1));
    if (!between.empty()) {
      closed = true;
      for (char c : between) {
        if (is_vowel(c)) {
          closed = false;
          break;
        }
      }
    }
  } else if (i + 1 < static_cast<int>(w.size()) && !is_vowel(w[static_cast<size_t>(i + 1)])) {
    closed = true;
  }
  if (v == 'a') {
    if (magic) {
      return {"eɪ", 1};
    }
    if (closed) {
      return {"æ", 1};
    }
    return {"ɑː", 1};
  }
  if (v == 'e') {
    if (magic) {
      return {"iː", 1};
    }
    if (closed || i == static_cast<int>(w.size()) - 1) {
      return {"ɛ", 1};
    }
    return {"iː", 1};
  }
  if (v == 'i') {
    if (magic) {
      return {"aɪ", 1};
    }
    if (closed) {
      return {"ɪ", 1};
    }
    return {"aɪ", 1};
  }
  if (v == 'o') {
    if (magic) {
      return {"oʊ", 1};
    }
    if (closed) {
      return {"ɒ", 1};
    }
    return {"oʊ", 1};
  }
  if (v == 'u') {
    if (magic) {
      return {"juː", 1};
    }
    if (closed) {
      return {"ʌ", 1};
    }
    return {"uː", 1};
  }
  if (v == 'y') {
    if (closed) {
      return {"ɪ", 1};
    }
    return {"aɪ", 1};
  }
  return {"ə", 1};
}

const char* kVowelPrefixes[] = {
    "aɪ", "aʊ", "eɪ", "oʊ", "ɔɪ", "juː", "iː", "uː", "ɑː", "ɔː", "ɜː", "ɛɹ", "ɑɹ", "ɔɹ",
    "ɪɹ", "ʊɹ", "aɪɹ", "ɪə", "eə", "ʊə", "iə", "ə",  "ɪ",  "ɛ",  "æ",  "ʌ",  "ʊ",  "ɑ",
    "ɔ",  "i",  "u",  "e",  "o",  "ɚ",  "ɝ",  "ɒ",
};

std::string add_primary_stress_if_missing(std::string s) {
  if (s.empty()) {
    return s;
  }
  if (utf8_starts_with(s, kIpaPrimaryStress) || utf8_starts_with(s, kIpaSecondaryStress)) {
    return s;
  }
  for (const char* pref : kVowelPrefixes) {
    const std::string_view p(pref);
    const size_t k = s.find(p);
    if (k != std::string::npos) {
      std::string out = s.substr(0, k);
      out.append(kIpaPrimaryStress);
      out.append(s, k, std::string::npos);
      return out;
    }
  }
  return std::string(kIpaPrimaryStress) + s;
}

std::string oov_grapheme_to_ipa(std::string_view word) {
  std::string w = normalize_word_for_lookup(word);
  if (w.empty()) {
    return "";
  }
  std::string letters;
  letters.reserve(w.size());
  for (unsigned char uc : w) {
    const char c = static_cast<char>(uc);
    if (c >= 'a' && c <= 'z') {
      letters.push_back(c);
    }
  }
  if (letters.empty()) {
    return "";
  }
  const auto fw = kFunctionWords.find(letters);
  if (fw != kFunctionWords.end()) {
    return fw->second;
  }
  std::string wv = letters;
  std::vector<std::string> parts;
  int i = 0;
  const int n = static_cast<int>(wv.size());
  while (i < n) {
    if (wv[static_cast<size_t>(i)] == 'e' && i == n - 1 && !parts.empty()) {
      ++i;
      continue;
    }
    bool matched = false;
    for (const Literal& lit : kLiterals) {
      const int L = static_cast<int>(lit.grapheme.size());
      if (L <= 0) {
        continue;
      }
      if (i + L > n) {
        continue;
      }
      if (std::string_view(wv).substr(static_cast<size_t>(i), static_cast<size_t>(L)) != lit.grapheme) {
        continue;
      }
      if (lit.grapheme == "gh") {
        bool silent_after_vowel = false;
        if (!parts.empty()) {
          const std::string& prev = parts.back();
          if (!prev.empty() && last_ipa_unit_is_vowel(prev)) {
            silent_after_vowel = true;
          }
        }
        if (silent_after_vowel) {
          i += 2;
          matched = true;
          break;
        }
        parts.emplace_back("ɡ");
        i += 2;
        matched = true;
        break;
      }
      if (lit.grapheme == "th") {
        parts.emplace_back(th_voiced_word(wv) ? "ð" : "θ");
        i += 2;
        matched = true;
        break;
      }
      parts.emplace_back(lit.ipa.begin(), lit.ipa.end());
      i += L;
      matched = true;
      break;
    }
    if (matched) {
      continue;
    }
    const char c = wv[static_cast<size_t>(i)];
    if (is_vowel(c)) {
      auto pr = oov_vowel(wv, i);
      parts.push_back(std::move(pr.first));
      i += pr.second;
      continue;
    }
    if (is_consonant(c)) {
      parts.push_back(oov_single_consonant(c, wv, i));
      ++i;
      continue;
    }
    ++i;
  }
  std::string out;
  for (const std::string& p : parts) {
    out += p;
  }
  return out;
}

}  // namespace

std::string english_hand_oov_rules_ipa(std::string_view word) {
  return add_primary_stress_if_missing(oov_grapheme_to_ipa(word));
}

}  // namespace moonshine_tts
