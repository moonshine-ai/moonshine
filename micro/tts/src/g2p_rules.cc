#include "g2p_rules.h"

#include <cctype>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tts {

namespace {

// U+02C8 / U+02CC (primary / secondary stress) as UTF-8.
constexpr std::string_view kPrimaryStress{"\xCB\x88", 2};
constexpr std::string_view kSecondaryStress{"\xCB\x8C", 2};

bool Utf8StartsWith(const std::string& s, std::string_view p) {
  return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

std::string_view LastUtf8Char(std::string_view s) {
  if (s.empty()) return {};
  size_t i = s.size();
  while (i > 0) {
    --i;
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if ((c & 0xC0U) != 0x80U) return s.substr(i);
  }
  return {};
}

bool LastIpaUnitIsVowel(std::string_view prev) {
  const std::string_view last = LastUtf8Char(prev);
  if (last.size() == 1) {
    const char ch = last[0];
    return ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u' ||
           ch == 'y';
  }
  return last == "\u00E6" || last == "\u025B" || last == "\u026A" ||
         last == "\u0254" || last == "\u028A" || last == "\u0251" ||
         last == "\u0252" || last == "\u0259" || last == "\u025A" ||
         last == "\u025D" || last == "\u0268" || last == "\u0289";
}

constexpr bool IsVowel(char c) {
  return c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' || c == 'y';
}

constexpr bool IsConsonant(char c) {
  return c >= 'a' && c <= 'z' && !IsVowel(c);
}

int NextVowelIndex(std::string_view w, int start) {
  for (int j = start; j < static_cast<int>(w.size()); ++j) {
    if (IsVowel(w[static_cast<size_t>(j)])) return j;
  }
  return -1;
}

bool MagicELengthens(std::string_view w, int vowel_i) {
  if (vowel_i < 0 || vowel_i >= static_cast<int>(w.size())) return false;
  if (w.empty() || w.back() != 'e' ||
      w.size() < static_cast<size_t>(vowel_i + 3)) {
    return false;
  }
  const int j = vowel_i + 1;
  if (j >= static_cast<int>(w.size()) - 1) return false;
  const char penult = w[static_cast<size_t>(w.size() - 2)];
  if (!(penult >= 'a' && penult <= 'z' && !IsVowel(penult))) return false;
  const std::string_view mid =
      w.substr(static_cast<size_t>(j), w.size() - 1 - static_cast<size_t>(j));
  if (mid.empty()) return false;
  for (char c : mid) {
    if (IsVowel(c)) return false;
  }
  return mid.size() == 1;
}

std::pair<std::string, int> RControlled(std::string_view w, int i) {
  if (i + 1 >= static_cast<int>(w.size()) ||
      w[static_cast<size_t>(i + 1)] != 'r') {
    return {"", 0};
  }
  switch (w[static_cast<size_t>(i)]) {
    case 'a':
      return {"\u0251\u0279", 2};  // ɑɹ
    case 'e':
      return {"\u025B\u0279", 2};  // ɛɹ
    case 'i':
      return {"\u026A\u0279", 2};  // ɪɹ
    case 'o':
      return {"\u0254\u0279", 2};  // ɔɹ
    case 'u':
      return {"\u028A\u0279", 2};  // ʊɹ
    case 'y':
      return {"a\u026A\u0279", 2};  // aɪɹ
    default:
      return {"", 0};
  }
}

struct Literal {
  std::string_view grapheme;
  std::string_view ipa;
};

// Longest graphemes first.
const Literal kLiterals[] = {
    {"tch", "t\u0283"},
    {"dge", "d\u0292"},
    {"tion", "\u0283\u0259n"},
    {"sion", "\u0292\u0259n"},
    {"sure", "\u0292\u025A"},
    {"ture", "t\u0283\u025A"},
    {"ough", "o\u028A"},
    {"augh", "\u0254\u02D0"},
    {"eigh", "e\u026A"},
    {"igh", "a\u026A"},
    {"oar", "\u0254\u0279"},
    {"our", "a\u028A\u0279"},
    {"oor", "\u0254\u0279"},
    {"ear", "\u026A\u0279"},
    {"eer", "\u026A\u0279"},
    {"ier", "\u026A\u0279"},
    {"air", "\u025B\u0279"},
    {"are", "\u025B\u0279"},
    {"ire", "a\u026A\u0279"},
    {"ure", "j\u028A\u0279"},
    {"ai", "e\u026A"},
    {"ay", "e\u026A"},
    {"au", "\u0254\u02D0"},
    {"aw", "\u0254\u02D0"},
    {"ea", "i\u02D0"},
    {"ee", "i\u02D0"},
    {"ei", "e\u026A"},
    {"ey", "e\u026A"},
    {"eu", "j\u0075\u02D0"},
    {"ew", "j\u0075\u02D0"},
    {"ie", "i\u02D0"},
    {"oa", "o\u028A"},
    {"oe", "o\u028A"},
    {"oi", "\u0254\u026A"},
    {"oy", "\u0254\u026A"},
    {"oo", "u\u02D0"},
    {"ou", "a\u028A"},
    {"ow", "o\u028A"},
    {"ph", "f"},
    {"gh", ""},
    {"ng", "\u014B"},
    {"ch", "t\u0283"},
    {"sh", "\u0283"},
    {"th", "\u03B8"},
    {"wh", "w"},
    {"qu", "kw"},
    {"ck", "k"},
    {"sch", "sk"},
    {"ss", "s"},
    {"ll", "l"},
    {"mm", "m"},
    {"nn", "n"},
    {"ff", "f"},
    {"pp", "p"},
    {"tt", "t"},
    {"zz", "z"},
    {"rr", "\u0279"},
    {"dd", "d"},
    {"bb", "b"},
    {"gg", "\u0261"},
};

const std::unordered_map<std::string, std::string>& FunctionWords() {
  static const std::unordered_map<std::string, std::string> m = {
      {"the", "\u00F0\u0259"},
      {"a", "\u0259"},
      {"an", "\u00E6n"},
      {"to", "t\u0259"},
      {"of", "\u0259v"},
      {"and", "\u00E6nd"},
      {"or", "\u0254\u0279"},
      {"are", "\u0251\u0279"},
      {"for", "f\u0254\u0279"},
      {"was", "w\u0259z"},
      {"were", "w\u025D"},
      {"from", "f\u0279\u028Cm"},
      {"have", "h\u00E6v"},
      {"has", "h\u00E6z"},
      {"been", "b\u026An"},
      {"do", "du"},
      {"does", "d\u028Cz"},
      {"your", "j\u0254\u0279"},
      {"you", "ju"},
      {"they", "\u00F0e\u026A"},
      {"their", "\u00F0\u025B\u0279"},
      {"there", "\u00F0\u025B\u0279"},
  };
  return m;
}

bool ThVoicedWord(std::string_view w) {
  return w == "the" || w == "this" || w == "that" || w == "they" ||
         w == "then" || w == "than" || w == "there" || w == "these" ||
         w == "those";
}

std::string SingleConsonant(char c, std::string_view w, int i) {
  if (c == 'c') {
    const char nxt = (i + 1 < static_cast<int>(w.size()))
                         ? w[static_cast<size_t>(i + 1)]
                         : '\0';
    return (nxt == 'e' || nxt == 'i' || nxt == 'y') ? "s" : "k";
  }
  if (c == 'g') {
    const char nxt = (i + 1 < static_cast<int>(w.size()))
                         ? w[static_cast<size_t>(i + 1)]
                         : '\0';
    return (nxt == 'e' || nxt == 'i' || nxt == 'y') ? "d\u0292" : "\u0261";
  }
  if (c == 'j') return "d\u0292";
  if (c == 'q') return "k";
  if (c == 'x') return "ks";
  if (c == 'y') {
    if (i == 0 && NextVowelIndex(w, 1) >= 0) return "j";
    return "a\u026A";
  }
  if (c == 'r') return "\u0279";
  if (c == 'h') return "h";
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

std::pair<std::string, int> Vowel(std::string_view w, int i) {
  const char v = w[static_cast<size_t>(i)];
  auto rc = RControlled(w, i);
  if (rc.second > 0) return rc;
  const bool magic = MagicELengthens(w, i);
  const int nxt_c = NextVowelIndex(w, i + 1);
  bool closed = false;
  if (nxt_c >= 0) {
    const std::string_view between = w.substr(
        static_cast<size_t>(i + 1), static_cast<size_t>(nxt_c - i - 1));
    if (!between.empty()) {
      closed = true;
      for (char c : between) {
        if (IsVowel(c)) {
          closed = false;
          break;
        }
      }
    }
  } else if (i + 1 < static_cast<int>(w.size()) &&
             !IsVowel(w[static_cast<size_t>(i + 1)])) {
    closed = true;
  }
  if (v == 'a') {
    if (magic) return {"e\u026A", 1};
    if (closed) return {"\u00E6", 1};
    return {"\u0251\u02D0", 1};
  }
  if (v == 'e') {
    if (magic) return {"i\u02D0", 1};
    if (closed || i == static_cast<int>(w.size()) - 1) return {"\u025B", 1};
    return {"i\u02D0", 1};
  }
  if (v == 'i') {
    if (magic) return {"a\u026A", 1};
    if (closed) return {"\u026A", 1};
    return {"a\u026A", 1};
  }
  if (v == 'o') {
    if (magic) return {"o\u028A", 1};
    if (closed) return {"\u0252", 1};
    return {"o\u028A", 1};
  }
  if (v == 'u') {
    if (magic) return {"j\u0075\u02D0", 1};
    if (closed) return {"\u028C", 1};
    return {"u\u02D0", 1};
  }
  if (v == 'y') {
    if (closed) return {"\u026A", 1};
    return {"a\u026A", 1};
  }
  return {"\u0259", 1};
}

// Vowel-onset prefixes, longest first, for placing the primary stress mark.
const char* kVowelPrefixes[] = {
    "a\u026A\u0279",
    "a\u026A",
    "a\u028A",
    "e\u026A",
    "o\u028A",
    "\u0254\u026A",
    "j\u0075\u02D0",
    "i\u02D0",
    "u\u02D0",
    "\u0251\u02D0",
    "\u0254\u02D0",
    "\u025C\u02D0",
    "\u025B\u0279",
    "\u0251\u0279",
    "\u0254\u0279",
    "\u026A\u0279",
    "\u028A\u0279",
    "\u0259",
    "\u026A",
    "\u025B",
    "\u00E6",
    "\u028C",
    "\u028A",
    "\u0251",
    "\u0254",
    "i",
    "u",
    "e",
    "o",
    "\u025A",
    "\u025D",
    "\u0252",
};

std::string AddPrimaryStressIfMissing(std::string s) {
  if (s.empty()) return s;
  if (Utf8StartsWith(s, kPrimaryStress) ||
      Utf8StartsWith(s, kSecondaryStress)) {
    return s;
  }
  for (const char* pref : kVowelPrefixes) {
    const std::string_view p(pref);
    const size_t k = s.find(p);
    if (k != std::string::npos) {
      std::string out = s.substr(0, k);
      out.append(kPrimaryStress);
      out.append(s, k, std::string::npos);
      return out;
    }
  }
  return std::string(kPrimaryStress) + s;
}

std::string GraphemeToIpa(std::string_view word) {
  std::string letters;
  letters.reserve(word.size());
  for (unsigned char uc : word) {
    const char c = static_cast<char>(std::tolower(uc));
    if (c >= 'a' && c <= 'z') letters.push_back(c);
  }
  if (letters.empty()) return "";

  const auto& fw = FunctionWords();
  const auto it = fw.find(letters);
  if (it != fw.end()) return it->second;

  const std::string& wv = letters;
  std::vector<std::string> parts;
  int i = 0;
  const int n = static_cast<int>(wv.size());
  while (i < n) {
    // Silent final 'e' (only if we already produced something).
    if (wv[static_cast<size_t>(i)] == 'e' && i == n - 1 && !parts.empty()) {
      ++i;
      continue;
    }
    bool matched = false;
    for (const Literal& lit : kLiterals) {
      const int L = static_cast<int>(lit.grapheme.size());
      if (L <= 0 || i + L > n) continue;
      if (std::string_view(wv).substr(static_cast<size_t>(i),
                                      static_cast<size_t>(L)) != lit.grapheme) {
        continue;
      }
      if (lit.grapheme == "gh") {
        bool silent_after_vowel = false;
        if (!parts.empty() && LastIpaUnitIsVowel(parts.back())) {
          silent_after_vowel = true;
        }
        if (silent_after_vowel) {
          i += 2;
          matched = true;
          break;
        }
        parts.emplace_back("\u0261");
        i += 2;
        matched = true;
        break;
      }
      if (lit.grapheme == "th") {
        parts.emplace_back(ThVoicedWord(wv) ? "\u00F0" : "\u03B8");
        i += 2;
        matched = true;
        break;
      }
      parts.emplace_back(lit.ipa);
      i += L;
      matched = true;
      break;
    }
    if (matched) continue;

    const char c = wv[static_cast<size_t>(i)];
    if (IsVowel(c)) {
      auto pr = Vowel(wv, i);
      parts.push_back(std::move(pr.first));
      i += pr.second;
      continue;
    }
    if (IsConsonant(c)) {
      parts.push_back(SingleConsonant(c, wv, i));
      ++i;
      continue;
    }
    ++i;
  }
  std::string out;
  for (const std::string& p : parts) out += p;
  return out;
}

}  // namespace

std::string RulesWordToIpa(std::string_view word) {
  // Function words keep their curated (reduced, unstressed) form: forcing a
  // primary stress here would make the synth put a pitch accent on every "the",
  // "a", "of", etc., which is unnatural for unstressed words.
  std::string letters;
  letters.reserve(word.size());
  for (unsigned char uc : word) {
    const char c = static_cast<char>(std::tolower(uc));
    if (c >= 'a' && c <= 'z') letters.push_back(c);
  }
  const auto& fw = FunctionWords();
  const auto it = fw.find(letters);
  if (it != fw.end()) return it->second;
  return AddPrimaryStressIfMissing(GraphemeToIpa(word));
}

}  // namespace tts
