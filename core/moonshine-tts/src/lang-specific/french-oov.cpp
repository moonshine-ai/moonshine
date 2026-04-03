#include "french-internal.h"

#include "ipa-symbols.h"
#include "utf8-utils.h"

#include <cctype>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_tts::french_detail {
namespace {

const std::string& kPrimaryStressUtf8 = moonshine_tts::ipa::kPrimaryStressUtf8;

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
  return c;
}

bool is_allowed_ortho_cp(char32_t c) {
  c = french_tolower_cp(c);
  if (c >= U'a' && c <= U'z') {
    return true;
  }
  return c == U'à' || c == U'â' || c == U'ä' || c == U'é' || c == U'è' || c == U'ê' || c == U'ë' ||
         c == U'ï' || c == U'î' || c == U'ô' || c == U'ù' || c == U'û' || c == U'ü' || c == U'ÿ' ||
         c == U'ç' || c == U'œ' || c == U'æ';
}

std::u32string letters_only_u32(const std::string& raw) {
  std::u32string out;
  size_t i = 0;
  while (i < raw.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(raw, i, cp, adv);
    if (is_allowed_ortho_cp(cp)) {
      out.push_back(french_tolower_cp(cp));
    }
    i += adv;
  }
  return out;
}

bool v_u32(char32_t ch) {
  switch (ch) {
  case U'a':
  case U'à':
  case U'â':
  case U'ä':
  case U'e':
  case U'é':
  case U'è':
  case U'ê':
  case U'ë':
  case U'i':
  case U'ï':
  case U'î':
  case U'o':
  case U'ô':
  case U'ö':
  case U'u':
  case U'ù':
  case U'û':
  case U'ü':
  case U'y':
  case U'œ':
  case U'æ':
    return true;
  default:
    return false;
  }
}

std::string insert_stress_final_syllable(std::string ipa) {
  if (ipa.empty() || ipa.find(kPrimaryStressUtf8) != std::string::npos) {
    return ipa;
  }
  static const char* nasals[] = {"œ̃", "ɑ̃", "ɛ̃", "ɔ̃"};
  int best_i = -1;
  for (const char* m : nasals) {
    const std::string ms(m);
    const size_t j = ipa.rfind(ms);
    if (j != std::string::npos && static_cast<int>(j) >= best_i) {
      best_i = static_cast<int>(j);
    }
  }
  if (best_i >= 0) {
    const size_t bi = static_cast<size_t>(best_i);
    return ipa.substr(0, bi) + kPrimaryStressUtf8 + ipa.substr(bi);
  }
  for (size_t j = ipa.size(); j > 0;) {
    size_t p = j - 1;
    while (p > 0 && (static_cast<unsigned char>(ipa[p]) & 0xC0) == 0x80) {
      --p;
    }
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(ipa, p, cp, adv);
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
      return ipa.substr(0, p) + kPrimaryStressUtf8 + ipa.substr(p);
    default:
      break;
    }
    j = p;
  }
  return kPrimaryStressUtf8 + ipa;
}

bool prev_is_nucleus_idx(const std::string& s, int idx) {
  if (idx < 0) {
    return false;
  }
  const size_t i = static_cast<size_t>(idx);
  if (i >= s.size()) {
    return false;
  }
  char32_t cp = 0;
  size_t adv = 0;
  utf8_decode_at(s, i, cp, adv);
  switch (cp) {
  case U'a':
  case U'e':
  case U'é':
  case U'è':
  case U'ê':
  case U'ë':
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
  case U'ʁ':
  case U'j':
  case U'w':
  case U'ɥ':
    return true;
  default:
    break;
  }
  if (i + 2 < s.size() && static_cast<unsigned char>(s[i]) == 0xCC &&
      static_cast<unsigned char>(s[i + 1]) == 0x83) {
    return true;
  }
  return false;
}

size_t utf8_last_cp_start(const std::string& s) {
  if (s.empty()) {
    return std::string::npos;
  }
  size_t p = s.size();
  while (p > 0) {
    --p;
    if ((static_cast<unsigned char>(s[p]) & 0xC0) != 0x80) {
      return p;
    }
  }
  return 0;
}

size_t utf8_prev_cp_start(const std::string& s, size_t cp_start) {
  if (cp_start == 0) {
    return std::string::npos;
  }
  size_t p = cp_start;
  while (p > 0) {
    --p;
    if ((static_cast<unsigned char>(s[p]) & 0xC0) != 0x80) {
      return p;
    }
  }
  return std::string::npos;
}

std::string trim_final_by_orthography(std::string ipa, const std::u32string& ortho_full) {
  std::u32string o = ortho_full;
  while (!o.empty() && o.back() == U'e') {
    o.pop_back();
  }
  if (o.empty() || ipa.empty()) {
    return ipa;
  }
  std::string s = ipa;
  while (!s.empty()) {
    const size_t last_st = utf8_last_cp_start(s);
    if (last_st == std::string::npos) {
      break;
    }
    char32_t last_cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, last_st, last_cp, adv);
    if (last_cp != U't' && last_cp != U'd') {
      break;
    }
    const size_t prev_st = utf8_prev_cp_start(s, last_st);
    if (prev_st == std::string::npos ||
        !prev_is_nucleus_idx(s, static_cast<int>(prev_st))) {
      break;
    }
    s.erase(last_st, adv);
  }
  while (!s.empty()) {
    const size_t last_st = utf8_last_cp_start(s);
    if (last_st == std::string::npos) {
      break;
    }
    char32_t last_cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, last_st, last_cp, adv);
    if (last_cp != U'p' && last_cp != U'b') {
      break;
    }
    const size_t prev_st = utf8_prev_cp_start(s, last_st);
    if (prev_st == std::string::npos ||
        !prev_is_nucleus_idx(s, static_cast<int>(prev_st))) {
      break;
    }
    s.erase(last_st, adv);
  }
  if (o.empty()) {
    return s;
  }
  const char32_t lastg = o.back();
  if (lastg == U's' || lastg == U'x' || lastg == U'z') {
    while (!s.empty()) {
      const size_t last_st = utf8_last_cp_start(s);
      if (last_st == std::string::npos) {
        break;
      }
      char32_t last_cp = 0;
      size_t adv = 0;
      utf8_decode_at(s, last_st, last_cp, adv);
      if (last_cp != U's' && last_cp != U'z') {
        break;
      }
      const size_t prev_st = utf8_prev_cp_start(s, last_st);
      if (prev_st == std::string::npos ||
          !prev_is_nucleus_idx(s, static_cast<int>(prev_st))) {
        break;
      }
      s.erase(last_st, adv);
    }
  }
  return s;
}

bool peek_eq(const std::u32string& w, size_t i, const char* ascii) {
  for (size_t k = 0; ascii[k] != '\0'; ++k) {
    const size_t j = i + k;
    if (j >= w.size()) {
      return false;
    }
    if (w[j] != static_cast<char32_t>(static_cast<unsigned char>(ascii[k]))) {
      return false;
    }
  }
  return true;
}

std::string scan_graphemes(const std::u32string& w) {
  size_t i = 0;
  const size_t n = w.size();
  std::vector<std::string> out;

  auto at_word_end = [&](size_t j) { return j >= n; };

  auto next_not_vowel = [&](size_t j) {
    if (j >= n) {
      return true;
    }
    return !v_u32(w[j]);
  };

  auto append_utf8 = [](std::vector<std::string>& o, const char* utf8_lit) { o.emplace_back(utf8_lit); };

  while (i < n) {
    const char32_t ch = w[i];

    if (ch == U'h') {
      ++i;
      continue;
    }

    if (peek_eq(w, i, "aient") && (i == 0 || !v_u32(w[i - 1]))) {
      append_utf8(out, "ɛ");
      i += 5;
      continue;
    }
    if (peek_eq(w, i, "ant") && at_word_end(i + 3)) {
      append_utf8(out, "ɑ̃");
      i += 3;
      continue;
    }
    if (peek_eq(w, i, "eaux")) {
      append_utf8(out, "o");
      i += 4;
      continue;
    }
    if (peek_eq(w, i, "eau")) {
      append_utf8(out, "o");
      i += 3;
      continue;
    }
    if (peek_eq(w, i, "tion") && next_not_vowel(i + 4)) {
      append_utf8(out, "sjɔ̃");
      i += 4;
      continue;
    }
    if (peek_eq(w, i, "sion") && next_not_vowel(i + 4)) {
      append_utf8(out, "zjɔ̃");
      i += 4;
      continue;
    }
    if (peek_eq(w, i, "oin") && next_not_vowel(i + 3)) {
      append_utf8(out, "wɛ̃");
      i += 3;
      continue;
    }
    if (peek_eq(w, i, "ien") && next_not_vowel(i + 3)) {
      append_utf8(out, "jɛ̃");
      i += 3;
      continue;
    }
    if (peek_eq(w, i, "ain") && next_not_vowel(i + 3)) {
      append_utf8(out, "ɛ̃");
      i += 3;
      continue;
    }
    if (peek_eq(w, i, "eil") && next_not_vowel(i + 3)) {
      append_utf8(out, "ɛj");
      i += 3;
      continue;
    }
    if (peek_eq(w, i, "ail") && next_not_vowel(i + 3)) {
      append_utf8(out, "aj");
      i += 3;
      continue;
    }
    if (peek_eq(w, i, "oui")) {
      append_utf8(out, "wi");
      i += 3;
      continue;
    }

    if (peek_eq(w, i, "ou")) {
      append_utf8(out, "u");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "oo")) {
      append_utf8(out, "u");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "oi")) {
      append_utf8(out, "wa");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "ai") || peek_eq(w, i, "ei")) {
      append_utf8(out, "ɛ");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "au") && (i + 2 >= n || !v_u32(w[i + 2]))) {
      append_utf8(out, "o");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "eu")) {
      append_utf8(out, "ø");
      i += 2;
      continue;
    }
    if (ch == U'œ' && i + 1 < n && w[i + 1] == U'u') {
      append_utf8(out, "ø");
      i += 2;
      continue;
    }
    if (ch == U'œ') {
      append_utf8(out, "œ");
      ++i;
      continue;
    }
    if (ch == U'æ') {
      append_utf8(out, "e");
      ++i;
      continue;
    }

    if ((peek_eq(w, i, "an") || peek_eq(w, i, "am")) && next_not_vowel(i + 2)) {
      append_utf8(out, "ɑ̃");
      i += 2;
      continue;
    }
    if ((peek_eq(w, i, "en") || peek_eq(w, i, "em")) && next_not_vowel(i + 2)) {
      if (i > 0 && (w[i - 1] == U'i' || w[i - 1] == U'ï' || w[i - 1] == U'y')) {
        append_utf8(out, "ɛ̃");
      } else {
        append_utf8(out, "ɑ̃");
      }
      i += 2;
      continue;
    }
    if ((peek_eq(w, i, "in") || peek_eq(w, i, "im") || peek_eq(w, i, "yn") || peek_eq(w, i, "ym")) &&
        next_not_vowel(i + 2)) {
      append_utf8(out, "ɛ̃");
      i += 2;
      continue;
    }
    if ((peek_eq(w, i, "on") || peek_eq(w, i, "om")) && next_not_vowel(i + 2)) {
      append_utf8(out, "ɔ̃");
      i += 2;
      continue;
    }
    if ((peek_eq(w, i, "un") || peek_eq(w, i, "um")) && next_not_vowel(i + 2)) {
      append_utf8(out, "œ̃");
      i += 2;
      continue;
    }

    if (peek_eq(w, i, "qu") && i + 2 < n && v_u32(w[i + 2])) {
      append_utf8(out, "k");
      i += 2;
      continue;
    }
    if (ch == U'g' && i + 1 < n && w[i + 1] == U'u' && i + 2 < n) {
      const char32_t e2 = w[i + 2];
      if (e2 == U'e' || e2 == U'é' || e2 == U'è' || e2 == U'ê' || e2 == U'ë' || e2 == U'i' ||
          e2 == U'ï' || e2 == U'y') {
        append_utf8(out, "ɡ");
        i += 2;
        continue;
      }
    }

    if (peek_eq(w, i, "ch")) {
      append_utf8(out, "ʃ");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "gn")) {
      append_utf8(out, "ɲ");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "ph")) {
      append_utf8(out, "f");
      i += 2;
      continue;
    }
    if (peek_eq(w, i, "th")) {
      append_utf8(out, "t");
      i += 2;
      continue;
    }

    if (ch == U'c' && i + 1 < n && w[i + 1] == U'ç') {
      append_utf8(out, "ks");
      i += 2;
      continue;
    }

    if (ch == U'ç') {
      append_utf8(out, "s");
      ++i;
      continue;
    }
    if (ch == U'c') {
      if (i + 1 < n) {
        const char32_t nx = w[i + 1];
        if (nx == U'e' || nx == U'é' || nx == U'è' || nx == U'ê' || nx == U'ë' || nx == U'i' ||
            nx == U'ï' || nx == U'y') {
          append_utf8(out, "s");
        } else {
          append_utf8(out, "k");
        }
      } else {
        append_utf8(out, "k");
      }
      ++i;
      continue;
    }
    if (ch == U'g') {
      if (i + 1 < n) {
        const char32_t nx = w[i + 1];
        if (nx == U'e' || nx == U'é' || nx == U'è' || nx == U'ê' || nx == U'ë' || nx == U'i' ||
            nx == U'ï' || nx == U'y') {
          append_utf8(out, "ʒ");
        } else {
          append_utf8(out, "ɡ");
        }
      } else {
        append_utf8(out, "ɡ");
      }
      ++i;
      continue;
    }
    if (ch == U'x') {
      if (out.empty()) {
        if (i + 1 < n && v_u32(w[i + 1])) {
          append_utf8(out, "ɡz");
        } else {
          append_utf8(out, "ks");
        }
      } else {
        const std::string& last = out.back();
        bool z_link = false;
        if (!last.empty()) {
          size_t p = last.size();
          while (p > 0) {
            --p;
            if ((static_cast<unsigned char>(last[p]) & 0xC0) != 0x80) {
              break;
            }
          }
          char32_t lcp = 0;
          size_t adv = 0;
          utf8_decode_at(last, p, lcp, adv);
          static const std::string kTilde = "\xCC\x83";
          switch (lcp) {
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
            z_link = true;
            break;
          default:
            break;
          }
          if (last.size() >= 2 && last.compare(last.size() - 2, 2, kTilde) == 0) {
            z_link = true;
          }
        }
        if (z_link) {
          append_utf8(out, "z");
        } else {
          append_utf8(out, "ks");
        }
      }
      ++i;
      continue;
    }

    if (v_u32(ch)) {
      if (ch == U'a' || ch == U'à' || ch == U'â' || ch == U'ä') {
        append_utf8(out, "a");
      } else if (ch == U'é') {
        append_utf8(out, "e");
      } else if (ch == U'è' || ch == U'ê' || ch == U'ë') {
        append_utf8(out, "ɛ");
      } else if (ch == U'e') {
        if (i + 1 >= n) {
          ++i;
          continue;
        }
        if (!v_u32(w[i + 1])) {
          append_utf8(out, "ə");
        } else {
          append_utf8(out, "e");
        }
        ++i;
        continue;
      } else if (ch == U'i' || ch == U'ï' || ch == U'î') {
        append_utf8(out, "i");
      } else if (ch == U'o' || ch == U'ô') {
        append_utf8(out, "o");
      } else if (ch == U'ö') {
        append_utf8(out, "ø");
      } else if (ch == U'u' || ch == U'ù' || ch == U'û' || ch == U'ü') {
        append_utf8(out, "y");
      } else if (ch == U'y') {
        append_utf8(out, "i");
      } else if (ch == U'œ') {
        append_utf8(out, "œ");
      } else if (ch == U'æ') {
        append_utf8(out, "e");
      } else {
        append_utf8(out, "a");
      }
      ++i;
      continue;
    }

    static const std::unordered_map<char32_t, const char*> cons = {
        {U'b', "b"}, {U'd', "d"}, {U'f', "f"}, {U'j', "ʒ"}, {U'k', "k"}, {U'l', "l"}, {U'm', "m"},
        {U'n', "n"}, {U'p', "p"}, {U'q', "k"}, {U'r', "ʁ"}, {U's', "s"}, {U't', "t"},
        {U'v', "v"}, {U'w', "w"}, {U'z', "z"},
    };
    const auto it = cons.find(ch);
    if (it != cons.end()) {
      out.push_back(it->second);
      ++i;
      continue;
    }

    ++i;
  }

  std::string joined;
  for (const auto& p : out) {
    joined += p;
  }
  return joined;
}

}  // namespace

std::string oov_word_to_ipa(const std::string& word, bool with_stress) {
  std::string raw = word;
  while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.front())) != 0) {
    raw.erase(raw.begin());
  }
  while (!raw.empty() && std::isspace(static_cast<unsigned char>(raw.back())) != 0) {
    raw.pop_back();
  }
  if (raw.empty()) {
    return "";
  }
  if (raw.find('-') != std::string::npos) {
    std::vector<std::string> chunks;
    size_t start = 0;
    while (start < raw.size()) {
      size_t d = raw.find('-', start);
      const size_t end = d == std::string::npos ? raw.size() : d;
      if (end > start) {
        chunks.push_back(raw.substr(start, end - start));
      }
      if (d == std::string::npos) {
        break;
      }
      start = d + 1;
    }
    if (chunks.empty()) {
      return "";
    }
    std::string merged;
    for (size_t c = 0; c < chunks.size(); ++c) {
      const std::string p = oov_word_to_ipa(chunks[c], with_stress);
      if (p.empty()) {
        return "";
      }
      if (c > 0) {
        merged.push_back('-');
      }
      merged += p;
    }
    return merged;
  }

  const std::u32string ortho_full = letters_only_u32(raw);
  if (ortho_full.empty()) {
    return "";
  }
  std::string ipa = scan_graphemes(ortho_full);
  ipa = trim_final_by_orthography(std::move(ipa), ortho_full);
  if (with_stress) {
    ipa = insert_stress_final_syllable(std::move(ipa));
  }
  return ipa;
}

}  // namespace moonshine_tts::french_detail
