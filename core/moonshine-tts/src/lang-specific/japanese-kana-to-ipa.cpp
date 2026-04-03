#include "japanese-kana-to-ipa.h"
#include "utf8-utils.h"

#include <cctype>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

extern "C" {
#include <utf8proc.h>
}

namespace moonshine_tts {
namespace {

std::string utf8_nfkc_utf8proc(std::string_view s) {
  const std::string tmp(s);
  utf8proc_uint8_t* p =
      utf8proc_NFKC(reinterpret_cast<const utf8proc_uint8_t*>(tmp.c_str()));
  if (p == nullptr) {
    return std::string(s);
  }
  std::string out(reinterpret_cast<char*>(p));
  std::free(p);
  return out;
}

std::u32string katakana_to_hiragana_u32(const std::u32string& in) {
  constexpr char32_t kHira0 = 0x3041;
  constexpr char32_t kKata0 = 0x30A1;
  constexpr char32_t kKata1 = 0x30FF;
  constexpr char32_t kProlong = 0x30FC;
  const char32_t delta = kKata0 - kHira0;
  std::u32string out;
  out.reserve(in.size());
  for (char32_t c : in) {
    if (c == kProlong) {
      out.push_back(c);
    } else if (c >= kKata0 && c <= kKata1) {
      out.push_back(c - delta);
    } else {
      out.push_back(c);
    }
  }
  return out;
}

std::string u32_to_utf8(const std::u32string& u) {
  std::string s;
  for (char32_t c : u) {
    utf8_append_codepoint(s, c);
  }
  return s;
}

bool utf8_starts_with_at(const std::string& s, std::size_t off, const std::string& pref) {
  if (off + pref.size() > s.size()) {
    return false;
  }
  return s.compare(off, pref.size(), pref) == 0;
}

void long_mark_extend_last(std::vector<std::string>& parts) {
  static const std::string mark = "\xCB\x90";  // U+02D0 ː
  if (parts.empty()) {
    parts.push_back(mark);
    return;
  }
  std::string& last = parts.back();
  static const char vowels[] = "aeiouɯ";
  for (std::size_t j = last.size(); j > 0; --j) {
    const char ch = last[j - 1];
    for (const char* v = vowels; *v != '\0'; ++v) {
      if (ch == *v) {
        last.insert(j, mark);
        return;
      }
    }
  }
  last += mark;
}

// Hiragana mora → (onset, nucleus) IPA. Longer keys must precede shorter prefixes (generator order).
static const std::vector<std::pair<std::string, std::pair<std::string, std::string>>> kMoraDesc = {
  {"きゃ", {"k", "ja"}},
  {"きゅ", {"k", "jɯ"}},
  {"きょ", {"k", "jo"}},
  {"ぎゃ", {"g", "ja"}},
  {"ぎゅ", {"g", "jɯ"}},
  {"ぎょ", {"g", "jo"}},
  {"しゃ", {"ɕ", "a"}},
  {"しゅ", {"ɕ", "ɯ"}},
  {"しょ", {"ɕ", "o"}},
  {"じゃ", {"dʑ", "a"}},
  {"じゅ", {"dʑ", "ɯ"}},
  {"じょ", {"dʑ", "o"}},
  {"ちゃ", {"tɕ", "a"}},
  {"ちゅ", {"tɕ", "ɯ"}},
  {"ちょ", {"tɕ", "o"}},
  {"にゃ", {"n", "ja"}},
  {"にゅ", {"n", "jɯ"}},
  {"にょ", {"n", "jo"}},
  {"ひゃ", {"ç", "a"}},
  {"ひゅ", {"ç", "ɯ"}},
  {"ひょ", {"ç", "o"}},
  {"びゃ", {"b", "ja"}},
  {"びゅ", {"b", "jɯ"}},
  {"びょ", {"b", "jo"}},
  {"ぴゃ", {"p", "ja"}},
  {"ぴゅ", {"p", "jɯ"}},
  {"ぴょ", {"p", "jo"}},
  {"みゃ", {"m", "ja"}},
  {"みゅ", {"m", "jɯ"}},
  {"みょ", {"m", "jo"}},
  {"りゃ", {"ɾ", "ja"}},
  {"りゅ", {"ɾ", "jɯ"}},
  {"りょ", {"ɾ", "jo"}},
  {"ふぁ", {"ɸ", "a"}},
  {"ふぃ", {"ɸ", "i"}},
  {"ふぇ", {"ɸ", "e"}},
  {"ふぉ", {"ɸ", "o"}},
  {"ふゃ", {"ɸ", "ja"}},
  {"ふゅ", {"ɸ", "jɯ"}},
  {"ふょ", {"ɸ", "jo"}},
  {"ヴぁ", {"v", "a"}},
  {"ヴぃ", {"v", "i"}},
  {"ヴぇ", {"v", "e"}},
  {"ヴぉ", {"v", "o"}},
  {"ヴゃ", {"v", "ja"}},
  {"ヴゅ", {"v", "jɯ"}},
  {"ヴょ", {"v", "jo"}},
  {"てぃ", {"t", "i"}},
  {"てゅ", {"t", "jɯ"}},
  {"でぃ", {"d", "i"}},
  {"でゅ", {"d", "jɯ"}},
  {"とぅ", {"t", "ɯ"}},
  {"どぅ", {"d", "ɯ"}},
  {"つぁ", {"ts", "a"}},
  {"つぃ", {"ts", "i"}},
  {"つぇ", {"ts", "e"}},
  {"つぉ", {"ts", "o"}},
  {"うぃ", {"ɰ", "i"}},
  {"うぇ", {"ɰ", "e"}},
  {"うぉ", {"ɰ", "o"}},
  {"あ", {"", "a"}},
  {"い", {"", "i"}},
  {"う", {"", "ɯ"}},
  {"え", {"", "e"}},
  {"お", {"", "o"}},
  {"か", {"k", "a"}},
  {"き", {"k", "i"}},
  {"く", {"k", "ɯ"}},
  {"け", {"k", "e"}},
  {"こ", {"k", "o"}},
  {"が", {"g", "a"}},
  {"ぎ", {"g", "i"}},
  {"ぐ", {"g", "ɯ"}},
  {"げ", {"g", "e"}},
  {"ご", {"g", "o"}},
  {"さ", {"s", "a"}},
  {"す", {"s", "ɯ"}},
  {"せ", {"s", "e"}},
  {"そ", {"s", "o"}},
  {"し", {"ɕ", "i"}},
  {"ざ", {"z", "a"}},
  {"ず", {"z", "ɯ"}},
  {"ぜ", {"z", "e"}},
  {"ぞ", {"z", "o"}},
  {"じ", {"dʑ", "i"}},
  {"た", {"t", "a"}},
  {"て", {"t", "e"}},
  {"と", {"t", "o"}},
  {"ち", {"tɕ", "i"}},
  {"つ", {"ts", "ɯ"}},
  {"だ", {"d", "a"}},
  {"で", {"d", "e"}},
  {"ど", {"d", "o"}},
  {"ぢ", {"dʑ", "i"}},
  {"づ", {"dz", "ɯ"}},
  {"な", {"n", "a"}},
  {"に", {"n", "i"}},
  {"ぬ", {"n", "ɯ"}},
  {"ね", {"n", "e"}},
  {"の", {"n", "o"}},
  {"は", {"h", "a"}},
  {"へ", {"h", "e"}},
  {"ほ", {"h", "o"}},
  {"ひ", {"ç", "i"}},
  {"ふ", {"ɸ", "ɯ"}},
  {"ば", {"b", "a"}},
  {"び", {"b", "i"}},
  {"ぶ", {"b", "ɯ"}},
  {"べ", {"b", "e"}},
  {"ぼ", {"b", "o"}},
  {"ぱ", {"p", "a"}},
  {"ぴ", {"p", "i"}},
  {"ぷ", {"p", "ɯ"}},
  {"ぺ", {"p", "e"}},
  {"ぽ", {"p", "o"}},
  {"ま", {"m", "a"}},
  {"み", {"m", "i"}},
  {"む", {"m", "ɯ"}},
  {"め", {"m", "e"}},
  {"も", {"m", "o"}},
  {"や", {"j", "a"}},
  {"ゆ", {"j", "ɯ"}},
  {"よ", {"j", "o"}},
  {"ら", {"ɾ", "a"}},
  {"り", {"ɾ", "i"}},
  {"る", {"ɾ", "ɯ"}},
  {"れ", {"ɾ", "e"}},
  {"ろ", {"ɾ", "o"}},
  {"わ", {"ɰ", "a"}},
  {"を", {"", "o"}},
  {"ん", {"", "ɴ"}},
  {"ぁ", {"", "a"}},
  {"ぃ", {"", "i"}},
  {"ぅ", {"", "ɯ"}},
  {"ぇ", {"", "e"}},
  {"ぉ", {"", "o"}},
  {"ゎ", {"ɰ", "a"}},
  {"ヴ", {"v", "ɯ"}},
  {"ゐ", {"j", "i"}},
  {"ゑ", {"j", "e"}},
};

std::optional<std::tuple<std::string, std::string, std::size_t>> next_mora_key(const std::string& s,
                                                                               std::size_t off) {
  for (const auto& ent : kMoraDesc) {
    const std::string& key = ent.first;
    if (utf8_starts_with_at(s, off, key)) {
      return {{ent.second.first, ent.second.second, key.size()}};
    }
  }
  return std::nullopt;
}

std::string geminate_onset(const std::string& onset, const std::string& nucleus) {
  static const std::string mark = "\xCB\x90";
  if (onset.empty()) {
    return nucleus;
  }
  return onset + mark + nucleus;
}

}  // namespace

std::string katakana_hiragana_to_ipa(std::string_view sv) {
  std::string t = utf8_nfkc_utf8proc(trim_ascii_ws_copy(sv));
  if (t.empty()) {
    return "";
  }
  std::u32string u = katakana_to_hiragana_u32(utf8_str_to_u32(t));
  const std::string s = u32_to_utf8(u);

  static const std::string kProlong = "\xE3\x83\xBC";  // ー
  static const std::string kSmallTsuH = "\xE3\x81\xA3";  // っ
  static const std::string kSmallTsuK = "\xE3\x83\x83";  // ッ

  std::vector<std::string> out;
  std::size_t i = 0;
  while (i < s.size()) {
    if (utf8_starts_with_at(s, i, kProlong)) {
      long_mark_extend_last(out);
      i += kProlong.size();
      continue;
    }
    if (utf8_starts_with_at(s, i, kSmallTsuH) || utf8_starts_with_at(s, i, kSmallTsuK)) {
      const std::size_t key_len = kSmallTsuH.size();
      const std::size_t j = i + key_len;
      const auto mora = next_mora_key(s, j);
      if (mora.has_value()) {
        const std::string& onset = std::get<0>(*mora);
        const std::string& nuc = std::get<1>(*mora);
        const std::size_t adv = std::get<2>(*mora);
        if (!onset.empty()) {
          out.push_back(geminate_onset(onset, nuc));
        } else {
          out.push_back(nuc);
        }
        i = j + adv;
      } else {
        i = j;
      }
      continue;
    }

    const auto mora = next_mora_key(s, i);
    if (!mora.has_value()) {
      char32_t cp = 0;
      size_t adv = 0;
      utf8_decode_at(s, i, cp, adv);
      i += adv > 0 ? adv : 1;
      continue;
    }
    const std::string& onset = std::get<0>(*mora);
    const std::string& nuc = std::get<1>(*mora);
    const std::size_t adv = std::get<2>(*mora);
    out.push_back(onset + nuc);
    i += adv;
  }

  std::string r;
  for (const std::string& p : out) {
    r += p;
  }
  return r;
}

bool japanese_is_kana_only(std::string_view sv) {
  std::string t = utf8_nfkc_utf8proc(trim_ascii_ws_copy(sv));
  if (t.empty()) {
    return false;
  }
  std::u32string u = katakana_to_hiragana_u32(utf8_str_to_u32(t));
  constexpr char32_t kHiraA = 0x3041;
  constexpr char32_t kHiraEnd = 0x309F;
  constexpr char32_t kKataA = 0x30A1;
  constexpr char32_t kKataEnd = 0x30FF;
  constexpr char32_t kProlong = 0x30FC;
  for (char32_t c : u) {
    if (c == U' ' || c == U'\t' || c == U'\n') {
      continue;
    }
    if (c == kProlong) {
      continue;
    }
    if (c == 0x3063 || c == 0x30C3) {
      continue;
    }
    if ((kHiraA <= c && c <= kHiraEnd) || (kKataA <= c && c <= kKataEnd)) {
      continue;
    }
    return false;
  }
  return true;
}

bool japanese_has_japanese_script(std::string_view sv) {
  std::string t = utf8_nfkc_utf8proc(trim_ascii_ws_copy(sv));
  for (char32_t c : utf8_str_to_u32(t)) {
    const std::uint32_t o = static_cast<std::uint32_t>(c);
    if ((0x4E00u <= o && o <= 0x9FFFu) || (0x3040u <= o && o <= 0x30FFu)) {
      return true;
    }
  }
  return false;
}

}  // namespace moonshine_tts
