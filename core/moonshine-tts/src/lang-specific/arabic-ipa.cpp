#include "arabic-ipa.h"

#include "utf8-utils.h"

extern "C" {
#include <utf8proc.h>
}

#include <cctype>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace moonshine_tts {
namespace {

const std::unordered_set<char32_t> kSun{U'\u062A', U'\u062B', U'\u062F', U'\u0630', U'\u0631', U'\u0632',
                                        U'\u0633', U'\u0634', U'\u0635', U'\u0636', U'\u0637', U'\u0638',
                                        U'\u0644', U'\u0646'};

bool is_ar_combining(char32_t ch) {
  const std::uint32_t o = static_cast<std::uint32_t>(ch);
  if (o >= 0x064Bu && o <= 0x065Fu) {
    return true;
  }
  if (o == 0x670u) {
    return true;
  }
  return utf8proc_category(static_cast<utf8proc_int32_t>(ch)) == UTF8PROC_CATEGORY_MN && o >= 0x0600u &&
         o <= 0x06FFu;
}

bool is_ar_base_letter(char32_t ch) {
  const std::uint32_t o = static_cast<std::uint32_t>(ch);
  if (o >= 0x064Bu && o <= 0x065Fu) {
    return false;
  }
  if (o >= 0x0621u && o <= 0x063Au) {
    return true;
  }
  if (o >= 0x0641u && o <= 0x064Au) {
    return true;
  }
  if (o == 0x671u || o == 0x672u || o == 0x673u) {
    return true;
  }
  return false;
}

std::u32string u32_nfc_u32(const std::u32string& s) {
  std::string u8;
  for (char32_t c : s) {
    utf8_append_codepoint(u8, c);
  }
  utf8proc_uint8_t* nfc = utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(u8.c_str()));
  if (nfc == nullptr) {
    return s;
  }
  std::string composed(reinterpret_cast<char*>(nfc));
  std::free(nfc);
  return utf8_str_to_u32(composed);
}

std::u32string strip_ar_diac_u32(const std::u32string& s) {
  std::u32string out;
  for (char32_t ch : s) {
    if (is_ar_combining(ch)) {
      continue;
    }
    out.push_back(ch);
  }
  return out;
}

std::string u32_to_utf8_str(const std::u32string& s) {
  std::string out;
  for (char32_t c : s) {
    utf8_append_codepoint(out, c);
  }
  return out;
}

bool has_vowel_mark_u32(const std::u32string& marks) {
  for (char32_t c : marks) {
    if (c == U'\u064E' || c == U'\u064F' || c == U'\u0650' || c == U'\u064B' || c == U'\u064C' || c == U'\u064D' ||
        c == U'\u0652') {
      return true;
    }
  }
  return false;
}

std::vector<std::pair<char32_t, std::u32string>> iter_clusters(const std::u32string& s_in) {
  const std::u32string s = u32_nfc_u32(s_in);
  std::vector<std::pair<char32_t, std::u32string>> out;
  std::size_t i = 0;
  while (i < s.size()) {
    char32_t ch = s[i];
    if (is_ar_combining(ch)) {
      if (!out.empty()) {
        out.back().second.push_back(ch);
      }
      ++i;
      continue;
    }
    if (is_ar_base_letter(ch) || (static_cast<std::uint32_t>(ch) >= 0x0600u &&
                                  static_cast<std::uint32_t>(ch) <= 0x06FFu &&
                                  utf8proc_category(static_cast<utf8proc_int32_t>(ch)) == UTF8PROC_CATEGORY_LO)) {
      std::u32string marks;
      std::size_t j = i + 1;
      while (j < s.size()) {
        char32_t c2 = s[j];
        if (is_ar_combining(c2)) {
          marks.push_back(c2);
          ++j;
        } else {
          break;
        }
      }
      out.push_back({ch, std::move(marks)});
      i = j;
      continue;
    }
    ++i;
  }
  return out;
}

std::u32string strip_spurious_tatweil_u32(const std::u32string& w) {
  std::u32string x = u32_nfc_u32(w);
  bool has = false;
  for (char32_t c : x) {
    if (c == U'\u0640') {
      has = true;
      break;
    }
  }
  if (!has) {
    return x;
  }
  std::u32string tmp;
  for (char32_t c : x) {
    if (c != U'\u0640') {
      tmp.push_back(c);
    }
  }
  if (strip_ar_diac_u32(tmp) == strip_ar_diac_u32(x)) {
    return tmp;
  }
  return x;
}

std::u32string apply_default_fatha_u32(const std::u32string& w) {
  const std::u32string s = u32_nfc_u32(w);
  std::u32string acc;
  for (const auto& pr : iter_clusters(s)) {
    char32_t base = pr.first;
    std::u32string marks = pr.second;
    if (!is_ar_base_letter(base)) {
      acc.push_back(base);
      acc += marks;
      continue;
    }
    if (base == U'\u0627' || base == U'\u0648' || base == U'\u064A' || base == U'\u0649' || base == U'\u0622' ||
        base == U'\u0629') {
      acc.push_back(base);
      acc += marks;
      continue;
    }
    std::u32string m2 = marks;
    for (;;) {
      auto p = m2.find(U'\u0640');
      if (p == std::u32string::npos) {
        break;
      }
      m2.erase(p, 1);
    }
    if (!m2.empty() && !has_vowel_mark_u32(m2) && m2.find(U'\u0651') != std::u32string::npos) {
      acc.push_back(base);
      acc += m2;
      continue;
    }
    if (!has_vowel_mark_u32(m2) && m2.find(U'\u0651') == std::u32string::npos) {
      acc.push_back(base);
      acc.push_back(U'\u064E');
      acc += m2;
    } else {
      acc.push_back(base);
      acc += m2;
    }
  }
  return u32_nfc_u32(acc);
}

std::string onset_ipa(char32_t base) {
  switch (base) {
  case U'\u0621':
    return "ʔ";
  case U'\u0623':
  case U'\u0625':
  case U'\u0624':
  case U'\u0626':
    return "ʔ";
  case U'\u0622':
    return "ʔaː";
  case U'\u0628':
    return "b";
  case U'\u062A':
    return "t";
  case U'\u062B':
    return "θ";
  case U'\u062C':
    return "dʒ";
  case U'\u062D':
    return "ħ";
  case U'\u062E':
    return "x";
  case U'\u062F':
    return "d";
  case U'\u0630':
    return "ð";
  case U'\u0631':
    return "r";
  case U'\u0632':
    return "z";
  case U'\u0633':
    return "s";
  case U'\u0634':
    return "ʃ";
  case U'\u0635':
    return "sˤ";
  case U'\u0636':
    return "dˤ";
  case U'\u0637':
    return "tˤ";
  case U'\u0638':
    return "ðˤ";
  case U'\u0639':
    return "ʕ";
  case U'\u063A':
    return "ɣ";
  case U'\u0641':
    return "f";
  case U'\u0642':
    return "q";
  case U'\u0643':
    return "k";
  case U'\u0644':
    return "l";
  case U'\u0645':
    return "m";
  case U'\u0646':
    return "n";
  case U'\u0647':
    return "h";
  case U'\u0648':
    return "w";
  case U'\u064A':
    return "j";
  default:
    return "";
  }
}

std::string vowel_from_marks(const std::u32string& marks) {
  std::u32string body = marks;
  for (;;) {
    auto p = body.find(U'\u0651');
    if (p == std::u32string::npos) {
      break;
    }
    body.erase(p, 1);
  }
  if (body.find(U'\u064E') != std::u32string::npos) {
    return "a";
  }
  if (body.find(U'\u064F') != std::u32string::npos) {
    return "u";
  }
  if (body.find(U'\u0650') != std::u32string::npos) {
    return "i";
  }
  if (body.find(U'\u064B') != std::u32string::npos) {
    return "an";
  }
  if (body.find(U'\u064C') != std::u32string::npos) {
    return "un";
  }
  if (body.find(U'\u064D') != std::u32string::npos) {
    return "in";
  }
  if (marks.find(U'\u0652') != std::u32string::npos) {
    return "";
  }
  if (marks.find(U'\u0640') != std::u32string::npos) {
    return "ː";
  }
  return "";
}

std::string gem_ipa(const std::string& onset) {
  if (onset.empty()) {
    return "";
  }
  if (onset.size() >= 2 && static_cast<unsigned char>(onset[0]) == 0xCAu &&
      static_cast<unsigned char>(onset[1]) == 0x94u) {
    return onset;
  }
  return onset + "." + onset;
}

std::string diac_word_to_ipa_u32(const std::u32string& word) {
  const std::u32string w = u32_nfc_u32(word);
  std::vector<std::string> parts;
  for (const auto& pr : iter_clusters(w)) {
    char32_t base = pr.first;
    const std::u32string& marks = pr.second;
    if (base == U' ' || base == U',' || base == U';' || base == U'?' || base == U'!') {
      continue;
    }
    if (!is_ar_base_letter(base) &&
        utf8proc_category(static_cast<utf8proc_int32_t>(base)) != UTF8PROC_CATEGORY_LO) {
      continue;
    }
    std::string v = vowel_from_marks(marks);
    bool sukun = marks.find(U'\u0652') != std::u32string::npos;
    bool sh = marks.find(U'\u0651') != std::u32string::npos;
    std::string onset = onset_ipa(base);

    if (base == U'\u0627' && marks.empty()) {
      if (!parts.empty() && (parts.back() == "a" || parts.back() == "i" || parts.back() == "u")) {
        parts.back() += "ː";
      } else {
        parts.push_back("aː");
      }
      continue;
    }
    if (base == U'\u0649' && marks.empty()) {
      parts.push_back("aː");
      continue;
    }
    if (base == U'\u0629') {
      parts.push_back((!sukun && v.empty()) ? "a" : "t");
      continue;
    }
    if (base == U'\u0648') {
      if (v == "u") {
        parts.push_back("uː");
      } else if (marks.empty()) {
        parts.push_back("w");
      } else {
        parts.push_back("w" + v);
      }
      continue;
    }
    if (base == U'\u064A') {
      if (v == "i") {
        parts.push_back("iː");
      } else if (marks.empty()) {
        parts.push_back("j");
      } else {
        parts.push_back("j" + v);
      }
      continue;
    }
    if (onset == "ʔaː") {
      parts.push_back("ʔaː");
      continue;
    }
    if (onset.empty() && base == U'\u0627') {
      continue;
    }
    std::string seg = (sh && !onset.empty()) ? gem_ipa(onset) : onset;
    if (!v.empty()) {
      parts.push_back(seg.empty() ? v : (seg + "." + v));
    } else if (sukun && !seg.empty()) {
      parts.push_back(seg);
    } else if (!seg.empty()) {
      parts.push_back(seg);
    }
  }
  std::string out;
  for (std::size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      out.push_back('.');
    }
    out += parts[i];
  }
  for (;;) {
    auto p = out.find("..");
    if (p == std::string::npos) {
      break;
    }
    out.replace(p, 2, ".");
  }
  while (!out.empty() && out.front() == '.') {
    out.erase(0, 1);
  }
  while (!out.empty() && out.back() == '.') {
    out.pop_back();
  }
  return out;
}

}  // namespace

std::string arabic_msa_strip_diacritics_utf8(std::string_view utf8) {
  std::u32string u = utf8_str_to_u32(std::string(utf8));
  return u32_to_utf8_str(strip_ar_diac_u32(u));
}

std::string arabic_msa_apply_onnx_partial_postprocess_utf8(std::string_view utf8) {
  std::u32string u = utf8_str_to_u32(std::string(utf8));
  u = strip_spurious_tatweil_u32(u);
  u = apply_default_fatha_u32(u);
  return u32_to_utf8_str(u32_nfc_u32(u));
}

std::string arabic_msa_word_to_ipa_with_assimilation_utf8(std::string_view filled_diac_utf8,
                                                            std::string_view assimilation_prefix_source_utf8) {
  const std::string wtrim = trim_ascii_ws_copy(filled_diac_utf8);
  std::u32string w = u32_nfc_u32(utf8_str_to_u32(wtrim));
  if (w.empty()) {
    return "";
  }
  const std::string key_src =
      assimilation_prefix_source_utf8.empty()
          ? std::string(filled_diac_utf8)
          : std::string(assimilation_prefix_source_utf8);
  const std::string key_trim = trim_ascii_ws_copy(key_src);
  std::u32string bare = strip_ar_diac_u32(u32_nfc_u32(utf8_str_to_u32(key_trim)));
  if (bare.size() >= 3 && bare[0] == U'\u0627' && bare[1] == U'\u0644' && kSun.count(bare[2]) != 0) {
    std::u32string stem = w.substr(2);
    std::string onset = onset_ipa(bare[2]);
    std::string stem_ipa = diac_word_to_ipa_u32(stem);
    if (stem_ipa.rfind(onset, 0) == 0) {
      stem_ipa = stem_ipa.substr(onset.size());
      while (!stem_ipa.empty() && stem_ipa.front() == '.') {
        stem_ipa.erase(0, 1);
      }
    }
    std::string gem = gem_ipa(onset);
    while (!stem_ipa.empty() && stem_ipa.back() == '.') {
      stem_ipa.pop_back();
    }
    std::string out = stem_ipa.empty() ? ("a" + gem) : ("a" + gem + "." + stem_ipa);
    while (!out.empty() && out.front() == '.') {
      out.erase(0, 1);
    }
    while (!out.empty() && out.back() == '.') {
      out.pop_back();
    }
    return out;
  }
  return diac_word_to_ipa_u32(w);
}

}  // namespace moonshine_tts
