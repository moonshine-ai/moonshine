#include "japanese-onnx-g2p.h"
#include "japanese-kana-to-ipa.h"
#include "moonshine-g2p-options.h"
#include "utf8-utils.h"

#include <algorithm>
#include <fstream>
#include <istream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" {
#include <utf8proc.h>
}

namespace moonshine_tts {
namespace {

std::string utf8_nfc_utf8proc(std::string_view s) {
  const std::string tmp(s);
  utf8proc_uint8_t* p =
      utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(tmp.c_str()));
  if (p == nullptr) {
    return std::string(s);
  }
  std::string out(reinterpret_cast<char*>(p));
  std::free(p);
  return out;
}

bool is_han_cp(char32_t cp) {
  const std::uint32_t o = static_cast<std::uint32_t>(cp);
  return (0x4E00u <= o && o <= 0x9FFFu) || (0x3400u <= o && o <= 0x4DBFu) ||
         (0xF900u <= o && o <= 0xFAFFu);
}

bool is_single_han(std::string_view s) {
  std::u32string u = utf8_str_to_u32(std::string(s));
  return u.size() == 1 && is_han_cp(u[0]);
}

bool only_hiragana(std::string_view s) {
  std::u32string u = utf8_str_to_u32(std::string(s));
  if (u.empty()) {
    return false;
  }
  for (char32_t c : u) {
    if (c < 0x3040 || c > 0x309F) {
      return false;
    }
  }
  return true;
}

bool only_katakana(std::string_view s) {
  std::u32string u = utf8_str_to_u32(std::string(s));
  if (u.empty()) {
    return false;
  }
  for (char32_t c : u) {
    if (c == U'ー') {
      continue;
    }
    if (c < 0x30A0 || c > 0x30FF) {
      return false;
    }
  }
  return true;
}

bool only_han(std::string_view s) {
  std::u32string u = utf8_str_to_u32(std::string(s));
  if (u.empty()) {
    return false;
  }
  for (char32_t c : u) {
    if (!is_han_cp(c)) {
      return false;
    }
  }
  return true;
}

std::vector<std::pair<std::string, std::string>> merge_single_han_luws(
    std::vector<std::pair<std::string, std::string>> pairs) {
  std::vector<std::pair<std::string, std::string>> out;
  std::size_t i = 0;
  while (i < pairs.size()) {
    const auto& surf = pairs[i].first;
    const auto& tag = pairs[i].second;
    if (is_single_han(surf)) {
      std::size_t j = i + 1;
      std::string acc = std::string(surf);
      const std::string first_tag = tag;
      while (j < pairs.size()) {
        const std::string& s2 = pairs[j].first;
        if (is_single_han(s2)) {
          acc += s2;
          ++j;
        } else {
          break;
        }
      }
      out.emplace_back(std::move(acc), first_tag);
      i = j;
      continue;
    }
    out.push_back(pairs[i]);
    ++i;
  }
  return out;
}

std::vector<std::pair<std::string, std::string>> merge_katakana_plus_han(
    std::vector<std::pair<std::string, std::string>> pairs) {
  std::vector<std::pair<std::string, std::string>> out;
  std::size_t i = 0;
  while (i < pairs.size()) {
    const auto& surf = pairs[i].first;
    const auto& tag = pairs[i].second;
    if (only_katakana(surf) && (tag == "NOUN" || tag == "PROPN") && i + 1 < pairs.size()) {
      const std::string& s2 = pairs[i + 1].first;
      const std::string& t2 = pairs[i + 1].second;
      if (is_single_han(s2) && (t2 == "NOUN" || t2 == "PROPN")) {
        out.emplace_back(surf + s2, tag);
        i += 2;
        continue;
      }
    }
    out.push_back(pairs[i]);
    ++i;
  }
  return out;
}

std::vector<std::pair<std::string, std::string>> merge_verb_adj_okurigana(
    std::vector<std::pair<std::string, std::string>> pairs) {
  std::vector<std::pair<std::string, std::string>> out;
  std::size_t i = 0;
  while (i < pairs.size()) {
    const auto& surf = pairs[i].first;
    const auto& tag = pairs[i].second;
    if (only_han(surf) && (tag == "VERB" || tag == "ADJ") && i + 1 < pairs.size()) {
      std::size_t j = i + 1;
      std::string acc = surf;
      while (j < pairs.size()) {
        if (only_hiragana(pairs[j].first)) {
          acc += pairs[j].first;
          ++j;
        } else {
          break;
        }
      }
      out.emplace_back(std::move(acc), tag);
      i = j;
      continue;
    }
    out.push_back(pairs[i]);
    ++i;
  }
  return out;
}

std::vector<std::pair<std::string, std::string>> merge_for_lexicon_lookup(
    std::vector<std::pair<std::string, std::string>> pairs) {
  auto a = merge_single_han_luws(std::move(pairs));
  auto b = merge_katakana_plus_han(std::move(a));
  return merge_verb_adj_okurigana(std::move(b));
}

const std::vector<std::string>& trailing_particles_sorted() {
  static const std::vector<std::string> k = [] {
    std::vector<std::string> v = {
        "について", "によって", "に対して", "では", "には", "から", "まで", "へは",
        "は",       "を",       "に",       "で",   "と",   "が",   "も",   "か",
        "や",       "へ",
    };
    std::sort(v.begin(), v.end(), [](const std::string& a, const std::string& b) {
      return a.size() > b.size();
    });
    return v;
  }();
  return k;
}

std::unordered_map<std::string, std::string> load_ja_lexicon_first_ipa_stream(std::istream& in) {
  std::unordered_map<std::string, std::string> lex;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string w = trim_ascii_ws_copy(line.substr(0, tab));
    if (w.empty() || lex.count(w) != 0) {
      continue;
    }
    std::string ipa_col = trim_ascii_ws_copy(line.substr(tab + 1));
    std::istringstream iss(ipa_col);
    std::string ipa0;
    if (!(iss >> ipa0)) {
      continue;
    }
    lex.emplace(std::move(w), std::move(ipa0));
  }
  return lex;
}

std::unordered_map<std::string, std::string> load_ja_lexicon_first_ipa(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in) {
    throw std::runtime_error("JapaneseOnnxG2p: cannot open dict " + p.string());
  }
  return load_ja_lexicon_first_ipa_stream(in);
}

void build_by_first(const std::unordered_map<std::string, std::string>& lex,
                    std::unordered_map<std::string, std::vector<std::string>>& by_first) {
  by_first.clear();
  for (const auto& pr : lex) {
    const std::string& w = pr.first;
    if (w.empty()) {
      continue;
    }
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(w, 0, cp, adv) || adv == 0) {
      continue;
    }
    by_first[w.substr(0, adv)].push_back(w);
  }
  for (auto& pr : by_first) {
    auto& vec = pr.second;
    std::sort(vec.begin(), vec.end(), [](const std::string& a, const std::string& b) {
      return a.size() > b.size();
    });
  }
}

}  // namespace

std::filesystem::path default_japanese_dict_path(const std::filesystem::path& g2p_data_root) {
  return g2p_data_root / "ja" / "dict.tsv";
}

JapaneseOnnxG2p::JapaneseOnnxG2p(std::filesystem::path model_dir,
                                   std::filesystem::path dict_tsv,
                                   bool use_cuda)
    : tok_(std::move(model_dir), use_cuda) {
  lex_ = load_ja_lexicon_first_ipa(dict_tsv);
  build_by_first(lex_, by_first_);
}

JapaneseOnnxG2p::JapaneseOnnxG2p(std::filesystem::path model_dir, std::string dict_tsv_utf8,
                                   bool use_cuda)
    : tok_(std::move(model_dir), use_cuda) {
  std::istringstream in(std::move(dict_tsv_utf8));
  lex_ = load_ja_lexicon_first_ipa_stream(in);
  build_by_first(lex_, by_first_);
}

JapaneseOnnxG2p::JapaneseOnnxG2p(const MoonshineG2POptions& opt, std::filesystem::path model_dir,
                                 std::filesystem::path dict_tsv, bool use_cuda)
    : tok_(&opt, kG2pJapaneseOnnxDirKey, std::move(model_dir), use_cuda) {
  lex_ = load_ja_lexicon_first_ipa(dict_tsv);
  build_by_first(lex_, by_first_);
}

JapaneseOnnxG2p::JapaneseOnnxG2p(const MoonshineG2POptions& opt, std::filesystem::path model_dir,
                                 std::string dict_tsv_utf8, bool use_cuda)
    : tok_(&opt, kG2pJapaneseOnnxDirKey, std::move(model_dir), use_cuda) {
  std::istringstream in(std::move(dict_tsv_utf8));
  lex_ = load_ja_lexicon_first_ipa_stream(in);
  build_by_first(lex_, by_first_);
}

std::string JapaneseOnnxG2p::g2p_word(std::string word_utf8) {
  std::string w = utf8_nfc_utf8proc(trim_ascii_ws_copy(word_utf8));
  if (w.empty() || !japanese_has_japanese_script(w)) {
    return "";
  }
  const auto it = lex_.find(w);
  if (it != lex_.end()) {
    return it->second;
  }
  const auto& parts = trailing_particles_sorted();
  for (const std::string& suf : parts) {
    if (w.size() > suf.size() && w.size() >= suf.size() && w.compare(w.size() - suf.size(), suf.size(), suf) == 0) {
      const std::string base = w.substr(0, w.size() - suf.size());
      const std::string ipa_b = g2p_word(base);
      const std::string ipa_s = g2p_word(suf);
      if (!ipa_b.empty() && !ipa_s.empty()) {
        return ipa_b + ipa_s;
      }
      if (!ipa_b.empty()) {
        return ipa_b + (ipa_s.empty() ? katakana_hiragana_to_ipa(suf) : ipa_s);
      }
      break;
    }
  }
  if (japanese_is_kana_only(w)) {
    return katakana_hiragana_to_ipa(w);
  }

  std::string ipa_acc;
  for (std::size_t i = 0; i < w.size();) {
    char32_t cp0 = 0;
    size_t adv0 = 0;
    utf8_decode_at(w, i, cp0, adv0);
    std::string fk;
    for (size_t j = 0; j < adv0; ++j) {
      fk.push_back(w[i + j]);
    }
    const auto itf = by_first_.find(fk);
    bool found = false;
    if (itf != by_first_.end()) {
      for (const std::string& cand : itf->second) {
        if (w.compare(i, cand.size(), cand) == 0) {
          ipa_acc += lex_.at(cand);
          i += cand.size();
          found = true;
          break;
        }
      }
    }
    if (found) {
      continue;
    }
    std::string one;
    for (size_t j = 0; j < adv0; ++j) {
      one.push_back(w[i + j]);
    }
    if (japanese_is_kana_only(one)) {
      ipa_acc += katakana_hiragana_to_ipa(one);
    }
    i += adv0 > 0 ? adv0 : 1;
  }
  return ipa_acc;
}

std::string JapaneseOnnxG2p::text_to_ipa(std::string text_utf8) {
  std::string raw = utf8_nfc_utf8proc(trim_ascii_ws_copy(text_utf8));
  if (raw.empty()) {
    return "";
  }
  auto pairs = tok_.annotate(raw);
  std::vector<std::pair<std::string, std::string>> pv(pairs.begin(), pairs.end());
  pv = merge_for_lexicon_lookup(std::move(pv));
  std::vector<std::string> ipa_words;
  for (const auto& pr : pv) {
    const std::string ipa = g2p_word(pr.first);
    if (!ipa.empty()) {
      ipa_words.push_back(ipa);
    }
  }
  std::string out;
  for (std::size_t k = 0; k < ipa_words.size(); ++k) {
    if (k > 0) {
      out.push_back(' ');
    }
    out += ipa_words[k];
  }
  return out;
}

}  // namespace moonshine_tts
