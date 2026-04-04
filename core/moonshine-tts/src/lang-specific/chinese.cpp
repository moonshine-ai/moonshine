#include "chinese.h"

#include "g2p-path.h"
#include "g2p-word-log.h"
#include "chinese-numbers.h"
#include "utf8-utils.h"

#include <cctype>
#include <cstdint>
#include <fstream>
#include <istream>
#include <sstream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace moonshine_tts {
namespace {

std::string trim_copy_line(std::string_view s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && (std::isspace(static_cast<unsigned char>(s[a])) != 0 || s[a] == '\r')) {
    ++a;
  }
  while (b > a && (std::isspace(static_cast<unsigned char>(s[b - 1])) != 0 || s[b - 1] == '\r')) {
    --b;
  }
  return std::string(s.substr(a, b - a));
}

bool is_cjk_cp(char32_t cp) {
  return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
         (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
         (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
         (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool is_ascii_digit_cp(char32_t cp) { return cp >= U'0' && cp <= U'9'; }

bool is_fullwidth_digit_cp(char32_t cp) { return cp >= U'\uFF10' && cp <= U'\uFF19'; }

bool token_has_g2p_content(const std::string& tok) {
  for (size_t i = 0; i < tok.size();) {
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(tok, i, cp, adv)) {
      return true;
    }
    if (is_cjk_cp(cp) || is_ascii_digit_cp(cp) || is_fullwidth_digit_cp(cp)) {
      return true;
    }
    if (cp <= 0x7F && std::isalpha(static_cast<unsigned char>(cp)) != 0) {
      return true;
    }
    i += adv;
  }
  return false;
}

bool pos_in_set(std::string_view p, const std::unordered_set<std::string>& s) {
  return s.count(std::string(p)) != 0;
}

const std::unordered_set<std::string>& skip_phonetic_pos() {
  static const std::unordered_set<std::string> k{"PU", "SP", "URL", "EM", "NOI", "PUNCT", "SYM", "X"};
  return k;
}

const std::unordered_set<std::string>& verb_like_pos() {
  static const std::unordered_set<std::string> k{
      "VV",  "VA",  "VE",  "VC",  "LB",  "BA",  "SB",  "MSP", "AS",  "DER", "DEV", "DEC",
      "VERB", "AUX", "ADV"};
  return k;
}

const std::unordered_set<std::string>& noun_like_pos() {
  static const std::unordered_set<std::string> k{"NN",  "NR",  "NT",  "LC",  "OD",  "M",   "CD",
                                                 "DT",  "PN",  "NOUN", "PROPN", "ADJ", "DET",
                                                 "PRON", "NUM"};
  return k;
}

bool ipa_contains(const std::string& ipa, std::string_view sub) { return ipa.find(sub) != std::string::npos; }

bool try_consume_g2p_token(const std::string& text, size_t pos, size_t& out_end) {
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(text, pos, cp, adv)) {
    return false;
  }
  if (is_cjk_cp(cp)) {
    size_t p = pos + adv;
    while (p < text.size()) {
      if (!utf8_decode_at(text, p, cp, adv)) {
        break;
      }
      if (!is_cjk_cp(cp)) {
        break;
      }
      p += adv;
    }
    out_end = p;
    return true;
  }

  size_t p = pos;
  utf8_decode_at(text, p, cp, adv);
  if (cp == U'+' || cp == U'-') {
    p += adv;
    if (p >= text.size()) {
      return false;
    }
    utf8_decode_at(text, p, cp, adv);
  }

  const auto is_d = [](char32_t c) { return is_ascii_digit_cp(c) || is_fullwidth_digit_cp(c); };

  if (is_d(cp)) {
    bool seen_dot = false;
    while (p < text.size()) {
      if (!utf8_decode_at(text, p, cp, adv)) {
        break;
      }
      if (is_d(cp)) {
        p += adv;
        continue;
      }
      if (!seen_dot && (cp == U'.' || cp == U',')) {
        const size_t q = p + adv;
        char32_t c2 = 0;
        size_t a2 = 0;
        if (q < text.size() && utf8_decode_at(text, q, c2, a2) && is_d(c2)) {
          seen_dot = true;
          p = q;
          p += a2;
          while (p < text.size()) {
            if (!utf8_decode_at(text, p, cp, adv)) {
              break;
            }
            if (!is_d(cp)) {
              break;
            }
            p += adv;
          }
        }
        break;
      }
      if (cp == U',' || cp == U'_') {
        p += adv;
        continue;
      }
      break;
    }
    out_end = p;
    return true;
  }

  p = pos;
  utf8_decode_at(text, p, cp, adv);
  if (cp <= 0x7F && std::isalpha(static_cast<unsigned char>(cp)) != 0) {
    p += adv;
    while (p < text.size()) {
      if (!utf8_decode_at(text, p, cp, adv)) {
        break;
      }
      if (cp > 0x7F || std::isalpha(static_cast<unsigned char>(cp)) == 0) {
        break;
      }
      p += adv;
    }
    out_end = p;
    return true;
  }
  return false;
}

void load_chinese_lexicon_stream(std::istream& in, std::unordered_map<std::string, std::vector<std::string>>& lex) {
  std::string line;
  while (std::getline(in, line)) {
    const std::string t = trim_copy_line(line);
    if (t.empty() || t[0] == '#') {
      continue;
    }
    const auto tab = t.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string key = trim_copy_line(std::string_view(t).substr(0, tab));
    std::string ipa = trim_copy_line(std::string_view(t).substr(tab + 1));
    if (key.empty()) {
      continue;
    }
    lex[std::move(key)].push_back(std::move(ipa));
  }
}

}  // namespace

ChineseRuleG2p::ChineseRuleG2p(std::filesystem::path dict_tsv) {
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    throw std::runtime_error("Chinese G2P: lexicon not found: " + dict_tsv.generic_string());
  }
  std::ifstream in(dict_tsv);
  if (!in) {
    throw std::runtime_error("Chinese G2P: failed to open " + dict_tsv.generic_string());
  }
  load_chinese_lexicon_stream(in, lex_);
  if (lex_.empty()) {
    throw std::runtime_error("Chinese G2P: empty lexicon: " + dict_tsv.generic_string());
  }
}

ChineseRuleG2p::ChineseRuleG2p(std::string dict_tsv_utf8) {
  std::istringstream in(std::move(dict_tsv_utf8));
  load_chinese_lexicon_stream(in, lex_);
  if (lex_.empty()) {
    throw std::runtime_error("Chinese G2P: empty lexicon (in-memory)");
  }
}

std::string ChineseRuleG2p::disambiguate_heteronym(std::string_view word,
                                                     std::string_view pos,
                                                     const std::vector<std::string>& readings) const {
  if (readings.empty()) {
    return "";
  }
  if (readings.size() == 1) {
    return readings[0];
  }
  const std::string w(word);
  const std::string p(pos);

  if (w == "\xe8\xa1\x8c") {  // 行
    std::vector<std::string> hang;
    std::vector<std::string> xing;
    for (const std::string& r : readings) {
      if (ipa_contains(r, "xɑŋ") || ipa_contains(r, "xɤŋ")) {
        hang.push_back(r);
      }
      if (ipa_contains(r, "\xc9\x95\xc9\xaa\xc5\x8b")) {  // ɕɪŋ as UTF-8
        xing.push_back(r);
      }
    }
    if (pos_in_set(p, noun_like_pos()) && !hang.empty()) {
      return hang[0];
    }
    if (pos_in_set(p, verb_like_pos()) && !xing.empty()) {
      return xing[0];
    }
    return readings[0];
  }
  if (w == "\xe4\xba\x86") {  // 了
    for (const std::string& r : readings) {
      if (ipa_contains(r, "l\xc9\xa4") && !ipa_contains(r, "lj\xc9\x91\xca\x8a")) {  // lɤ, ljɑʊ
        if (p == "AS" || p == "SP" || p == "ETC" || p == "PART") {
          return r;
        }
      }
    }
    if (p == "VV" || p == "VERB") {
      for (const std::string& r : readings) {
        if (ipa_contains(r, "lj\xc9\x91\xca\x8a")) {
          return r;
        }
      }
    }
    return readings[0];
  }
  if (w == "\xe6\xb2\xa1") {  // 没
    if (pos_in_set(p, verb_like_pos())) {
      for (const std::string& r : readings) {
        if (ipa_contains(r, "me\xc9\xaa")) {
          return r;
        }
        if (r.size() >= 3 && r.compare(0, 3, "m\xc9\xa4") == 0) {
          return r;
        }
      }
    }
    if (pos_in_set(p, noun_like_pos())) {
      for (const std::string& r : readings) {
        if (ipa_contains(r, "m\xc9\x94")) {
          return r;
        }
      }
    }
    return readings[0];
  }
  if (w == "\xe7\x9d\x80") {  // 着
    for (const std::string& r : readings) {
        if (ipa_contains(r, "\xca\x88\xca\x82\xc9\x91\xca\x8a")) {  // ʈʂɑʊ
        if (pos_in_set(p, verb_like_pos())) {
          return r;
        }
      }
    }
    for (const std::string& r : readings) {
      if (ipa_contains(r, "\xca\x88\xca\x82\xc9\xa4") || ipa_contains(r, "\xca\x88\xca\x82u\xc9\x94")) {
        if (p == "AS" || p == "MSP" || p == "ETC" || p == "PART") {
          return r;
        }
      }
    }
    return readings[0];
  }
  if (w == "\xe5\x9c\xb0") {  // 地
    for (const std::string& r : readings) {
      if (ipa_contains(r, "t\xc9\xa4")) {
        if (p == "DEV" || p == "ADV") {
          return r;
        }
      }
    }
    for (const std::string& r : readings) {
      if (ipa_contains(r, "ti\xcb\xa5\xcb\xa9")) {  // ti˥˩ rough
        if (pos_in_set(p, noun_like_pos())) {
          return r;
        }
      }
    }
    return readings[0];
  }
  if (w == "\xe5\xbe\x97") {  // 得
    for (const std::string& r : readings) {
      if (ipa_contains(r, "t\xc9\xa4") || ipa_contains(r, "t\xc9\x99")) {
        if (p == "DER" || p == "DEV" || p == "AS" || p == "PART") {
          return r;
        }
      }
    }
    for (const std::string& r : readings) {
      if (ipa_contains(r, "te\xc9\xaa")) {
        if (pos_in_set(p, verb_like_pos())) {
          return r;
        }
      }
    }
    return readings[0];
  }
  if (w == "\xe9\x95\xbf") {  // 长 — cháng (ʈʂʰɑŋ) vs zhǎng (ʈʂɑŋ)
    if (pos_in_set(p, verb_like_pos())) {
      for (const std::string& r : readings) {
        if (ipa_contains(r, "\xca\x88\xca\x82\xca\xb0\xc9\x91\xc5\x8b")) {
          return r;
        }
      }
    }
    if (pos_in_set(p, noun_like_pos())) {
      for (const std::string& r : readings) {
        if (ipa_contains(r, "\xca\x88\xca\x82\xc9\x91\xc5\x8b") && !ipa_contains(r, "\xca\x82\xca\xb0")) {
          return r;
        }
      }
    }
    return readings[0];
  }
  if (w == "\xe6\x95\xb0") {  // 数
    for (const std::string& r : readings) {
      if (ipa_contains(r, "\xca\x82u\xcb\xa8\xcb\xa9\xcb\xa6")) {  // ʂu˨˩˦
        if (pos_in_set(p, verb_like_pos())) {
          return r;
        }
      }
    }
    for (const std::string& r : readings) {
      if (r.rfind("\xca\x82u\xcb\xa5\xcb\xa9", 0) == 0 && !ipa_contains(r, "\xca\x82u\xc9\x94") &&
          !ipa_contains(r, "ts\xca\xb0")) {
        if (pos_in_set(p, noun_like_pos())) {
          return r;
        }
      }
    }
    return readings[0];
  }
  return readings[0];
}

std::string ChineseRuleG2p::han_reading_to_ipa(std::string_view han) const {
  const std::string h(han);
  std::string out;
  for (size_t i = 0; i < h.size();) {
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(h, i, cp, adv)) {
      return "";
    }
    std::string ch;
    utf8_append_codepoint(ch, cp);
    auto it = lex_.find(ch);
    if (it == lex_.end() || it->second.empty()) {
      return "";
    }
    const std::string syll = disambiguate_heteronym(ch, "", it->second);
    if (!out.empty()) {
      out.push_back(' ');
    }
    out += syll;
    i += adv;
  }
  return out;
}

std::string ChineseRuleG2p::char_fallback_ipa(std::string_view word) const {
  for (size_t i = 0; i < word.size();) {
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(std::string(word), i, cp, adv)) {
      return "";
    }
    if (!is_cjk_cp(cp)) {
      return "";
    }
    i += adv;
  }
  return han_reading_to_ipa(word);
}

std::string ChineseRuleG2p::g2p_word_impl(std::string_view word, std::string_view pos) const {
  const std::string w = trim_copy_line(word);
  if (w.empty()) {
    return "";
  }
  if (!token_has_g2p_content(w)) {
    return "";
  }
  if (!pos.empty() && pos_in_set(pos, skip_phonetic_pos())) {
    return "";
  }

  auto it = lex_.find(w);
  if (it != lex_.end() && !it->second.empty()) {
    return disambiguate_heteronym(w, pos, it->second);
  }
  const std::string fb = char_fallback_ipa(w);
  if (!fb.empty()) {
    return fb;
  }

  if (auto han = arabic_numeral_token_to_han(w)) {
    const std::string ipa = han_reading_to_ipa(*han);
    if (!ipa.empty()) {
      return ipa;
    }
  }

  bool all_ascii_alpha = true;
  for (unsigned char c : w) {
    if (!std::isalpha(c)) {
      all_ascii_alpha = false;
      break;
    }
  }
  if (all_ascii_alpha && !w.empty()) {
    std::string lo;
    lo.reserve(w.size());
    for (unsigned char c : w) {
      lo.push_back(static_cast<char>(std::tolower(c)));
    }
    return lo;
  }
  return "";
}

std::string ChineseRuleG2p::word_to_ipa(std::string_view word) const {
  return g2p_word_impl(word, "");
}

std::string ChineseRuleG2p::word_to_ipa_with_pos(std::string_view word, std::string_view pos) const {
  return g2p_word_impl(word, pos);
}

std::string ChineseRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  std::string out;
  size_t pos = 0;
  const size_t n = text.size();
  while (pos < n) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(text, pos, cp, adv);
    if (cp == U' ' || cp == U'\t' || cp == U'\n' || cp == U'\r') {
      if (!out.empty() && out.back() != ' ') {
        out.push_back(' ');
      }
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
    if (try_consume_g2p_token(text, pos, wend)) {
      const std::string tok = text.substr(pos, wend - pos);
      std::string wipa = g2p_word_impl(tok, "");
      if (per_word_log != nullptr) {
        per_word_log->push_back(
            G2pWordLog{tok, tok, G2pWordPath::kRuleBasedG2p, wipa});
      }
      if (!wipa.empty()) {
        if (!out.empty() && out.back() != ' ') {
          out.push_back(' ');
        }
        out += wipa;
      }
      pos = wend;
      continue;
    }
    pos += adv;
  }
  while (!out.empty() && out.front() == ' ') {
    out.erase(out.begin());
  }
  while (!out.empty() && out.back() == ' ') {
    out.pop_back();
  }
  return out;
}

bool dialect_resolves_to_chinese_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "zh" || s == "zh-hans" || s == "zh-cn" || s == "cmn" || s == "chinese";
}

std::vector<std::string> ChineseRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first(
      {"zh", "zh-Hans", "zh_CN", "zh-CN", "zh_hans", "cmn", "Chinese"});
}

std::filesystem::path resolve_chinese_dict_path(const std::filesystem::path& model_root) {
  const std::filesystem::path under_data = model_root.parent_path() / "data" / "zh_hans" / "dict.tsv";
  if (std::filesystem::is_regular_file(under_data)) {
    return under_data;
  }
  return model_root / "zh_hans" / "dict.tsv";
}

std::filesystem::path resolve_chinese_onnx_model_dir(const std::filesystem::path& model_root) {
  const auto try_dir = [](const std::filesystem::path& dir) -> std::optional<std::filesystem::path> {
    const auto onnx = resolve_prefer_ort_model(dir, "model.ort");
    if (std::filesystem::is_regular_file(onnx)) {
      return dir;
    }
    return std::nullopt;
  };
  if (auto o = try_dir(model_root.parent_path() / "data" / "zh_hans" / "roberta_chinese_base_upos_onnx")) {
    return *o;
  }
  if (auto o = try_dir(model_root.parent_path().parent_path() / "data" / "zh_hans" /
                       "roberta_chinese_base_upos_onnx")) {
    return *o;
  }
  return model_root / "zh_hans" / "roberta_chinese_base_upos_onnx";
}

}  // namespace moonshine_tts
