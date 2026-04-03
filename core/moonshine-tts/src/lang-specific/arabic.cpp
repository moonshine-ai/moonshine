#include "arabic.h"
#include "arabic-ipa.h"
#include "utf8-utils.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace moonshine_tts {

namespace {

std::filesystem::path absolute_model_root_ar(const std::filesystem::path& model_root) {
  if (model_root.is_relative()) {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::weakly_canonical(std::filesystem::absolute(model_root), ec);
    if (!ec) {
      return p;
    }
    return std::filesystem::absolute(model_root);
  }
  return model_root;
}

bool has_arabic_script(std::string_view s) {
  std::string tmp(s);
  for (size_t i = 0; i < tmp.size();) {
    char32_t cp = 0;
    size_t adv = 0;
    if (!utf8_decode_at(tmp, i, cp, adv) || adv == 0) {
      ++i;
      continue;
    }
    const std::uint32_t o = static_cast<std::uint32_t>(cp);
    if (o >= 0x0600u && o <= 0x06FFu) {
      return true;
    }
    i += adv;
  }
  return false;
}

std::unordered_map<std::string, std::string> load_lex_first_tsv(const std::filesystem::path& p) {
  std::unordered_map<std::string, std::string> lex;
  std::ifstream in(p);
  if (!in) {
    return lex;
  }
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto t = line.find_first_not_of(" \t");
    if (t == std::string::npos || line[t] == '#') {
      continue;
    }
    const auto tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string w = trim_ascii_ws_copy(line.substr(0, tab));
    std::string ipa_col = trim_ascii_ws_copy(line.substr(tab + 1));
    if (w.empty()) {
      continue;
    }
    std::istringstream iss(ipa_col);
    std::string ipa0;
    if (!(iss >> ipa0)) {
      continue;
    }
    if (lex.count(w) == 0) {
      lex.emplace(std::move(w), std::move(ipa0));
    }
  }
  return lex;
}

}  // namespace

ArabicRuleG2p::ArabicRuleG2p(std::filesystem::path onnx_model_dir, std::filesystem::path dict_tsv,
                             bool use_cuda)
    : diac_(std::make_unique<ArabicDiacOnnx>(std::move(onnx_model_dir), use_cuda)),
      lex_(load_lex_first_tsv(std::move(dict_tsv))) {}

std::vector<std::string> ArabicRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first(
      {"ar", "ar-MSA", "ar_msa", "ar-msa", "arabic", "Arabic", "msa"});
}

bool dialect_resolves_to_arabic_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "ar" || s == "ar-msa" || s == "ar_msa" || s == "arabic" || s == "msa";
}

std::filesystem::path resolve_arabic_dict_path(const std::filesystem::path& model_root) {
  const std::filesystem::path base = absolute_model_root_ar(model_root);
  const std::filesystem::path p1 = base.parent_path() / "data" / "ar_msa" / "dict.tsv";
  if (std::filesystem::is_regular_file(p1)) {
    return p1;
  }
  const std::filesystem::path p2 = base.parent_path().parent_path() / "data" / "ar_msa" / "dict.tsv";
  if (std::filesystem::is_regular_file(p2)) {
    return p2;
  }
  return base / "ar_msa" / "dict.tsv";
}

std::filesystem::path resolve_arabic_onnx_model_dir(const std::filesystem::path& model_root) {
  const std::filesystem::path base = absolute_model_root_ar(model_root);
  const auto try_dir = [](const std::filesystem::path& dir) -> std::optional<std::filesystem::path> {
    const auto onnx = dir / "model.onnx";
    if (std::filesystem::is_regular_file(onnx)) {
      return dir;
    }
    return std::nullopt;
  };
  if (auto o = try_dir(base.parent_path() / "data" / "ar_msa" / "arabertv02_tashkeel_fadel_onnx")) {
    return *o;
  }
  if (auto o = try_dir(base.parent_path().parent_path() / "data" / "ar_msa" /
                       "arabertv02_tashkeel_fadel_onnx")) {
    return *o;
  }
  return base / "ar_msa" / "arabertv02_tashkeel_fadel_onnx";
}

std::string ArabicRuleG2p::g2p_word(std::string_view word_utf8) {
  const std::string w = trim_ascii_ws_copy(word_utf8);
  if (w.empty() || !has_arabic_script(w)) {
    return "";
  }
  const std::string key = arabic_msa_strip_diacritics_utf8(w);
  const auto it = lex_.find(key);
  if (it != lex_.end()) {
    return it->second;
  }
  const std::string diac = diac_->diacritize(w);
  const std::string filled = arabic_msa_apply_onnx_partial_postprocess_utf8(diac);
  return arabic_msa_word_to_ipa_with_assimilation_utf8(filled, w);
}

std::string ArabicRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  (void)per_word_log;
  std::string raw = trim_ascii_ws_copy(text);
  if (raw.empty()) {
    return "";
  }
  std::vector<std::string> ipa_words;
  std::size_t i = 0;
  while (i < raw.size()) {
    while (i < raw.size() && std::isspace(static_cast<unsigned char>(raw[i])) != 0) {
      ++i;
    }
    if (i >= raw.size()) {
      break;
    }
    std::size_t j = i;
    while (j < raw.size() && std::isspace(static_cast<unsigned char>(raw[j])) == 0) {
      ++j;
    }
    const std::string tok = raw.substr(i, j - i);
    i = j;
    const std::string ipa = g2p_word(tok);
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
