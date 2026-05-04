#ifndef MOONSHINE_TTS_LANG_SPECIFIC_CHINESE_H
#define MOONSHINE_TTS_LANG_SPECIFIC_CHINESE_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Simplified Chinese lexicon G2P (``data/zh_hans/dict.tsv`` ipa-dict IPA), mirroring
/// ``chinese_rule_g2p.ChineseOnnxLexiconG2p`` **without** ONNX segmentation: ``text_to_ipa``
/// splits on whitespace; **unbroken Han substrings** are one lookup (then per-character
/// fallback). Optional **CTB POS** on ``word_to_ipa_with_pos`` disambiguates common
/// polyphones (行/了/没/…). Arabic numerals expand to Mandarin cardinals like the Python layer.
class ChineseRuleG2p : public RuleBasedG2p {
 public:
  explicit ChineseRuleG2p(std::filesystem::path dict_tsv);
  explicit ChineseRuleG2p(std::string dict_tsv_utf8);

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  /// Same as Python ``g2p_word`` with no POS (first lexicon reading when ambiguous).
  std::string word_to_ipa(std::string_view word) const;

  /// Pass CTB-style or UD UPOS tag (e.g. ``NN``, ``NOUN``, ``VV``, ``VERB``) for heteronym hints;
  /// empty *pos* → default.
  std::string word_to_ipa_with_pos(std::string_view word, std::string_view pos) const;

  std::string text_to_ipa(std::string text,
                            std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::string dialect_id_{"zh-Hans"};
  std::unordered_map<std::string, std::vector<std::string>> lex_;

  std::string han_reading_to_ipa(std::string_view han) const;
  std::string char_fallback_ipa(std::string_view word) const;
  std::string disambiguate_heteronym(std::string_view word,
                                     std::string_view pos,
                                     const std::vector<std::string>& readings) const;
  std::string g2p_word_impl(std::string_view word, std::string_view pos) const;
};

bool dialect_resolves_to_chinese_rules(std::string_view dialect_id);

/// ``<model-root>/zh_hans/dict.tsv``.
std::filesystem::path resolve_chinese_dict_path(const std::filesystem::path& model_root);

/// Directory with Chinese RoBERTa UPOS ONNX (``model.onnx``, ``vocab.txt``, …):
/// ``<model-root>/zh_hans/roberta_chinese_base_upos_onnx``.
std::filesystem::path resolve_chinese_onnx_model_dir(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_CHINESE_H
