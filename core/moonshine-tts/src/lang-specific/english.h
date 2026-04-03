#ifndef MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_H
#define MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// US English lexicon + homograph merge + optional ONNX heteronym/OOV + hand OOV fallback
/// (mirrors ``english_rule_g2p.EnglishLexiconRuleG2p`` + ``moonshine_onnx_g2p`` heteronym/OOV wiring).
class EnglishRuleG2p : public RuleBasedG2p {
 public:
  EnglishRuleG2p(std::filesystem::path dict_tsv,
                 std::filesystem::path homograph_json,
                 std::optional<std::filesystem::path> heteronym_onnx,
                 std::optional<std::filesystem::path> oov_onnx,
                 bool use_cuda = false);
  ~EnglishRuleG2p() override;

  EnglishRuleG2p(EnglishRuleG2p&&) noexcept;
  EnglishRuleG2p& operator=(EnglishRuleG2p&&) noexcept;

  EnglishRuleG2p(const EnglishRuleG2p&) = delete;
  EnglishRuleG2p& operator=(const EnglishRuleG2p&) = delete;

  static std::vector<std::string> dialect_ids();

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// True for ``en_us``, ``en-us``, ``en``, ``english`` (case-insensitive).
bool dialect_resolves_to_english_rules(std::string_view dialect_id);

/// Default: ``<model-root>/en_us/dict_filtered_heteronyms.tsv``.
std::filesystem::path resolve_english_dict_path(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_H
