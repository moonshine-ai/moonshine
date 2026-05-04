#ifndef MOONSHINE_TTS_LANG_SPECIFIC_HINDI_H
#define MOONSHINE_TTS_LANG_SPECIFIC_HINDI_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Hindi Devanagari G2P: ``dict.tsv`` lookup + rule-based parsing (mirrors ``hindi_rule_g2p.py``).
class HindiRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    bool expand_cardinal_digits = true;
  };

  explicit HindiRuleG2p(std::filesystem::path dict_tsv);
  explicit HindiRuleG2p(std::filesystem::path dict_tsv, Options options);
  explicit HindiRuleG2p(std::string dict_tsv_utf8, Options options);
  explicit HindiRuleG2p(std::string dict_tsv_utf8)
      : HindiRuleG2p(std::move(dict_tsv_utf8), Options{}) {}

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                            std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::string dialect_id_{"hi-IN"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  std::string g2p_single_word(std::string_view word) const;
  std::string text_to_ipa_no_expand(std::string text,
                                      std::vector<G2pWordLog>* per_word_log) const;
};

bool dialect_resolves_to_hindi_rules(std::string_view dialect_id);

/// Lexicon path: ``<model-root>/hi/dict.tsv``.
std::filesystem::path resolve_hindi_dict_path(const std::filesystem::path& model_root);

/// Default lexicon path when no explicit root is supplied: ``<cwd>/hi/dict.tsv``.
std::filesystem::path builtin_hindi_dict_path();

/// Convenience for tests / CLI. When ``dict_tsv`` is empty, uses ``builtin_hindi_dict_path()``.
std::string hindi_text_to_ipa(const std::string& text, bool with_stress = true,
                                std::vector<G2pWordLog>* per_word_log = nullptr,
                                bool expand_cardinal_digits = true,
                                const std::filesystem::path& dict_tsv = {});

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_HINDI_H
