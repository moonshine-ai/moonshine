#ifndef MOONSHINE_TTS_LANG_SPECIFIC_UKRAINIAN_H
#define MOONSHINE_TTS_LANG_SPECIFIC_UKRAINIAN_H

#include "rule-based-g2p.h"

#include <string>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Rule-based Ukrainian G2P (mirrors ``ukrainian_rule_g2p.py``): Cyrillic + palatalization lookahead,
/// apostrophe blocking, **г/ґ**, **в** allophony, penultimate stress heuristic, optional digit expansion.
class UkrainianRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    bool expand_cardinal_digits = true;
  };

  UkrainianRuleG2p();
  explicit UkrainianRuleG2p(Options options);

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }
  bool with_stress() const { return options_.with_stress; }
  bool expand_cardinal_digits() const { return options_.expand_cardinal_digits; }

  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                            std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::string dialect_id_{"uk-UA"};
  Options options_;

  std::string text_to_ipa_no_expand(const std::string& text,
                                    std::vector<G2pWordLog>* per_word_log) const;
};

bool dialect_resolves_to_ukrainian_rules(std::string_view dialect_id);

std::string ukrainian_word_to_ipa(const std::string& word, bool with_stress = true,
                                  bool expand_cardinal_digits = true);

std::string ukrainian_text_to_ipa(const std::string& text, bool with_stress = true,
                                  std::vector<G2pWordLog>* per_word_log = nullptr,
                                  bool expand_cardinal_digits = true);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_UKRAINIAN_H
