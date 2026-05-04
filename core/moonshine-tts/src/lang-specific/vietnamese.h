#ifndef MOONSHINE_TTS_LANG_SPECIFIC_VIETNAMESE_H
#define MOONSHINE_TTS_LANG_SPECIFIC_VIETNAMESE_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Vietnamese lexicon + greedy longest-match + OOV syllable rules (Northern IPA, ``data/vi/dict.tsv``).
class VietnameseRuleG2p : public RuleBasedG2p {
 public:
  explicit VietnameseRuleG2p(std::filesystem::path dict_tsv);
  explicit VietnameseRuleG2p(std::string dict_tsv_utf8);

  static std::vector<std::string> dialect_ids();

  /// One surface token (no spaces); mirrors :func:`vietnamese_rule_g2p.vietnamese_word_to_ipa`.
  std::string word_to_ipa(std::string_view word) const;

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

  /// Exposed for tests / CLI parity with Python ``vietnamese_syllable_to_ipa``.
  static std::string syllable_to_ipa(std::string_view syllable_utf8);

 private:
  std::unordered_map<std::string, std::string> lex_;
  int max_key_words_{1};

  std::string g2p_single_token(std::string_view token) const;
};

bool dialect_resolves_to_vietnamese_rules(std::string_view dialect_id);

std::filesystem::path resolve_vietnamese_dict_path(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_VIETNAMESE_H
