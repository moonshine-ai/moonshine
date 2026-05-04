#ifndef MOONSHINE_TTS_LANG_SPECIFIC_RUSSIAN_H
#define MOONSHINE_TTS_LANG_SPECIFIC_RUSSIAN_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Rule- and lexicon-based Russian G2P, mirroring ``russian_rule_g2p.py``.
class RussianRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    /// When true (default), move ˈ/ˌ before the syllable nucleus (same as Python / German helper).
    bool vocoder_stress = true;
    /// Expand ASCII ``\\b\\d+\\b`` and ``\\b\\d+-\\d+\\b`` to Russian cardinal words before G2P.
    bool expand_cardinal_digits = true;
  };

  explicit RussianRuleG2p(std::filesystem::path dict_tsv);
  explicit RussianRuleG2p(std::filesystem::path dict_tsv, Options options);
  explicit RussianRuleG2p(std::string dict_tsv_utf8, Options options);
  explicit RussianRuleG2p(std::string dict_tsv_utf8)
      : RussianRuleG2p(std::move(dict_tsv_utf8), Options{}) {}

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::string dialect_id_{"ru-RU"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  std::string lookup_or_rules(const std::string& raw_word) const;
  std::string finalize_ipa(std::string ipa) const;
  std::string text_to_ipa_no_expand(const std::string& text,
                                    std::vector<G2pWordLog>* per_word_log) const;
};

/// True for ``ru``, ``ru-RU``, ``russian`` (case-insensitive).
bool dialect_resolves_to_russian_rules(std::string_view dialect_id);

/// Default lexicon: ``<model-root>/../data/ru/dict.tsv`` then ``<model-root>/ru/dict.tsv``.
std::filesystem::path resolve_russian_dict_path(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_RUSSIAN_H
