#ifndef MOONSHINE_TTS_LANG_SPECIFIC_DUTCH_H
#define MOONSHINE_TTS_LANG_SPECIFIC_DUTCH_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Rule- and lexicon-based Dutch G2P, mirroring ``dutch_rule_g2p.py`` / ``dutch_numbers.py``.
class DutchRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    /// When true (default), move ˈ/ˌ before the syllable nucleus for **rule-based** IPA only.
    bool vocoder_stress = true;
    /// Expand ``\b\d+\b`` and ``\b\d+-\d+\b`` to Dutch cardinal words before G2P.
    bool expand_cardinal_digits = true;
  };

  explicit DutchRuleG2p(std::filesystem::path dict_tsv);
  explicit DutchRuleG2p(std::filesystem::path dict_tsv, Options options);
  explicit DutchRuleG2p(std::string dict_tsv_utf8, Options options);
  explicit DutchRuleG2p(std::string dict_tsv_utf8)
      : DutchRuleG2p(std::move(dict_tsv_utf8), Options{}) {}

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

  /// Same algorithm as Python ``normalize_ipa_stress_for_vocoder`` (Dutch nucleus list).
  static std::string normalize_ipa_stress_for_vocoder(std::string ipa);

 private:
  std::string dialect_id_{"nl-NL"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  std::string lookup_or_rules(const std::string& raw_word) const;
  std::string finalize_ipa(std::string ipa, bool from_lexicon) const;
  std::string text_to_ipa_no_expand(const std::string& text,
                                    std::vector<G2pWordLog>* per_word_log) const;
};

/// True for ``nl``, ``nl-NL``, ``nl_nl``, ``dutch`` (case-insensitive).
bool dialect_resolves_to_dutch_rules(std::string_view dialect_id);

/// Default lexicon path: ``<model-root>/../data/nl/dict.tsv`` then ``<model-root>/nl/dict.tsv``.
std::filesystem::path resolve_dutch_dict_path(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_DUTCH_H
