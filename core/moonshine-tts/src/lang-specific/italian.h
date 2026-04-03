#ifndef MOONSHINE_TTS_LANG_SPECIFIC_ITALIAN_H
#define MOONSHINE_TTS_LANG_SPECIFIC_ITALIAN_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Rule- and lexicon-based Italian G2P, mirroring ``italian_rule_g2p.py`` / ``italian_numbers.py``.
class ItalianRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    /// When true (default), move ˈ/ˌ before the syllable nucleus for **rule-based** IPA only.
    bool vocoder_stress = true;
    /// Expand ``\b\d+\b`` and ``\b\d+-\d+\b`` to Italian cardinal words before G2P.
    bool expand_cardinal_digits = true;
  };

  explicit ItalianRuleG2p(std::filesystem::path dict_tsv);
  explicit ItalianRuleG2p(std::filesystem::path dict_tsv, Options options);

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::string dialect_id_{"it-IT"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  std::string lookup_or_rules(const std::string& raw_word) const;
  std::string finalize_ipa(std::string ipa, bool from_lexicon) const;
  std::string text_to_ipa_no_expand(const std::string& text,
                                    std::vector<G2pWordLog>* per_word_log) const;
};

/// True for ``it``, ``it-IT``, ``it_it``, ``italian`` (case-insensitive).
bool dialect_resolves_to_italian_rules(std::string_view dialect_id);

/// Default lexicon: ``<model-root>/../data/it/dict.tsv`` then ``<model-root>/it/dict.tsv``.
std::filesystem::path resolve_italian_dict_path(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_ITALIAN_H
