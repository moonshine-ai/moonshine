#ifndef MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_H
#define MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Lexicon + liaison + OOV rules + cardinal digit expansion (mirrors ``french_g2p.py``).
class FrenchRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    bool liaison = true;
    bool liaison_optional = true;
    bool oov_rules = true;
    bool expand_cardinal_digits = true;
  };

  /// Load ``word<TAB>IPA`` TSV and POS CSV inventory from *csv_dir* (if directory exists).
  explicit FrenchRuleG2p(std::filesystem::path dict_tsv, std::filesystem::path csv_dir);
  explicit FrenchRuleG2p(std::filesystem::path dict_tsv, std::filesystem::path csv_dir, Options options);

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

  /// Exported for tests (same as Python :func:`ensure_french_nuclear_stress`).
  static std::string ensure_french_nuclear_stress(std::string ipa);

 private:
  std::string dialect_id_{"fr-FR"};
  Options options_;
  std::filesystem::path csv_dir_;
  std::unordered_map<std::string, std::string> lexicon_;
  std::unordered_map<std::string, std::unordered_set<std::string>> pos_by_cat_;

  std::string text_to_ipa_impl(const std::string& text, bool expand_digits,
                               std::vector<G2pWordLog>* per_word_log) const;

  std::string word_to_ipa_impl(const std::string& raw_word, bool expand_digits) const;
  std::string finalize_word_ipa(std::string ipa, bool from_compound) const;
};

/// True for ``fr``, ``fr-FR``, ``fr_fr``, ``french`` (case-insensitive).
bool dialect_resolves_to_french_rules(std::string_view dialect_id);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_H
