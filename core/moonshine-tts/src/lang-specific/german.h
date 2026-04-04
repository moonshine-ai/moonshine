#ifndef MOONSHINE_TTS_LANG_SPECIFIC_GERMAN_H
#define MOONSHINE_TTS_LANG_SPECIFIC_GERMAN_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Rule- and lexicon-based German G2P (High German), mirroring ``german_rule_g2p.py``.
class GermanRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    /// When true (default), move ˈ/ˌ before the syllable nucleus (vocoder-style).
    bool vocoder_stress = true;
    /// Expand ``\\b\\d+\\b`` and year-style ``\\b\\d+-\\d+\\b`` to German cardinal words before G2P.
    bool expand_cardinal_digits = true;
  };

  /// Load ``word<TAB>IPA`` TSV; throws ``std::runtime_error`` if the file is missing/unreadable.
  explicit GermanRuleG2p(std::filesystem::path dict_tsv);
  explicit GermanRuleG2p(std::filesystem::path dict_tsv, Options options);
  explicit GermanRuleG2p(std::string dict_tsv_utf8, Options options);
  explicit GermanRuleG2p(std::string dict_tsv_utf8)
      : GermanRuleG2p(std::move(dict_tsv_utf8), Options{}) {}

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  /// Single surface word (no spaces); punctuation should be stripped by the caller.
  std::string word_to_ipa(const std::string& word) const;

  /// Tokenize like Python ``text_to_ipa`` (German letters, digits, hyphen; preserve other chars).
  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

  /// Exported for tests / tooling (same as the Python helper).
  static std::string normalize_ipa_stress_for_vocoder(std::string ipa);

 private:
  std::string dialect_id_{"de-DE"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  std::string lookup_or_rules(const std::string& raw_word) const;
  std::string finalize_ipa(std::string ipa) const;
  std::string text_to_ipa_no_expand(const std::string& text,
                                    std::vector<G2pWordLog>* per_word_log) const;
};

/// True for ``de``, ``de-DE``, ``de_de``, ``german`` (case-insensitive).
bool dialect_resolves_to_german_rules(std::string_view dialect_id);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_GERMAN_H
