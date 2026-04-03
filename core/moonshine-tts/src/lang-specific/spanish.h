#ifndef MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_H
#define MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_H

#include "rule-based-g2p.h"

#include <string>
#include <vector>

namespace moonshine_tts {

/// Pronunciation preset for rule-based Spanish G2P (mirrors ``spanish_rule_g2p.SpanishDialect``).
struct SpanishDialect {
  std::string id;
  std::string ce_ci_z_ipa; // "s" (seseo) or "θ" (distinción)
  bool yeismo = true;
  std::string y_consonant_ipa; // e.g. "ʝ" or "ʒ"
  std::string ll_ipa;          // e.g. "ʎ" when yeismo is false
  std::string x_intervocalic_default = "ks";
  std::string x_initial_before_vowel = "s";
  std::string voiceless_velar_fricative; // "x" or "h"
  std::string trill_ipa = "r";
  std::string tap_ipa; // e.g. "ɾ"
  bool nasal_assimilation = false;
  bool narrow_intervocalic_obstruents = true;
  enum class CodaS { Keep, H, Drop } coda_s_mode = CodaS::Keep;
};

/// Rule-based Spanish G2P (mirrors ``spanish_rule_g2p.py``).
class SpanishRuleG2p : public RuleBasedG2p {
 public:
  SpanishRuleG2p(SpanishDialect dialect, bool with_stress, bool expand_cardinal_digits = true);

  static std::vector<std::string> dialect_ids();

  const SpanishDialect& dialect() const { return dialect_; }
  bool with_stress() const { return with_stress_; }
  bool expand_cardinal_digits() const { return expand_cardinal_digits_; }

  /// Single surface word (no spaces); punctuation should be stripped by the caller.
  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  SpanishDialect dialect_;
  bool with_stress_;
  bool expand_cardinal_digits_{true};

  std::string text_to_ipa_no_expand(const std::string& text,
                                    std::vector<G2pWordLog>* per_word_log) const;
};

/// Sorted CLI ids (same set as Python ``dialect_ids()``).
std::vector<std::string> spanish_dialect_cli_ids();

/// Build dialect from CLI id, or throw ``std::invalid_argument``.
SpanishDialect spanish_dialect_from_cli_id(const std::string& cli_id,
                                           bool narrow_intervocalic_obstruents = true);

/// One word → IPA (letters outside Spanish set stripped; punctuation should be handled by caller).
std::string spanish_word_to_ipa(const std::string& word, const SpanishDialect& dialect,
                                bool with_stress = true, bool expand_cardinal_digits = true);

/// Tokenize like Python ``text_to_ipa`` (Spanish letters + digits; collapse spaces).
/// If *per_word_log* is non-null, appends one entry per Spanish word span (rule-based path).
std::string spanish_text_to_ipa(const std::string& text, const SpanishDialect& dialect,
                                bool with_stress = true,
                                std::vector<G2pWordLog>* per_word_log = nullptr,
                                bool expand_cardinal_digits = true);

} // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_H
