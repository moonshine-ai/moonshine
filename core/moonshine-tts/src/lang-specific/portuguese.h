#ifndef MOONSHINE_TTS_LANG_SPECIFIC_PORTUGUESE_H
#define MOONSHINE_TTS_LANG_SPECIFIC_PORTUGUESE_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Rule- and lexicon-based Portuguese G2P (Brazil / Portugal), mirroring ``portuguese_rule_g2p.py``
/// and ``portuguese_numbers.py``.
class PortugueseRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    bool with_stress = true;
    bool vocoder_stress = true;
    bool keep_syllable_dots = false;
    bool apply_pt_pt_final_esh = true;
    bool expand_cardinal_digits = true;
  };

  explicit PortugueseRuleG2p(std::filesystem::path dict_tsv, bool is_portugal);
  explicit PortugueseRuleG2p(std::filesystem::path dict_tsv, bool is_portugal, Options options);

  static std::vector<std::string> dialect_ids();

  bool is_portugal() const { return is_portugal_; }
  const std::string& dialect_id() const { return dialect_id_; }

  std::string word_to_ipa(const std::string& word) const;

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  bool is_portugal_{false};
  std::string dialect_id_{"pt-BR"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  std::string lookup_or_rules(const std::string& raw_word) const;
  std::string finalize_ipa(std::string ipa, bool from_lexicon) const;
  std::string text_to_ipa_no_expand(const std::string& text,
                                    std::vector<G2pWordLog>* per_word_log) const;
};

/// True for ``pt``, ``pt-PT``, ``pt_pt``, ``portugal``, etc. (European); false for Brazil variants.
bool dialect_resolves_to_portugal_rules(std::string_view dialect_id);

/// True for ``pt-BR``, ``pt_br``, ``brazil``, etc.
bool dialect_resolves_to_brazilian_portuguese_rules(std::string_view dialect_id);

/// Default lexicon: ``<model-root>/../data/pt_br/dict.tsv`` or ``.../pt_pt/dict.tsv``.
std::filesystem::path resolve_portuguese_dict_path(const std::filesystem::path& model_root,
                                                    bool is_portugal);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_PORTUGUESE_H
