#ifndef MOONSHINE_TTS_LANG_SPECIFIC_KOREAN_H
#define MOONSHINE_TTS_LANG_SPECIFIC_KOREAN_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Lexicon + Hangul rule G2P (연음, 유음화, 비음화, ㅎ aspiration, 경음화), mirroring ``korean_rule_g2p.py``.
class KoreanRuleG2p : public RuleBasedG2p {
 public:
  struct Options {
    std::string syllable_sep{""};
    bool expand_cardinal_digits = true;
    bool log_g2p = true;  // log grapheme→IPA mappings to stderr via LOGF
  };

  explicit KoreanRuleG2p(std::filesystem::path dict_tsv);
  explicit KoreanRuleG2p(std::filesystem::path dict_tsv, Options options);
  explicit KoreanRuleG2p(std::string dict_tsv_utf8, Options options);
  explicit KoreanRuleG2p(std::string dict_tsv_utf8)
      : KoreanRuleG2p(std::move(dict_tsv_utf8), Options{}) {}

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  /// Map lexicon-style IPA to the broad inventory used by rule output (same as Python
  /// ``normalize_korean_ipa``).  When ``voice_lenis`` is true (default, used for lexicon
  /// entries), k/t/p are converted to voiced ɡ/d/b.  When false (rule-based output),
  /// voicing is skipped because ``syllables_to_ipa`` already produces the correct forms.
  static std::string normalize_korean_ipa(std::string ipa, bool voice_lenis = true);

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::string dialect_id_{"ko-KR"};
  Options options_;
  std::unordered_map<std::string, std::string> lexicon_;

  std::string g2p_single_fragment(std::string_view frag) const;
  std::string g2p_hangul_rules_only(std::string_view hangul) const;
  std::string extract_hangul(std::string_view s) const;
};

bool dialect_resolves_to_korean_rules(std::string_view dialect_id);

/// ``<model-root>/../data/ko/dict.tsv`` or ``<model-root>/ko/dict.tsv``.
std::filesystem::path resolve_korean_dict_path(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_KOREAN_H
