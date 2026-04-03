#ifndef MOONSHINE_TTS_RULE_BASED_G2P_H
#define MOONSHINE_TTS_RULE_BASED_G2P_H

#include <string>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// Shared interface for lexicon + rules G2P backends used by ``MoonshineG2P``.
class RuleBasedG2p {
 public:
  virtual ~RuleBasedG2p() = default;

  virtual std::string text_to_ipa(std::string text,
                                  std::vector<G2pWordLog>* per_word_log = nullptr) = 0;
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_RULE_BASED_G2P_H
