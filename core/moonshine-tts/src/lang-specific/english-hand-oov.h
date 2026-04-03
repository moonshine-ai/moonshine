#ifndef MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_HAND_OOV_H
#define MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_HAND_OOV_H

#include <string>
#include <string_view>

namespace moonshine_tts {

/// Greedy grapheme rules for English OOV (mirrors ``english_oov_rules_ipa`` in ``english_rule_g2p.py``).
std::string english_hand_oov_rules_ipa(std::string_view word);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_HAND_OOV_H
