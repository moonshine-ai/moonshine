#ifndef MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_INTERNAL_H
#define MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_INTERNAL_H

#include <string>

namespace moonshine_tts::french_detail {

/// Rule-based OOV IPA (mirrors ``french_oov_rules.oov_word_to_ipa``).
std::string oov_word_to_ipa(const std::string& word, bool with_stress);

}  // namespace moonshine_tts::french_detail

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_FRENCH_INTERNAL_H
