#ifndef MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_NUMBERS_H
#define MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_NUMBERS_H

#include <optional>
#include <string>
#include <string_view>

namespace moonshine_tts {

/// If *token* is a supported plain numeral, return IPA; else ``nullopt`` (see ``english_number_token_ipa``).
std::optional<std::string> english_number_token_ipa(std::string_view token);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_NUMBERS_H
