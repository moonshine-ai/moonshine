#ifndef MOONSHINE_TTS_LANG_SPECIFIC_CHINESE_NUMBERS_H
#define MOONSHINE_TTS_LANG_SPECIFIC_CHINESE_NUMBERS_H

#include <cstdint>
#include <optional>
#include <string>

namespace moonshine_tts {

/// Mandarin cardinal Han characters for ``0 <= n < 10^16`` (larger → digit-by-digit).
std::string int_to_mandarin_cardinal_han(std::uint64_t n);

/// Parse ASCII / fullwidth Arabic numeral token → Han reading, or nullopt if not a numeral.
std::optional<std::string> arabic_numeral_token_to_han(std::string_view token);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_CHINESE_NUMBERS_H
