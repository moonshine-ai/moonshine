#ifndef MOONSHINE_TTS_LANG_SPECIFIC_HINDI_NUMBERS_H
#define MOONSHINE_TTS_LANG_SPECIFIC_HINDI_NUMBERS_H

#include <string>
#include <string_view>

namespace moonshine_tts {

/// ASCII digits only → Hindi cardinal words (Devanagari), mirroring ``hindi_numbers.expand_cardinal_digits_to_hindi_words``.
std::string expand_cardinal_digits_to_hindi_words(std::string_view s);

/// Expand ``\\b\\d+-\\d+\\b`` and ``\\b\\d+\\b`` in *text* (ASCII digit tokens).
std::string expand_hindi_digit_tokens_in_text(std::string text);

/// Replace runs of Devanagari digits (U+0966–U+096F) with Hindi cardinal words.
std::string expand_devanagari_digit_runs_in_text(std::string text);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_HINDI_NUMBERS_H
