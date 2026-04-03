#ifndef MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_UNICODE_H
#define MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_UNICODE_H

#include <cstdint>
#include <string>

namespace moonshine_tts::spanish_unicode {

std::string utf32_to_utf8(const std::u32string& u);
std::u32string utf8_to_utf32(const std::string& s);
std::string unicode_lower_utf8(const std::string& s);
std::string strip_accents_utf8(const std::string& s);
std::string word_key(const std::string& wraw);
bool is_word_char(char32_t cp);
bool is_space_char(char32_t cp);
/// UTF-8 replacement from the strip table, or nullptr when absent.
const char* strip_replacement_utf8(char32_t cp);

}  // namespace moonshine_tts::spanish_unicode

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_UNICODE_H
