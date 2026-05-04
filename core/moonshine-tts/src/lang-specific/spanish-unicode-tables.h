#ifndef MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_UNICODE_TABLES_H
#define MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_UNICODE_TABLES_H

#include <cstddef>
#include <cstdint>
#include <utility>

namespace moonshine_tts::spanish_unicode {

// Definitions in spanish_unicode_tables.cpp (generated Unicode data).
extern const std::pair<char32_t, const char*> k_unicode_strip_table[];
extern const std::size_t k_unicode_strip_table_size;
extern const std::pair<char32_t, const char*> k_unicode_lower_table[];
extern const std::size_t k_unicode_lower_table_size;
extern const std::uint32_t k_unicode_word_bitmap[];
extern const std::uint32_t k_unicode_word_bitmap_words;
extern const std::uint32_t k_unicode_space_bitmap[];
extern const std::uint32_t k_unicode_space_bitmap_words;

}  // namespace moonshine_tts::spanish_unicode

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_SPANISH_UNICODE_TABLES_H
