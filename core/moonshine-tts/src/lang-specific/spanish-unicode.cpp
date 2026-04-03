#include "spanish-unicode.h"
#include "spanish-unicode-tables.h"
#include "utf8-utils.h"

#include <algorithm>
#include <string>

namespace moonshine_tts::spanish_unicode {
namespace {

const char* lookup_sorted_pair(const std::pair<char32_t, const char*>* table, size_t n, char32_t key) {
  const auto* first = table;
  const auto* last = table + n;
  const auto* it =
      std::lower_bound(first, last, key, [](const std::pair<char32_t, const char*>& e, char32_t k) {
        return e.first < k;
      });
  if (it != last && it->first == key) {
    return it->second;
  }
  return nullptr;
}

bool unicode_bitmap_get(const std::uint32_t* bitmap, std::uint32_t nwords, char32_t cp) {
  if (cp >= 0x110000) {
    return false;
  }
  const std::uint32_t wi = static_cast<std::uint32_t>(cp) / 32u;
  if (wi >= nwords) {
    return false;
  }
  return (bitmap[wi] & (1u << (static_cast<std::uint32_t>(cp) % 32u))) != 0u;
}

}  // namespace

std::string utf32_to_utf8(const std::u32string& u) {
  std::string out;
  out.reserve(u.size() * 3);
  for (char32_t cp : u) {
    utf8_append_codepoint(out, cp);
  }
  return out;
}

std::u32string utf8_to_utf32(const std::string& s) {
  return utf8_str_to_u32(s);
}

std::string unicode_lower_utf8(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, i, cp, adv);
    const char* rep = lookup_sorted_pair(k_unicode_lower_table, k_unicode_lower_table_size, cp);
    if (rep != nullptr) {
      out += rep;
    } else {
      utf8_append_codepoint(out, cp);
    }
    i += adv;
  }
  return out;
}

std::string strip_accents_utf8(const std::string& s) {
  std::string out;
  size_t i = 0;
  while (i < s.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, i, cp, adv);
    const char* rep = lookup_sorted_pair(k_unicode_strip_table, k_unicode_strip_table_size, cp);
    if (rep != nullptr) {
      out += rep;
    } else {
      utf8_append_codepoint(out, cp);
    }
    i += adv;
  }
  return out;
}

std::string word_key(const std::string& wraw) {
  return strip_accents_utf8(unicode_lower_utf8(wraw));
}

bool is_word_char(char32_t cp) {
  return unicode_bitmap_get(k_unicode_word_bitmap, k_unicode_word_bitmap_words, cp);
}

bool is_space_char(char32_t cp) {
  return unicode_bitmap_get(k_unicode_space_bitmap, k_unicode_space_bitmap_words, cp);
}

const char* strip_replacement_utf8(char32_t cp) {
  return lookup_sorted_pair(k_unicode_strip_table, k_unicode_strip_table_size, cp);
}

}  // namespace moonshine_tts::spanish_unicode
