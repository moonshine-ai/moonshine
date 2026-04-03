#ifndef MOONSHINE_TTS_UTF8_UTILS_H
#define MOONSHINE_TTS_UTF8_UTILS_H

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moonshine_tts {

void utf8_append_codepoint(std::string& out, char32_t cp);

/// Decode one UTF-8 code point starting at byte index *i* in *s*. On success returns true and
/// advances *out_len* bytes (always >= 1). Mirrors the lenient scan used across rule-based G2P.
bool utf8_decode_at(const std::string& s, size_t i, char32_t& out_cp, size_t& out_len);

/// Full-string scan to UTF-32 (invalid sequences yield one code unit per leading byte, same as
/// :func:`utf8_decode_at`).
std::u32string utf8_str_to_u32(const std::string& s);

/// Remove every occurrence of *sub* from *s*.
inline void erase_utf8_substr(std::string& s, std::string_view sub) {
  if (sub.empty()) {
    return;
  }
  for (;;) {
    const size_t p = s.find(sub);
    if (p == std::string::npos) {
      break;
    }
    s.erase(p, sub.size());
  }
}

/// Trim ASCII whitespace only (same policy as legacy ``trim_copy_sv`` helpers in language files).
inline std::string trim_ascii_ws_copy(std::string_view s) {
  size_t a = 0;
  size_t b = s.size();
  while (a < b && std::isspace(static_cast<unsigned char>(s[a])) != 0) {
    ++a;
  }
  while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1])) != 0) {
    --b;
  }
  return std::string(s.substr(a, b - a));
}

std::vector<std::string> utf8_split_codepoints(const std::string& utf8);

// Find *token* as a contiguous subsequence of code points in *text*, starting at
// code point index >= start_cp. Returns [start, end) code point indices into *text*.
std::optional<std::pair<int, int>> utf8_find_token_codepoints(const std::string& text,
                                                              const std::string& token,
                                                              int start_cp);

/// Rough Python ``\\w`` neighbor for ``re`` digit spans: ASCII alnum + underscore + major script blocks.
bool codepoint_is_unicode_word_neighbor_for_digits(char32_t cp);

std::optional<char32_t> utf8_codepoint_before_index(const std::string& s, size_t byte_idx);
std::optional<char32_t> utf8_codepoint_at_index(const std::string& s, size_t byte_idx);

/// True if the ASCII digit substring ``text[start_byte:end_byte]`` should expand like Python ``\\b\\d+\\b``.
bool digit_ascii_span_expandable_python_w(const std::string& text, size_t start_byte, size_t end_byte);

/// Trim ASCII whitespace, map ``_`` → ``-``, and ASCII-lowercase ``A``–``Z``. Used for MoonshineG2P /
/// ``create_rule_based_g2p`` dialect matching (tags are ASCII).
std::string normalize_rule_based_dialect_cli_key(std::string_view raw);

/// Keep the first surface form for each ``normalize_rule_based_dialect_cli_key`` value (drops e.g.
/// both ``pt-PT`` and ``pt_PT`` when they normalize to the same key).
std::vector<std::string> dedupe_dialect_ids_preserve_first(std::vector<std::string> ids);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_UTF8_UTILS_H
