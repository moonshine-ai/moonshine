#include "utf8-utils.h"

#include <unordered_set>

namespace moonshine_tts {

bool utf8_decode_at(const std::string& s, size_t i, char32_t& out_cp, size_t& out_len) {
  const size_t n = s.size();
  if (i >= n) {
    return false;
  }
  const unsigned char c0 = static_cast<unsigned char>(s[i]);
  if (c0 < 0x80) {
    out_cp = c0;
    out_len = 1;
    return true;
  }
  if ((c0 >> 5) == 0x6 && i + 1 < n) {
    const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
    if ((c1 >> 6) != 0x2) {
      out_cp = c0;
      out_len = 1;
      return true;
    }
    out_cp = (static_cast<char32_t>(c0 & 0x1Fu) << 6) | (c1 & 0x3Fu);
    out_len = 2;
    return true;
  }
  if ((c0 >> 4) == 0xE && i + 2 < n) {
    const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
    const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
    if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2) {
      out_cp = c0;
      out_len = 1;
      return true;
    }
    out_cp = (static_cast<char32_t>(c0 & 0x0Fu) << 12) | ((c1 & 0x3Fu) << 6) | (c2 & 0x3Fu);
    out_len = 3;
    return true;
  }
  if ((c0 >> 3) == 0x1E && i + 3 < n) {
    const unsigned char c1 = static_cast<unsigned char>(s[i + 1]);
    const unsigned char c2 = static_cast<unsigned char>(s[i + 2]);
    const unsigned char c3 = static_cast<unsigned char>(s[i + 3]);
    if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2 || (c3 >> 6) != 0x2) {
      out_cp = c0;
      out_len = 1;
      return true;
    }
    out_cp = (static_cast<char32_t>(c0 & 0x07u) << 18) | ((c1 & 0x3Fu) << 12) |
             ((c2 & 0x3Fu) << 6) | (c3 & 0x3Fu);
    out_len = 4;
    return true;
  }
  out_cp = c0;
  out_len = 1;
  return true;
}

std::u32string utf8_str_to_u32(const std::string& s) {
  std::u32string out;
  size_t i = 0;
  while (i < s.size()) {
    char32_t cp = 0;
    size_t adv = 0;
    utf8_decode_at(s, i, cp, adv);
    out.push_back(cp);
    i += adv;
  }
  return out;
}

void utf8_append_codepoint(std::string& out, char32_t cp) {
  if (cp <= 0x7F) {
    out.push_back(static_cast<char>(cp));
  } else if (cp <= 0x7FF) {
    out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0xFFFF) {
    out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp <= 0x10FFFF) {
    out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

std::vector<std::string> utf8_split_codepoints(const std::string& utf8) {
  std::vector<std::string> out;
  const size_t n = utf8.size();
  size_t i = 0;
  while (i < n) {
    const unsigned char c0 = static_cast<unsigned char>(utf8[i]);
    char32_t cp = 0;
    size_t adv = 1;
    if (c0 < 0x80) {
      cp = c0;
    } else if ((c0 >> 5) == 0x6 && i + 1 < n) {
      const unsigned char c1 = static_cast<unsigned char>(utf8[i + 1]);
      if ((c1 >> 6) != 0x2) {
        out.emplace_back(1, utf8[i++]);
        continue;
      }
      cp = ((c0 & 0x1F) << 6) | (c1 & 0x3F);
      adv = 2;
    } else if ((c0 >> 4) == 0xE && i + 2 < n) {
      const unsigned char c1 = static_cast<unsigned char>(utf8[i + 1]);
      const unsigned char c2 = static_cast<unsigned char>(utf8[i + 2]);
      if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2) {
        out.emplace_back(1, utf8[i++]);
        continue;
      }
      cp = ((c0 & 0x0F) << 12) | ((c1 & 0x3F) << 6) | (c2 & 0x3F);
      adv = 3;
    } else if ((c0 >> 3) == 0x1E && i + 3 < n) {
      const unsigned char c1 = static_cast<unsigned char>(utf8[i + 1]);
      const unsigned char c2 = static_cast<unsigned char>(utf8[i + 2]);
      const unsigned char c3 = static_cast<unsigned char>(utf8[i + 3]);
      if ((c1 >> 6) != 0x2 || (c2 >> 6) != 0x2 || (c3 >> 6) != 0x2) {
        out.emplace_back(1, utf8[i++]);
        continue;
      }
      cp = ((c0 & 0x07) << 18) | ((c1 & 0x3F) << 12) | ((c2 & 0x3F) << 6) | (c3 & 0x3F);
      adv = 4;
    } else {
      out.emplace_back(1, utf8[i++]);
      continue;
    }
    std::string one;
    utf8_append_codepoint(one, cp);
    out.push_back(std::move(one));
    i += adv;
  }
  return out;
}

std::optional<std::pair<int, int>> utf8_find_token_codepoints(const std::string& text,
                                                              const std::string& token,
                                                              int start_cp) {
  const auto hay = utf8_split_codepoints(text);
  const auto nd = utf8_split_codepoints(token);
  if (nd.empty()) {
    return std::nullopt;
  }
  const int hn = static_cast<int>(hay.size());
  const int tn = static_cast<int>(nd.size());
  if (start_cp < 0) {
    start_cp = 0;
  }
  for (int i = start_cp; i + tn <= hn; ++i) {
    bool ok = true;
    for (int j = 0; j < tn; ++j) {
      if (hay[static_cast<size_t>(i + j)] != nd[static_cast<size_t>(j)]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return std::make_pair(i, i + tn);
    }
  }
  // Fallback: first occurrence (Python text.find(token))
  for (int i = 0; i + tn <= hn; ++i) {
    bool ok = true;
    for (int j = 0; j < tn; ++j) {
      if (hay[static_cast<size_t>(i + j)] != nd[static_cast<size_t>(j)]) {
        ok = false;
        break;
      }
    }
    if (ok) {
      return std::make_pair(i, i + tn);
    }
  }
  return std::nullopt;
}

bool codepoint_is_unicode_word_neighbor_for_digits(char32_t c) {
  if (c == U'_' || (c >= U'0' && c <= U'9')) {
    return true;
  }
  if ((c >= U'a' && c <= U'z') || (c >= U'A' && c <= U'Z')) {
    return true;
  }
  if (c < 0x80u) {
    return false;
  }
  if (c >= U'\u00AA' && c <= U'\u00FF' && c != U'\u00D7' && c != U'\u00F7') {
    return true;
  }
  if (c >= U'\u0100' && c <= U'\u024F') {
    return true;
  }
  if (c >= U'\u0400' && c <= U'\u04FF') {
    return true;
  }
  if (c >= U'\u0500' && c <= U'\u052F') {
    return true;
  }
  if (c >= U'\u2DE0' && c <= U'\u2DFF') {
    return true;
  }
  if (c >= U'\u3040' && c <= U'\u30FF') {
    return true;
  }
  if (c >= U'\u3400' && c <= U'\u9FFF') {
    return true;
  }
  if (c >= U'\uAC00' && c <= U'\uD7AF') {
    return true;
  }
  return false;
}

std::optional<char32_t> utf8_codepoint_before_index(const std::string& s, size_t byte_idx) {
  if (byte_idx == 0 || byte_idx > s.size()) {
    return std::nullopt;
  }
  size_t i = byte_idx;
  do {
    --i;
  } while (i > 0 && (static_cast<unsigned char>(s[i]) & 0xC0) == 0x80);
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(s, i, cp, adv)) {
    return std::nullopt;
  }
  return cp;
}

std::optional<char32_t> utf8_codepoint_at_index(const std::string& s, size_t byte_idx) {
  if (byte_idx >= s.size()) {
    return std::nullopt;
  }
  char32_t cp = 0;
  size_t adv = 0;
  if (!utf8_decode_at(s, byte_idx, cp, adv)) {
    return std::nullopt;
  }
  return cp;
}

bool digit_ascii_span_expandable_python_w(const std::string& text, size_t start_byte, size_t end_byte) {
  const std::optional<char32_t> prev = utf8_codepoint_before_index(text, start_byte);
  if (prev.has_value() && codepoint_is_unicode_word_neighbor_for_digits(*prev)) {
    return false;
  }
  const std::optional<char32_t> next = utf8_codepoint_at_index(text, end_byte);
  if (next.has_value() && codepoint_is_unicode_word_neighbor_for_digits(*next)) {
    return false;
  }
  return true;
}

std::string normalize_rule_based_dialect_cli_key(std::string_view raw) {
  std::string s = trim_ascii_ws_copy(raw);
  for (char& c : s) {
    if (c == '_') {
      c = '-';
    } else if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

std::vector<std::string> dedupe_dialect_ids_preserve_first(std::vector<std::string> ids) {
  std::unordered_set<std::string> seen;
  std::vector<std::string> out;
  out.reserve(ids.size());
  for (std::string& id : ids) {
    const std::string n = normalize_rule_based_dialect_cli_key(id);
    if (n.empty()) {
      continue;
    }
    if (seen.insert(n).second) {
      out.push_back(std::move(id));
    }
  }
  return out;
}

}  // namespace moonshine_tts
