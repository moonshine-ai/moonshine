#include "text-normalize.h"

#include "utf8-utils.h"

#include <cctype>
namespace moonshine_tts {

namespace {

bool is_word_char_utf8(std::string_view unit) {
  if (unit.empty()) {
    return false;
  }
  if (unit.size() == 1) {
    const unsigned char u = static_cast<unsigned char>(unit[0]);
    return std::isalnum(u) != 0;
  }
  // Non-ASCII UTF-8 sequence: treat as word character (accents, IPA in keys, etc.).
  return true;
}

}  // namespace

std::vector<std::string> split_text_to_words(std::string_view text) {
  std::vector<std::string> out;
  std::string current;
  bool in_token = false;
  for (unsigned char uc : text) {
    const char ch = static_cast<char>(uc);
    if (std::isspace(static_cast<unsigned char>(ch)) != 0) {
      if (in_token) {
        out.push_back(std::move(current));
        current.clear();
        in_token = false;
      }
    } else {
      current.push_back(ch);
      in_token = true;
    }
  }
  if (in_token) {
    out.push_back(std::move(current));
  }
  return out;
}

std::string normalize_word_for_lookup(std::string_view token) {
  std::string s(token.begin(), token.end());
  // tolower ASCII range; leave UTF-8 multibyte sequences unchanged except via copy
  for (char& ch : s) {
    if (static_cast<unsigned char>(ch) < 128) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
  }
  const auto units = utf8_split_codepoints(s);
  size_t i = 0, j = units.size();
  while (i < j && !is_word_char_utf8(units[i])) {
    ++i;
  }
  while (i < j && !is_word_char_utf8(units[j - 1])) {
    --j;
  }
  if (i >= j) {
    return "";
  }
  std::string out;
  for (size_t k = i; k < j; ++k) {
    out += units[k];
  }
  return out;
}

std::string normalize_grapheme_key(std::string_view word_token) {
  std::string s(word_token.begin(), word_token.end());
  for (char& ch : s) {
    if (static_cast<unsigned char>(ch) < 128) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
  }
  if (!s.empty() && s.back() == ')') {
    const auto open = s.rfind('(');
    if (open != std::string::npos && open + 1 < s.size() - 1) {
      bool all_digit = true;
      for (size_t k = open + 1; k < s.size() - 1; ++k) {
        if (!std::isdigit(static_cast<unsigned char>(s[k]))) {
          all_digit = false;
          break;
        }
      }
      if (all_digit) {
        s.resize(open);
      }
    }
  }
  return s;
}

}  // namespace moonshine_tts
