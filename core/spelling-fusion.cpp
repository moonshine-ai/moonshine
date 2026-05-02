#include "spelling-fusion.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>

#include "spelling-fusion-data.h"

namespace {

constexpr float kDisagreeThreshold = 0.5f;
constexpr float kWeakHomonymOverrideThreshold = 0.3f;

// Bytes that ``spelling_normalize`` strips before lookup. Mirrors the
// Python ``_NORMALIZE_DROP_CHARS`` set: sentence-ending punctuation and
// straight/curly quotes. Curly quotes are matched as their UTF-8 byte
// sequences below since this function operates on raw bytes.
constexpr std::array<char, 4> kAsciiDropChars = {'.', ',', '!', '?'};
constexpr std::array<char, 2> kAsciiQuoteChars = {'"', '\''};

bool is_ascii_drop(char c) {
  for (char d : kAsciiDropChars) {
    if (c == d) return true;
  }
  for (char d : kAsciiQuoteChars) {
    if (c == d) return true;
  }
  return false;
}

bool is_ascii_letter(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
}

char ascii_to_lower(char c) {
  if (c >= 'A' && c <= 'Z') return c + ('a' - 'A');
  return c;
}

// Try to consume a 3-byte UTF-8 curly quote starting at ``input[i]``.
// Returns the number of bytes consumed (3) on success, 0 otherwise.
size_t consume_curly_quote(const std::string &input, size_t i) {
  if (i + 2 >= input.size()) return 0;
  uint8_t a = static_cast<uint8_t>(input[i]);
  uint8_t b = static_cast<uint8_t>(input[i + 1]);
  uint8_t c = static_cast<uint8_t>(input[i + 2]);
  // U+2018, U+2019 → 0xE2 0x80 0x98 / 0x99   (curly single quotes)
  // U+201C, U+201D → 0xE2 0x80 0x9C / 0x9D   (curly double quotes)
  if (a == 0xE2 && b == 0x80) {
    if (c == 0x98 || c == 0x99 || c == 0x9C || c == 0x9D) {
      return 3;
    }
  }
  return 0;
}

// Split ``s`` on ASCII whitespace, dropping empty tokens.
std::vector<std::string> split_on_whitespace(const std::string &s) {
  std::vector<std::string> tokens;
  std::string current;
  for (char c : s) {
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
        c == '\v') {
      if (!current.empty()) {
        tokens.push_back(std::move(current));
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    tokens.push_back(std::move(current));
  }
  return tokens;
}

// Basic English number-word parser, range 10-1000. Mirrors the Python
// helper: anything not a recognizable number phrase (or a single digit)
// returns std::nullopt so the caller can fall through to other lookups.
const std::unordered_map<std::string, int> &ones_table() {
  static const std::unordered_map<std::string, int> t = {
      {"one", 1}, {"two", 2}, {"three", 3}, {"four", 4}, {"five", 5},
      {"six", 6}, {"seven", 7}, {"eight", 8}, {"nine", 9},
  };
  return t;
}
const std::unordered_map<std::string, int> &teens_table() {
  static const std::unordered_map<std::string, int> t = {
      {"ten", 10}, {"eleven", 11}, {"twelve", 12}, {"thirteen", 13},
      {"fourteen", 14}, {"fifteen", 15}, {"sixteen", 16}, {"seventeen", 17},
      {"eighteen", 18}, {"nineteen", 19},
  };
  return t;
}
const std::unordered_map<std::string, int> &tens_table() {
  static const std::unordered_map<std::string, int> t = {
      {"twenty", 20}, {"thirty", 30}, {"forty", 40}, {"fifty", 50},
      {"sixty", 60}, {"seventy", 70}, {"eighty", 80}, {"ninety", 90},
  };
  return t;
}

std::optional<int> parse_number_words(const std::string &text) {
  std::string s;
  s.reserve(text.size());
  for (char c : text) {
    s.push_back(c == '-' ? ' ' : c);
  }
  std::vector<std::string> raw_tokens = split_on_whitespace(s);
  std::vector<std::string> words;
  words.reserve(raw_tokens.size());
  for (auto &tok : raw_tokens) {
    if (tok != "and") words.push_back(std::move(tok));
  }
  if (words.empty()) return std::nullopt;

  if (words[0] == "a") words[0] = "one";

  int result = 0;
  size_t i = 0;
  const auto &ones = ones_table();

  // Optional hundreds: "<n> hundred"
  if (i < words.size() && ones.count(words[i]) && i + 1 < words.size() &&
      words[i + 1] == "hundred") {
    result += ones.at(words[i]) * 100;
    i += 2;
  }

  // Bare "hundred" = 100 (only valid as the leading token).
  if (i == 0 && words.size() >= 1 && words[0] == "hundred") {
    result += 100;
    i += 1;
  }

  // "<n> thousand" — only n=1 supported (range tops out at 1000).
  if (i < words.size() && ones.count(words[i]) && i + 1 < words.size() &&
      words[i + 1] == "thousand") {
    int val = ones.at(words[i]);
    if (val == 1) {
      result += 1000;
      i += 2;
      if (i == words.size()) return result;
    }
    return std::nullopt;
  }
  if (i == 0 && words.size() >= 1 && words[0] == "thousand") {
    result += 1000;
    i += 1;
    if (i == words.size()) return result;
    return std::nullopt;
  }

  const auto &teens = teens_table();
  const auto &tens = tens_table();
  if (i < words.size() && teens.count(words[i])) {
    result += teens.at(words[i]);
    i += 1;
  } else if (i < words.size() && tens.count(words[i])) {
    result += tens.at(words[i]);
    i += 1;
    if (i < words.size() && ones.count(words[i])) {
      result += ones.at(words[i]);
      i += 1;
    }
  } else if (i < words.size() && ones.count(words[i])) {
    result += ones.at(words[i]);
    i += 1;
  }

  if (i != words.size()) return std::nullopt;
  if (result < 10 || result > 1000) return std::nullopt;
  return result;
}

bool is_ascii_digit_string(const std::string &s) {
  if (s.empty()) return false;
  for (char c : s) {
    if (c < '0' || c > '9') return false;
  }
  return true;
}

bool is_printable_ascii(char c) {
  return static_cast<uint8_t>(c) >= 0x20 && static_cast<uint8_t>(c) < 0x7F;
}

}  // namespace

std::string spelling_normalize(const std::string &text) {
  if (text.empty()) return "";

  // First pass: lowercase, drop punctuation / quote bytes (ASCII +
  // multibyte curly quotes), and collapse all internal whitespace runs
  // to a single space.
  std::string result;
  result.reserve(text.size());
  bool in_space = false;
  bool seen_non_space = false;
  for (size_t i = 0; i < text.size();) {
    if (size_t skip = consume_curly_quote(text, i); skip > 0) {
      i += skip;
      continue;
    }
    char c = text[i];
    if (is_ascii_drop(c)) {
      i += 1;
      continue;
    }
    if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
        c == '\v') {
      if (seen_non_space) {
        in_space = true;
      }
      i += 1;
      continue;
    }
    if (in_space) {
      result.push_back(' ');
      in_space = false;
    }
    result.push_back(ascii_to_lower(c));
    seen_non_space = true;
    i += 1;
  }
  return result;
}

SpellingMatcher::SpellingMatcher()
    : lookup_(&spelling_fusion_data::lookup_table()),
      upper_modifiers_(&spelling_fusion_data::upper_modifiers()),
      upper_modifiers_by_len_(
          &spelling_fusion_data::upper_modifiers_by_length()),
      undo_words_(&spelling_fusion_data::undo_words()),
      clear_words_(&spelling_fusion_data::clear_words()),
      stop_words_(&spelling_fusion_data::stop_words()),
      weak_homonyms_(&spelling_fusion_data::default_weak_homonyms()) {}

SpellingMatch SpellingMatcher::classify(const std::string &raw_text) const {
  SpellingMatch out;
  std::string text = spelling_normalize(raw_text);
  if (text.empty()) return out;

  if (stop_words_->count(text)) {
    out.type = SpellingMatchType::STOPPED;
    return out;
  }
  if (clear_words_->count(text)) {
    out.type = SpellingMatchType::CLEAR;
    return out;
  }
  if (undo_words_->count(text)) {
    out.type = SpellingMatchType::UNDO;
    return out;
  }

  // Strip an upper-case modifier prefix when present (longest-first).
  bool make_upper = false;
  for (const std::string &mod : *upper_modifiers_by_len_) {
    std::string prefix = mod + " ";
    if (text.size() > prefix.size() &&
        text.compare(0, prefix.size(), prefix) == 0) {
      text.erase(0, prefix.size());
      // Trim any leftover leading whitespace (defensive — normalize
      // already collapsed runs, but a stray space after stripping is
      // possible if the prefix matched mid-string).
      while (!text.empty() && text.front() == ' ') {
        text.erase(text.begin());
      }
      make_upper = true;
      break;
    }
    if (text == mod) {
      // Bare modifier with no following character — not a hit.
      return out;
    }
  }

  std::optional<std::string> resolved = resolve(text);
  if (!resolved.has_value()) {
    return out;
  }
  std::string ch = std::move(*resolved);
  if (ch.empty()) {
    return out;
  }
  if (make_upper && ch.size() == 1 && is_ascii_letter(ch[0])) {
    ch[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(ch[0])));
  }
  out.type = SpellingMatchType::CHARACTER;
  out.character = std::move(ch);
  return out;
}

bool SpellingMatcher::is_weak_homonym(const std::string &raw_text) const {
  if (weak_homonyms_->empty()) return false;
  return weak_homonyms_->count(spelling_normalize(raw_text)) > 0;
}

std::optional<std::string> SpellingMatcher::resolve(
    const std::string &text) const {
  auto it = lookup_->find(text);
  if (it != lookup_->end()) {
    return it->second;
  }
  if (auto spelled = resolve_spelled_letter(text); spelled.has_value()) {
    return spelled;
  }
  if (std::optional<int> num = parse_number_words(text); num.has_value()) {
    return std::to_string(*num);
  }
  if (is_ascii_digit_string(text)) {
    return text;
  }
  if (text.size() == 1 && is_printable_ascii(text[0])) {
    return text;
  }
  return std::nullopt;
}

std::optional<std::string> SpellingMatcher::resolve_spelled_letter(
    const std::string &text) const {
  // Speller patterns: "A as in Alpha" / "B for Bravo" / "C is for Charlie"
  // etc. The left side must resolve to a single letter through the
  // built-in lookup; the right side must be a single word starting with
  // that letter (case-insensitive).
  static const std::array<const char *, 4> connectors = {
      " as in ", " is for ", " like ", " for ",
  };
  for (const char *connector : connectors) {
    size_t idx = text.find(connector);
    if (idx == std::string::npos || idx == 0) continue;
    size_t connector_len = std::strlen(connector);
    std::string left = text.substr(0, idx);
    std::string right = text.substr(idx + connector_len);
    auto trim = [](std::string &s) {
      while (!s.empty() && s.front() == ' ') s.erase(s.begin());
      while (!s.empty() && s.back() == ' ') s.pop_back();
    };
    trim(left);
    trim(right);
    if (left.empty() || right.empty()) continue;
    auto left_it = lookup_->find(left);
    if (left_it == lookup_->end()) continue;
    const std::string &left_char = left_it->second;
    if (left_char.size() != 1 || !is_ascii_letter(left_char[0])) continue;
    std::vector<std::string> right_words = split_on_whitespace(right);
    if (right_words.size() != 1) continue;
    if (right_words[0].empty()) continue;
    if (ascii_to_lower(right_words[0][0]) != ascii_to_lower(left_char[0])) {
      continue;
    }
    return left_char;
  }
  return std::nullopt;
}

namespace {

// Mirror Python's ``str.isalpha()`` / ``str.isdigit()`` semantics on the
// ASCII strings that the matcher emits. The ASR side can return
// multi-digit number strings (e.g. ``"1944"`` from ``str.isdigit()``
// fallback in the matcher), so single-character predicates would treat
// those as "special" and break the digit-class tiebreak below.
bool string_is_letter(const std::string &c) {
  if (c.empty()) return false;
  for (char ch : c) {
    if (!is_ascii_letter(ch)) return false;
  }
  return true;
}

bool string_is_digit(const std::string &c) {
  if (c.empty()) return false;
  for (char ch : c) {
    if (ch < '0' || ch > '9') return false;
  }
  return true;
}

bool single_char_is_letter(const std::string &c) {
  return c.size() == 1 && is_ascii_letter(c[0]);
}

std::string apply_case(const std::string &ch, const std::string &hint) {
  // Up-case ``ch`` iff ``hint`` was upper-case. Used when the spelling
  // model wins a tiebreak: the audio classifier is case-blind so we
  // lean on the matcher to know whether the user said "capital".
  if (!single_char_is_letter(hint) || !single_char_is_letter(ch)) return ch;
  bool hint_upper = std::isupper(static_cast<unsigned char>(hint[0])) != 0;
  if (!hint_upper) return ch;
  std::string out = ch;
  out[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(out[0])));
  return out;
}

}  // namespace

FusedResult fuse_default(const std::string &raw_text,
                         const SpellingMatch &match,
                         const SpellingPrediction *prediction,
                         const SpellingMatcher &matcher) {
  FusedResult out;

  // Command words are owned by the matcher; the spelling model has no
  // class for them and we don't want a stray prediction to consume an
  // explicit "stop".
  if (match.type == SpellingMatchType::STOPPED ||
      match.type == SpellingMatchType::CLEAR ||
      match.type == SpellingMatchType::UNDO) {
    out.type = match.type;
    return out;
  }

  std::optional<std::string> asr_char;
  if (match.type == SpellingMatchType::CHARACTER) {
    asr_char = match.character;
  }

  // Weak-homonym demotion: phrases like "okay" / "you" only had ~20 %
  // precision against the People's Speech labels, so when a confident
  // audio prediction is available we treat the matcher's hit as a miss
  // and fall through to the spelling-only branch below.
  if (asr_char.has_value() && prediction != nullptr &&
      prediction->probability >= kWeakHomonymOverrideThreshold &&
      matcher.is_weak_homonym(raw_text)) {
    asr_char.reset();
  }

  // Smart-router fusion. Identical to the Python implementation:
  //   * No prediction → use ASR (or NONE if matcher missed).
  //   * No ASR        → use prediction if any.
  //   * Same letter   → ASR (preserves the matcher's casing).
  //   * Cross-class   → digits go to ASR, letters go to spelling.
  //   * Same class    → break ties on the spelling probability.
  if (prediction == nullptr) {
    if (asr_char.has_value()) {
      out.type = SpellingMatchType::CHARACTER;
      out.character = *asr_char;
    }
    return out;
  }
  if (!asr_char.has_value()) {
    out.type = SpellingMatchType::CHARACTER;
    out.character = prediction->character;
    return out;
  }

  std::string asr_lower = *asr_char;
  if (string_is_letter(asr_lower)) {
    for (char &c : asr_lower) c = ascii_to_lower(c);
  }
  std::string spell_lower = prediction->character;
  if (string_is_letter(spell_lower)) {
    for (char &c : spell_lower) c = ascii_to_lower(c);
  }

  if (asr_lower == spell_lower) {
    out.type = SpellingMatchType::CHARACTER;
    out.character = *asr_char;
    return out;
  }

  bool asr_is_digit = string_is_digit(*asr_char);
  bool spell_is_digit = string_is_digit(prediction->character);
  if (asr_is_digit && !spell_is_digit) {
    out.type = SpellingMatchType::CHARACTER;
    out.character = *asr_char;
    return out;
  }
  if (spell_is_digit && !asr_is_digit) {
    out.type = SpellingMatchType::CHARACTER;
    out.character = prediction->character;
    return out;
  }

  // Same class, both confident enough to fire — break the tie on the
  // spelling probability.
  if (prediction->probability >= kDisagreeThreshold) {
    out.type = SpellingMatchType::CHARACTER;
    out.character = apply_case(prediction->character, *asr_char);
    return out;
  }
  out.type = SpellingMatchType::CHARACTER;
  out.character = *asr_char;
  return out;
}
