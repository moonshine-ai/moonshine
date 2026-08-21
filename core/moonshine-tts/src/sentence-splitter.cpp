#include "sentence-splitter.h"

#include <utf8proc.h>

#include <array>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "utf8-utils.h"

namespace moonshine_tts {

namespace {

/// Leading two ASCII letters, lowercased. "en_US" and "en-GB" both give "en".
std::string lang_key(std::string_view language) {
  std::string key;
  for (const char c : language) {
    if (c >= 'A' && c <= 'Z') {
      key.push_back(static_cast<char>(c - 'A' + 'a'));
    } else if (c >= 'a' && c <= 'z') {
      key.push_back(c);
    } else {
      break;
    }
    if (key.size() == 2) {
      break;
    }
  }
  return key;
}

/// Titles and Latin-script abbreviations that recur across languages. Stored
/// without the trailing period and ASCII-lowercased; the lookup lowercases too.
constexpr std::array<std::string_view, 40> kCommonAbbreviations = {
    "dr",   "mr",   "mrs", "ms",  "mx",  "prof", "st",   "jr",
    "sr",   "vs",   "etc", "al",  "cf",  "inc",  "ltd",  "co",
    "corp", "dept", "est", "fig", "no",  "op",   "p",    "pp",
    "vol",  "ed",   "eds", "rev", "hon", "gov",  "sen",  "rep",
    "sgt",  "capt", "lt",  "col", "gen", "ave",  "blvd", "approx"};

/// Extra abbreviations per language. The common list above still applies.
const std::unordered_set<std::string>& abbreviations_for(
    std::string_view language) {
  static const std::unordered_set<std::string> kNone;
  static const std::unordered_set<std::string> kGerman = {
      "z", "b", "bzw", "ca",  "ggf", "evtl", "usw", "bspw",
      "d", "h", "nr",  "abb", "hr",  "geb",  "jh",  "vgl"};
  static const std::unordered_set<std::string> kFrench = {
      "mme", "mlle", "mm", "env", "chap", "av", "apr", "ex", "cf"};
  static const std::unordered_set<std::string> kSpanish = {
      "sra", "srta", "ud", "uds", "dra", "ee", "uu", "pág", "núm"};
  static const std::unordered_set<std::string> kItalian = {
      "sig", "sig.ra", "dott", "ing", "avv", "ecc", "pag"};
  static const std::unordered_set<std::string> kPortuguese = {
      "sra", "dra", "exmo", "exma", "pág", "séc"};
  static const std::unordered_set<std::string> kDutch = {
      "dhr", "mevr", "bijv", "enz", "blz", "nl", "o", "a"};
  static const std::unordered_set<std::string> kRussian = {
      "г", "гг", "др", "т", "п", "д", "тыс", "млн", "млрд", "рис", "стр"};

  const std::string key = lang_key(language);
  if (key == "de") {
    return kGerman;
  }
  if (key == "fr") {
    return kFrench;
  }
  if (key == "es") {
    return kSpanish;
  }
  if (key == "it") {
    return kItalian;
  }
  if (key == "pt") {
    return kPortuguese;
  }
  if (key == "nl") {
    return kDutch;
  }
  if (key == "ru" || key == "uk") {
    return kRussian;
  }
  return kNone;
}

bool is_common_abbreviation(std::string_view lowered) {
  for (const std::string_view candidate : kCommonAbbreviations) {
    if (candidate == lowered) {
      return true;
    }
  }
  return false;
}

bool codepoint_is_letter(char32_t cp) {
  const utf8proc_category_t cat =
      utf8proc_category(static_cast<utf8proc_int32_t>(cp));
  return cat == UTF8PROC_CATEGORY_LU || cat == UTF8PROC_CATEGORY_LL ||
         cat == UTF8PROC_CATEGORY_LT || cat == UTF8PROC_CATEGORY_LM ||
         cat == UTF8PROC_CATEGORY_LO;
}

bool codepoint_is_lowercase_letter(char32_t cp) {
  return utf8proc_category(static_cast<utf8proc_int32_t>(cp)) ==
         UTF8PROC_CATEGORY_LL;
}

/// Closing punctuation that belongs to the unit the terminator just ended:
/// `He said "Stop!" Then left.` keeps the quote with the first unit.
bool codepoint_is_closer(char32_t cp) {
  switch (cp) {
    case U')':
    case U']':
    case U'}':
    case U'"':
    case U'\'':
    case 0x00BB:  // »
    case 0x2019:  // ’
    case 0x201D:  // ”
    case 0x2018:  // ‘ (some locales close with this)
    case 0x300D:  // 」
    case 0x300F:  // 』
    case 0xFF09:  // ）
    case 0xFF3D:  // ］
      return true;
    default:
      return false;
  }
}

int bracket_delta(char32_t cp) {
  switch (cp) {
    case U'(':
    case U'[':
    case U'{':
    case 0xFF08:  // （
    case 0xFF3B:  // ［
      return 1;
    case U')':
    case U']':
    case U'}':
    case 0xFF09:
    case 0xFF3D:
      return -1;
    default:
      return 0;
  }
}

/// Paired quotes we can track without ambiguity. Straight `"` is deliberately
/// left out: an unbalanced one (inches, a typo) would suppress every later
/// boundary, which is worse than splitting inside a quoted passage.
int quote_delta(char32_t cp) {
  switch (cp) {
    case 0x201C:  // “
    case 0x00AB:  // «
    case 0x300C:  // 「
    case 0x300E:  // 『
      return 1;
    case 0x201D:  // ”
    case 0x00BB:  // »
    case 0x300D:  // 」
    case 0x300F:  // 』
      return -1;
    default:
      return 0;
  }
}

/// Terminators that need trailing whitespace (or a closer plus whitespace) to
/// count, because they are also used inside numbers and abbreviations.
bool is_ambiguous_terminator(char32_t cp, const SentenceSplitOptions& options) {
  if (cp == U'.' || cp == U'!' || cp == U'?' || cp == 0x2026) {
    return true;
  }
  if (cp == U':' && options.split_on_colon) {
    return true;
  }
  // Greek uses the ASCII semicolon (and U+037E) as its question mark. Enabling
  // it everywhere would break English clauses, so it is language-gated.
  if ((cp == U';' || cp == 0x037E) && lang_key(options.language) == "el") {
    return true;
  }
  return false;
}

/// Terminators that are unambiguous, so they close a unit even at the very end
/// of the buffer and without following whitespace. Scripts that use these
/// mostly do not put spaces after them.
bool is_unambiguous_terminator(char32_t cp) {
  switch (cp) {
    case 0x3002:  // 。
    case 0xFF01:  // ！
    case 0xFF1F:  // ？
    case 0xFF61:  // ｡
    case 0x061F:  // ؟ Arabic question mark
    case 0x06D4:  // ۔ Urdu full stop
    case 0x0964:  // । Devanagari danda
    case 0x0965:  // ॥ Devanagari double danda
    case 0x1362:  // ። Ethiopic full stop
    case 0x0589:  // ։ Armenian full stop
      return true;
    default:
      return false;
  }
}

/// The run of letters immediately before *dot_byte*, ASCII-lowercased, plus how
/// many code points it contained. Used for both the abbreviation lookup and the
/// single-initial rule ("J. R. R. Tolkien").
void preceding_word(const std::string& text, size_t dot_byte,
                    std::string& out_lowered, int& out_codepoints) {
  out_lowered.clear();
  out_codepoints = 0;
  // Walk back over UTF-8 continuation bytes to find code point starts.
  std::vector<size_t> starts;
  size_t i = 0;
  while (i < dot_byte) {
    char32_t cp = 0;
    size_t len = 0;
    if (!utf8_decode_at(text, i, cp, len)) {
      break;
    }
    starts.push_back(i);
    i += len;
  }
  size_t first = dot_byte;
  for (size_t idx = starts.size(); idx > 0; --idx) {
    const size_t start = starts[idx - 1];
    char32_t cp = 0;
    size_t len = 0;
    if (!utf8_decode_at(text, start, cp, len)) {
      break;
    }
    if (!codepoint_is_letter(cp)) {
      break;
    }
    first = start;
    ++out_codepoints;
  }
  out_lowered = text.substr(first, dot_byte - first);
  for (char& c : out_lowered) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
}

/// Leading whitespace only. The remainder has to keep its trailing space: an
/// incremental caller pushing "Hello " then "there." would otherwise get
/// "Hellothere.".
std::string_view ltrim_ascii_ws(std::string_view s) {
  size_t a = 0;
  while (a < s.size() &&
         is_ascii_whitespace(static_cast<unsigned char>(s[a]))) {
    ++a;
  }
  return s.substr(a);
}

int count_codepoints(std::string_view s) {
  const std::string owned(s);
  int n = 0;
  size_t i = 0;
  while (i < owned.size()) {
    char32_t cp = 0;
    size_t len = 0;
    if (!utf8_decode_at(owned, i, cp, len)) {
      break;
    }
    i += len;
    ++n;
  }
  return n;
}

}  // namespace

SentenceSplit split_sentences_incremental(std::string_view text,
                                          const SentenceSplitOptions& options) {
  SentenceSplit result;
  const std::string s(text);
  const size_t n = s.size();
  const std::unordered_set<std::string>& extra_abbrevs =
      abbreviations_for(options.language);

  size_t unit_start = 0;
  size_t i = 0;
  int bracket_depth = 0;
  int quote_depth = 0;
  std::string pending;  // unit held back because it was too short

  auto emit = [&](const std::string& piece) {
    if (piece.empty()) {
      return;
    }
    std::string joined = pending.empty() ? piece : pending + " " + piece;
    pending.clear();
    if (options.min_codepoints > 0 &&
        count_codepoints(joined) < options.min_codepoints) {
      pending = std::move(joined);
      return;
    }
    result.units.push_back(std::move(joined));
  };

  while (i < n) {
    char32_t cp = 0;
    size_t len = 0;
    if (!utf8_decode_at(s, i, cp, len)) {
      break;
    }
    bracket_depth += bracket_delta(cp);
    if (bracket_depth < 0) {
      bracket_depth = 0;
    }
    quote_depth += quote_delta(cp);
    if (quote_depth < 0) {
      quote_depth = 0;
    }

    const bool ambiguous = is_ambiguous_terminator(cp, options);
    const bool unambiguous = !ambiguous && is_unambiguous_terminator(cp);
    if (!ambiguous && !unambiguous) {
      i += len;
      continue;
    }

    // Absorb any closing punctuation so it stays with this unit. Their depth
    // deltas are applied speculatively: `「やめて。」` is a complete unit, so
    // the quote has to count as closed before we judge the boundary.
    size_t unit_end = i + len;
    int closed_brackets = 0;
    int closed_quotes = 0;
    while (unit_end < n) {
      char32_t closer = 0;
      size_t closer_len = 0;
      if (!utf8_decode_at(s, unit_end, closer, closer_len)) {
        break;
      }
      if (!codepoint_is_closer(closer)) {
        break;
      }
      closed_brackets += bracket_delta(closer);
      closed_quotes += quote_delta(closer);
      unit_end += closer_len;
    }

    size_t next_start = unit_end;
    while (next_start < n &&
           is_ascii_whitespace(static_cast<unsigned char>(s[next_start]))) {
      ++next_start;
    }
    const bool has_trailing_space = next_start > unit_end;

    if (ambiguous) {
      // No whitespace yet means either mid-token ("3.14") or a buffer that may
      // grow. Either way, not a boundary.
      if (!has_trailing_space) {
        i += len;
        continue;
      }
      if (cp == U'.') {
        std::string word;
        int word_cps = 0;
        preceding_word(s, i, word, word_cps);
        // Single initials ("J. R. R.") and known abbreviations ("Dr.") are not
        // sentence ends. This also covers "e.g." and "z.B.", whose last letter
        // run is a single character.
        if (word_cps == 1 || is_common_abbreviation(word) ||
            extra_abbrevs.count(word) > 0) {
          i += len;
          continue;
        }
        // A following lowercase word almost always means we misread an
        // abbreviation we do not have on file. Prefer a long unit over a wrong
        // break: a missed split only costs latency.
        if (next_start < n) {
          char32_t next_cp = 0;
          size_t next_len = 0;
          if (utf8_decode_at(s, next_start, next_cp, next_len) &&
              codepoint_is_lowercase_letter(next_cp)) {
            i += len;
            continue;
          }
        }
      }
    }

    const int effective_brackets = bracket_depth + closed_brackets;
    const int effective_quotes = quote_depth + closed_quotes;
    if (effective_brackets > 0 || effective_quotes > 0) {
      i += len;
      continue;
    }

    emit(trim_ascii_ws_copy(
        std::string_view(s).substr(unit_start, unit_end - unit_start)));
    bracket_depth = effective_brackets;
    quote_depth = effective_quotes;
    unit_start = next_start;
    i = next_start;
  }

  std::string tail =
      std::string(ltrim_ascii_ws(std::string_view(s).substr(unit_start)));
  if (!pending.empty()) {
    tail = tail.empty() ? pending : pending + " " + tail;
  }
  result.remainder = std::move(tail);
  return result;
}

std::vector<std::string> split_sentences(std::string_view text,
                                         const SentenceSplitOptions& options) {
  SentenceSplit split = split_sentences_incremental(text, options);
  std::string tail = trim_ascii_ws_copy(split.remainder);
  if (!tail.empty()) {
    split.units.push_back(std::move(tail));
  }
  return std::move(split.units);
}

}  // namespace moonshine_tts
