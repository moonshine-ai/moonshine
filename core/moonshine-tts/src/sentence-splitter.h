#ifndef MOONSHINE_TTS_SENTENCE_SPLITTER_H
#define MOONSHINE_TTS_SENTENCE_SPLITTER_H

#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

struct SentenceSplitOptions {
  /// BCP-47-ish tag ("en", "en_US", "de-DE"). Only the leading two ASCII
  /// letters are used, to pick the abbreviation list and a few per-language
  /// terminators (Greek treats ";" as a question mark).
  std::string language{};

  /// Treat ":" as an utterance boundary. On by default because `say()` has
  /// always broken there, which lets "Warning: ..." start speaking sooner.
  bool split_on_colon = true;

  /// Units shorter than this many code points are merged into the following
  /// unit rather than spoken alone. Zero disables the merge.
  int min_codepoints = 0;
};

struct SentenceSplit {
  /// Units that are certainly complete: safe to synthesize now.
  std::vector<std::string> units;
  /// Trailing text whose terminator has not been confirmed yet. Empty unless
  /// the input ends mid-unit.
  std::string remainder;
};

/// Split *text* into complete units plus a trailing remainder.
///
/// Written for incremental input: an ASCII terminator at the very end of the
/// buffer stays in `remainder`, because the next push could turn "3." into
/// "3.14". Terminators that cannot be ambiguous this way (CJK, Devanagari,
/// Arabic) close their unit immediately. Callers that have no more text should
/// use `split_sentences`, or speak `remainder` themselves.
SentenceSplit split_sentences_incremental(std::string_view text,
                                          const SentenceSplitOptions& options);

/// `split_sentences_incremental` with the remainder appended as a final unit.
/// This is the whole-string entry point used by `say()`.
std::vector<std::string> split_sentences(std::string_view text,
                                         const SentenceSplitOptions& options);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_SENTENCE_SPLITTER_H
