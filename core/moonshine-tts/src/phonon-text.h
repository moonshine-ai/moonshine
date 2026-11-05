#ifndef MOONSHINE_TTS_PHONON_TEXT_H
#define MOONSHINE_TTS_PHONON_TEXT_H

#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

/// Text preparation for Phonon, a port of the ``preprocess`` and ``plan``
/// modules of Gradium's reference runtime, ``xn-ptts``
/// (https://github.com/gradium-ai/xn-ptts, MIT or Apache 2.0). The model reads
/// raw English text, so what happens here is folding typography it never saw
/// into the characters it was trained on, reading out numbers it would
/// otherwise stumble over, and shaping each prompt the way training prompts
/// were shaped. Keeping this in step with the runtime is what keeps the
/// rendering the same as Gradium's.

/// Fold ``text`` into Phonon's character set. Curly quotes become ASCII
/// quotes; dashes, brackets, bullets, arrows, emoji and control characters
/// become spaces; ``…`` becomes ``.``; ``;``, parentheses and a ``:`` with
/// space around it become commas, which is how the model is asked to pause
/// (a ``:`` inside ``10:30`` is kept); ``@``, ``+`` and ``=`` are read out.
/// Numbers of a thousand or more are read with their scale words
/// (``1,234.5`` -> ``1 thousand 234 point 5``) because the model only learnt to
/// read values under a thousand and years as digits; a leading zero marks a
/// code and is spelled digit by digit.
std::string phonon_normalize_text(std::string_view text);

/// What one prompt looks like after ``phonon_prepare_prompt``.
struct PhononPrompt {
  /// Trimmed, whitespace collapsed, first letter upper-cased and ending in a
  /// full stop when it ended in a letter or digit.
  std::string text{};
  /// Whitespace-separated words, which decides how many frames are generated
  /// after end of speech (more for very short prompts).
  int words = 0;
};

PhononPrompt phonon_prepare_prompt(std::string_view text);

/// Group the sentences of already-normalized ``text`` into prompts.
///
/// Whole sentences are packed greedily into chunks of at most
/// ``max_tokens`` tokens; a single sentence longer than that stays one chunk
/// unless it exceeds ``hard_max_tokens``, in which case it is cut at the
/// sentence end, clause mark or word nearest its middle until it fits. The
/// model stops near the length of speech it learnt to say in one go and skips
/// words to get there, so a prompt far over ``hard_max_tokens`` would lose
/// text rather than just sound worse. ``count_tokens`` is the tokenizer's view
/// of a piece of prepared text.
std::vector<std::string> phonon_plan_chunks(
    std::string_view text,
    const std::function<int(std::string_view)>& count_tokens, int max_tokens,
    int hard_max_tokens);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_PHONON_TEXT_H
