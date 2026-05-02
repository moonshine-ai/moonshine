#ifndef SPELLING_FUSION_H
#define SPELLING_FUSION_H

#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* C++ port of the matcher / fusion logic from
   ``python/src/moonshine_voice/alphanumeric_listener.py``.

   The matcher classifies a single utterance (typically the text of one
   completed transcription line) into one of:
     - CHARACTER  : a single letter, digit, or special character
     - STOPPED    : the speaker said "stop" / "done" / etc.
     - CLEAR      : "clear" / "reset"
     - UNDO       : "delete" / "backspace"
     - NONE       : nothing matched

   The fuser combines a matcher result with an optional spelling-model
   prediction using the smart-router strategy that is the proven default
   on the moonshine-spelling People's Speech eval. Only this default
   strategy is implemented (no AUTO/SPELLING_ONLY/ASR_ONLY toggle). */

enum class SpellingMatchType {
  NONE,
  CHARACTER,
  STOPPED,
  CLEAR,
  UNDO,
};

struct SpellingMatch {
  SpellingMatchType type = SpellingMatchType::NONE;
  // Only populated for CHARACTER. Single-character UTF-8 string except for
  // multibyte symbols where we never produce one (every entry in the
  // built-in vocabulary is a 1-byte ASCII output).
  std::string character;

  bool is_character() const { return type == SpellingMatchType::CHARACTER; }
  bool is_recognized() const { return type != SpellingMatchType::NONE; }
};

// A top-1 prediction from the spelling-CNN, in canonical form (digits as
// "0".."9", letters as "a".."z").
struct SpellingPrediction {
  std::string character;
  float probability = 0.0f;
  std::string raw_class;  // model's raw label, e.g. "zero" for digit classes.
};

class SpellingMatcher {
 public:
  SpellingMatcher();

  // Classify a single utterance. Returns NONE if the text cannot be
  // resolved to a character or command word.
  SpellingMatch classify(const std::string &raw_text) const;

  // True iff ``raw_text`` normalizes to one of the weak-homonym phrases
  // (``okay`` / ``ok`` / ``you``). These phrases trigger constantly as
  // fillers in real conversational speech, so when a spelling-CNN
  // prediction is available the fuser demotes the matcher's hit to NONE
  // and lets the audio-side model take over.
  bool is_weak_homonym(const std::string &raw_text) const;

 private:
  // Resolved single-character lookup (already-normalized keys).
  const std::unordered_map<std::string, std::string> *lookup_ = nullptr;
  const std::unordered_set<std::string> *upper_modifiers_ = nullptr;
  const std::vector<std::string> *upper_modifiers_by_len_ = nullptr;
  const std::unordered_set<std::string> *undo_words_ = nullptr;
  const std::unordered_set<std::string> *clear_words_ = nullptr;
  const std::unordered_set<std::string> *stop_words_ = nullptr;
  const std::unordered_set<std::string> *weak_homonyms_ = nullptr;

  std::optional<std::string> resolve(const std::string &text) const;
  std::optional<std::string> resolve_spelled_letter(
      const std::string &text) const;
};

// Combined output of (matcher, optional prediction) → final character.
//
// For non-character commands (STOPPED / CLEAR / UNDO) the matcher always
// wins -- the spelling model has no class for them and we don't want a
// stray prediction to consume an explicit "stop".
struct FusedResult {
  SpellingMatchType type = SpellingMatchType::NONE;
  std::string character;
  bool is_character() const { return type == SpellingMatchType::CHARACTER; }
};

// Apply the default smart-router fusion to the given matcher hit and
// (optional) spelling prediction.
//
// raw_text is needed for the weak-homonym demotion check (we re-check
// the original utterance, not just the matcher's resolved character).
//
// Constants used (matching the Python defaults):
//   * disagree_threshold = 0.5
//   * weak_homonym_override_threshold = 0.3
//
// Special characters that the spelling-CNN has no class for (anything
// not in 0-9 / a-z) always pass through from the matcher unchanged --
// this is what protects "dollar sign" → "$" and the rest of the symbol
// vocabulary from being silently overwritten by the audio model.
FusedResult fuse_default(const std::string &raw_text,
                         const SpellingMatch &match,
                         const SpellingPrediction *prediction,
                         const SpellingMatcher &matcher);

// Lower-case ``text`` and strip the same punctuation / quote chars as
// the Python ``_normalize`` helper. Exposed for the unit tests; you
// don't normally need to call this from application code.
std::string spelling_normalize(const std::string &text);

#endif  // SPELLING_FUSION_H
