#ifndef SPELLING_FUSION_DATA_H
#define SPELLING_FUSION_DATA_H

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* Compiled-in tables for the spelling matcher. Values are hand-ported
   from the Python sources (alphanumeric_listener.py); see the
   ``Vocabulary tables`` section of that file for the canonical comments
   on which spoken forms map to which character. The C++ tables are kept
   in a separate translation unit so spelling-fusion.cpp stays small and
   we can swap defaults by editing this file alone. */

namespace spelling_fusion_data {

// Combined letter + digit + special-char lookup. Keys are normalized
// (lower-cased, with punctuation/quotes stripped, internal whitespace
// collapsed). Values are 1-character ASCII strings.
const std::unordered_map<std::string, std::string> &lookup_table();

// Set of normalized "make next letter upper" modifier phrases. The
// ``_by_len`` view sorts them by descending length so the matcher can
// strip the longest matching prefix first ("upper case" wins over
// "upper").
const std::unordered_set<std::string> &upper_modifiers();
const std::vector<std::string> &upper_modifiers_by_length();

// Normalized command-word vocabularies. These are matched verbatim
// against ``spelling_normalize(raw_text)``.
const std::unordered_set<std::string> &undo_words();
const std::unordered_set<std::string> &clear_words();
const std::unordered_set<std::string> &stop_words();

// Default weak-homonym phrases (``okay`` / ``ok`` / ``you``). These are
// resolved by the matcher to single characters but fire so often as
// non-spelling fillers that the fuser demotes them whenever a confident
// spelling-CNN prediction is available.
const std::unordered_set<std::string> &default_weak_homonyms();

// Spelling-CNN class label canonicalization: the model emits digit
// classes as spoken words ("zero".."nine") but we want canonical digit
// characters ("0".."9") in the fused output. Letters pass through
// unchanged (they're already a-z in the model's class list).
const std::unordered_map<std::string, std::string> &spell_class_to_char();

// Default spelling-CNN metadata (sample rate, clip length, class list,
// input/output names) baked in at compile time. Loading a custom
// ``.ort`` model file at runtime will use the embedded
// ``custom_metadata_map`` to override these per-key, but a caller that
// uses the bundled model never needs the sidecar JSON file.
struct DefaultSpellingMeta {
  int32_t sample_rate;
  float clip_seconds;
  const char *input_name;
  const char *output_name;
  const std::vector<std::string> &classes;
};
const DefaultSpellingMeta &default_meta();

}  // namespace spelling_fusion_data

#endif  // SPELLING_FUSION_DATA_H
