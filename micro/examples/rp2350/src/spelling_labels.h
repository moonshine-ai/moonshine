// Spoken "sound-alike" word for each recognized class label, so the readback
// TTS says "bee" for 'b' rather than an unvoiceable bare consonant.
//
// Based on the project's letter-name table (LETTER_TO_SPELLED in
// scripts/extract_aligned_letter_clips.py), but using real English words rather
// than that table's forced-aligner transliterations wherever a homophone of the
// letter name exists -- the formant TTS's G2P pronounces real words far more
// reliably. So H is "aitch" (not "aich"), R is "are" (not "ar"), W is
// "double u" (not "dublu"), X is "ex" (not "eks"). S is "eh s": a short vowel
// ("eh" -> /ɛ/) then the letter name ("s" -> /ɛs/ via the baked dictionary),
// which keeps the readback distinct from "eff" (/ɛf/) and bare /s/ hiss.
// A is "hay" (/heɪ/): there is no zero-onset English word for the letter-A
// vowel, and "ay"/"a" come out as "eye"/"uh" through the G2P, so we accept
// "eye"/"uh" through the G2P, so we accept the soft leading breath. E keeps the
// phonetic "ee" (no clean homophone). Digits already arrive as words
// ("zero".."nine") and are spoken as-is.

#ifndef SPELLING_SPELLING_LABELS_H_
#define SPELLING_SPELLING_LABELS_H_

namespace spelling {

// Map a class label to the text the TTS should speak. Letters are single
// characters 'a'..'z'; everything else (the digit words) passes through.
inline const char* SpokenForLabel(const char* label) {
  // Indexed by letter: 'a'->[0] .. 'z'->[25].
  static constexpr const char* kLetterSpoken[26] = {
      "hay", "bee", "see", "dee", "ee",       "eff", "gee", "aitch", "eye",
      "jay", "kay", "el",  "em",  "en",       "oh",  "pee", "cue",   "are",
      "eh s", "tee", "you", "vee", "double u", "ex",  "why", "zee"};
  if (label != nullptr && label[0] >= 'a' && label[0] <= 'z' &&
      label[1] == '\0') {
    return kLetterSpoken[label[0] - 'a'];
  }
  return label;
}

}  // namespace spelling

#endif  // SPELLING_SPELLING_LABELS_H_
