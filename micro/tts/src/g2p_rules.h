// Rule-based English letter-to-sound (grapheme -> IPA).
//
// Longest-first grapheme literals, r-controlled vowels, magic-e,
// open/closed-syllable vowel quality, soft/hard c & g, th-voicing, a small
// function-word map, and primary stress insertion. Fallback when a word is not
// in the baked dictionary or a user override.
//
// The returned IPA string is consumed by TokenizeIpa (phonemes.h), which folds
// any symbols not directly in the synth table (e.g. ɚ->ɝ, ɡ->g, ː dropped).

#ifndef TTS_G2P_RULES_H_
#define TTS_G2P_RULES_H_

#include <string>
#include <string_view>

namespace tts {

// Convert a single (already word-segmented) English word to an IPA string.
// Non-letters are ignored; an empty/letterless input yields "".
std::string RulesWordToIpa(std::string_view word);

}  // namespace tts

#endif  // TTS_G2P_RULES_H_
