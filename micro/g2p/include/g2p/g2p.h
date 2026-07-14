// English grapheme-to-phoneme front end.
//
// Pipeline per word: runtime override -> number normalizer -> baked common-word
// dictionary -> rule-based letter-to-sound. The result is IPA, folded to the
// synths' base-phone tokens by TokenizeIpa.
//
//   - Overrides:  g2p_dict.h  (user-supplied word->IPA, e.g. proper nouns)
//   - Numbers:    src/g2p_numbers.h
//   - Dictionary: g2p_dict.h + generated src/g2p_dict_data.h
//   - Rules:      src/g2p_rules.h
//
// The token vocabulary is shared by both TTS back ends (the Klatt formant
// synth in tts/ and the neural diphone synth in neural-tts/): one UTF-8 IPA
// base phone per token, plus " " (word gap), "." (sentence pause), and the
// stress marks "ˈ" / "ˌ" as their own tokens.

#ifndef G2P_G2P_H_
#define G2P_G2P_H_

#include <string>
#include <vector>

#include "g2p_dict.h"  // Lexicon

namespace g2p {

// Convert plain English text to a flat list of base-phone tokens, including
// " " and "." pause tokens for word gaps and sentence breaks. `overrides`, if
// non-null, is consulted first.
std::vector<std::string> TextToPhones(const std::string& text,
                                      const Lexicon* overrides = nullptr);

// Split an IPA string into a sequence of base-phone tokens.
//
// Handles multi-codepoint clusters: diphthongs (eɪ aɪ aʊ ɔɪ oʊ) and
// affricates (tʃ dʒ) are decomposed into their base symbols; length marks
// (ː) are dropped; a few common alternate symbols (ɚ ɡ r a y etc.) are folded
// to canonical keys. Unknown codepoints are skipped.
//
// Stress marks (ˈ primary, ˌ secondary) are preserved as their own tokens so
// a synth can place pitch accents on the following vowel; they are not
// phones. Whitespace is preserved as the literal token " " so the caller can
// insert inter-word pauses.
std::vector<std::string> TokenizeIpa(const std::string& ipa);

}  // namespace g2p

#endif  // G2P_G2P_H_
