// English grapheme-to-phoneme front end.
//
// Pipeline per word: runtime override -> number normalizer -> baked common-word
// dictionary -> rule-based letter-to-sound. The result is IPA, folded to the
// synth's phone tokens by TokenizeIpa (phonemes.h).
//
//   - Overrides:  g2p_dict.h  (user-supplied word->IPA, e.g. proper nouns)
//   - Numbers:    g2p_numbers.h
//   - Dictionary: g2p_dict.h + generated g2p_dict_data.h
//   - Rules:      g2p_rules.h
//
// For evaluating synthesis quality without the G2P, feed IPA directly (the
// --ipa flag in main.cc, which calls TokenizeIpa).

#ifndef TTS_G2P_H_
#define TTS_G2P_H_

#include <string>
#include <vector>

#include "g2p_dict.h"  // Lexicon

namespace tts {

// Convert plain English text to a flat list of base-phone tokens (the same
// vocabulary TokenizeIpa produces), including " " and "." pause tokens for word
// gaps and sentence breaks. `overrides`, if non-null, is consulted first.
std::vector<std::string> TextToPhones(const std::string& text,
                                      const Lexicon* overrides = nullptr);

}  // namespace tts

#endif  // TTS_G2P_H_
