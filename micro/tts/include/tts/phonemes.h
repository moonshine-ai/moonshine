// English phoneme inventory for the formant synthesizer.
//
// Keyed by IPA (UTF-8), matching the G2P front-end output.
// Each entry carries the data a Klatt-style cascade synthesizer needs:
//   * a steady-state formant target (vowels / sonorants), or a formant
//     "locus" the transition engine aims at (obstruents),
//   * source amplitudes that decide what excites the resonators, and
//   * an intrinsic duration that prosody later scales.
//
// The numbers are adult-male reference values drawn from standard phonetics
// tables (Peterson-Barney vowels, Klatt locus theory). They are a starting
// point tuned for "robotic but understandable", not for naturalness, and are
// expected to be adjusted by ear.
//
// Only monophthongs and single consonants live in the table. Diphthongs and
// affricates are expanded to a pair of base symbols by the tokenizer (see
// TokenizeIpa), so the transition engine renders the glide/closure for free.

#ifndef TTS_PHONEMES_H_
#define TTS_PHONEMES_H_

#include <cstdint>
#include <string>
#include <vector>

namespace tts {

enum class PhoneClass : uint8_t {
  kVowel,
  kNasal,
  kStop,
  kFricative,
  kApproximant,
  kLateral,
  kSilence,
};

enum class Source : uint8_t {
  kVoiced,     // glottal pulse only
  kVoiceless,  // noise only (frication / aspiration)
  kMixed,      // voicing + frication (voiced fricatives)
  kSilence,
};

struct Phone {
  const char* ipa;  // UTF-8 key
  PhoneClass cls;
  Source src;

  // Formant targets (vowels/sonorants) or consonant loci (obstruents), Hz.
  float f1, f2, f3;
  // Resonator bandwidths, Hz.
  float b1, b2, b3;

  float dur_ms;  // intrinsic duration before prosodic scaling

  // Nasal pole/zero (nasals only); 0 if unused.
  float fnp, fnz;

  // Frication / burst noise centre frequency, Hz (fricatives + stop bursts).
  float fric_cf;

  // Relative source amplitudes in [0, 1].
  float av;  // voicing
  float af;  // frication
  float ah;  // aspiration
};

// Look up a phone by exact IPA key in the built-in default table. Returns
// nullptr if unknown. (Runtime synthesis goes through VoiceParams::Lookup so a
// loaded config can override these; this is just the compiled-in default.)
const Phone* LookupPhone(const std::string& ipa);

// A fresh, mutable copy of the built-in default phone table. The `ipa` pointers
// still reference static string literals (stable for the program lifetime), so
// callers may mutate the numeric fields but must not free or rewrite `ipa`.
std::vector<Phone> DefaultPhoneTable();

// Split an IPA string into a sequence of known base-phone keys.
//
// Handles multi-codepoint clusters: diphthongs (eɪ aɪ aʊ ɔɪ oʊ) and
// affricates (tʃ dʒ) are decomposed into their base symbols; length marks
// (ː) are dropped; a few common alternate symbols (ɚ r ɹ̩ etc.) are folded to
// table keys. Unknown codepoints are skipped.
//
// Stress marks (ˈ primary, ˌ secondary) are preserved as their own tokens so
// the synth can place pitch accents on the following vowel; they are not
// phones.
//
// Whitespace is preserved as the literal token " " so the caller can insert
// inter-word pauses.
std::vector<std::string> TokenizeIpa(const std::string& ipa);

}  // namespace tts

#endif  // TTS_PHONEMES_H_
