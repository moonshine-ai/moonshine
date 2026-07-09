#include "phonemes.h"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <utility>

namespace tts {

namespace {

// Default bandwidths (Hz).
constexpr float kBv1 = 60.0f, kBv2 = 90.0f, kBv3 = 150.0f;  // vowels
// Nasals: F2/F3 heavily damped so the murmur is dominated by the low nasal
// formant (otherwise the formant structure reads as a lateral or a vowel).
// Tuned so the murmur is dominated by the low nasal formant; b2 is per-place.
constexpr float kBn1 = 120.0f, kBn3 = 300.0f;                 // nasals
constexpr float kBc1 = 100.0f, kBc2 = 150.0f, kBc3 = 220.0f;  // consonants

// The master phone table. See phonemes.h for the field meanings.
// clang-format off
const std::array<Phone, 37> kPhones = {{
  // --- Vowels ---------------------------------------------------------------
  // ipa  cls                    src                  f1    f2    f3    b1    b2    b3   dur   fnp  fnz  fricCf  av   af   ah
  {"i",  PhoneClass::kVowel,     Source::kVoiced,    270, 2290, 3010, kBv1, kBv2, kBv3, 130,   0,   0,    0,  1.0f, 0,   0},
  {"\u026A", PhoneClass::kVowel, Source::kVoiced,    383, 2140, 2550, kBv1, kBv2, kBv3,  90,   0,   0,    0,  1.0f, 0,   0}, // ɪ
  {"e",  PhoneClass::kVowel,     Source::kVoiced,    460, 1990, 2530, kBv1, kBv2, kBv3, 120,   0,   0,    0,  1.0f, 0,   0},
  {"\u025B", PhoneClass::kVowel, Source::kVoiced,    528, 1784, 2480, kBv1, kBv2, kBv3, 110,   0,   0,    0,  1.0f, 0,   0}, // ɛ
  {"\u00E6", PhoneClass::kVowel, Source::kVoiced,    722, 1822, 2410, kBv1, kBv2, kBv3, 150,   0,   0,    0,  1.0f, 0,   0}, // æ
  {"\u0251", PhoneClass::kVowel, Source::kVoiced,    747, 994, 2440, kBv1, kBv2, kBv3, 150,   0,   0,    0,  1.0f, 0,   0}, // ɑ
  {"\u0254", PhoneClass::kVowel, Source::kVoiced,    482,  834, 2410, kBv1, kBv2, kBv3, 140,   0,   0,    0,  1.0f, 0,   0}, // ɔ
  {"o",  PhoneClass::kVowel,     Source::kVoiced,    450,  900, 2300, kBv1, kBv2, kBv3, 120,   0,   0,    0,  1.0f, 0,   0},
  {"\u028A", PhoneClass::kVowel, Source::kVoiced,    440, 1020, 2240, kBv1, kBv2, kBv3,  90,   0,   0,    0,  1.0f, 0,   0}, // ʊ
  {"u",  PhoneClass::kVowel,     Source::kVoiced,    300,  870, 2240, kBv1, kBv2, kBv3, 130,   0,   0,    0,  1.0f, 0,   0},
  {"\u028C", PhoneClass::kVowel, Source::kVoiced,    582, 1247, 2390, kBv1, kBv2, kBv3, 110,   0,   0,    0,  1.0f, 0,   0}, // ʌ
  {"\u025D", PhoneClass::kVowel, Source::kVoiced,    490, 1350, 1690, kBv1, kBv2, kBv3, 150,   0,   0,    0,  1.0f, 0,   0}, // ɝ
  {"\u0259", PhoneClass::kVowel, Source::kVoiced,    426, 1498, 2500, kBv1, kBv2, kBv3,  70,   0,   0,    0,  1.0f, 0,   0}, // ə

  // --- Stops (loci + burst centre; expanded into closure/burst at synth) -----
  {"p",  PhoneClass::kStop,      Source::kVoiceless, 300,  720, 2200, kBc1, kBc2, kBc3,  90,   0,   0, 1200, 0.0f, 0.5f, 0.4f},
  {"b",  PhoneClass::kStop,      Source::kVoiced,    300,  720, 2200, kBc1, kBc2, kBc3,  80,   0,   0, 1200, 0.4f, 0.4f, 0.0f},
  {"t",  PhoneClass::kStop,      Source::kVoiceless, 300, 1750, 2600, kBc1, kBc2, kBc3,  90,   0,   0, 3800, 0.0f, 0.6f, 0.4f},
  {"d",  PhoneClass::kStop,      Source::kVoiced,    300, 1750, 2600, kBc1, kBc2, kBc3,  80,   0,   0, 3800, 0.4f, 0.5f, 0.0f},
  {"k",  PhoneClass::kStop,      Source::kVoiceless, 300, 1900, 2400, kBc1, kBc2, kBc3,  90,   0,   0, 2200, 0.0f, 0.5f, 0.5f},
  {"g",  PhoneClass::kStop,      Source::kVoiced,    300, 1900, 2400, kBc1, kBc2, kBc3,  80,   0,   0, 2200, 0.4f, 0.45f,0.0f},

  // --- Nasals ---------------------------------------------------------------
  // fnz is the place-dependent oral-cavity anti-resonance (the nasal place cue):
  // low for labial /m/, mid for alveolar /n/, high for velar /ŋ/.
  {"m",  PhoneClass::kNasal,     Source::kVoiced,    220, 1000, 2200, kBn1, 330, kBn3,  80, 250, 1033,  0,  1.0f, 0,   0},
  {"n",  PhoneClass::kNasal,     Source::kVoiced,    220, 1600, 2700, kBn1, 197, kBn3,  80, 250, 1308,  0,  1.0f, 0,   0},
  {"\u014B", PhoneClass::kNasal, Source::kVoiced,    220, 2000, 2600, kBn1, 259, kBn3,  80, 250, 2415,  0,  1.0f, 0,   0}, // ŋ

  // --- Fricatives -----------------------------------------------------------
  {"f",  PhoneClass::kFricative, Source::kVoiceless, 300, 1100, 2200, kBc1, kBc2, kBc3, 110,   0,   0, 1827, 0.0f, 0.18f,0},
  {"v",  PhoneClass::kFricative, Source::kMixed,     300, 1100, 2200, kBc1, kBc2, kBc3,  80,   0,   0, 1827, 0.25f,0.16f,0},
  {"\u03B8", PhoneClass::kFricative, Source::kVoiceless, 300, 1400, 2400, kBc1, kBc2, kBc3, 100, 0, 0, 2770, 0.0f, 0.16f,0}, // θ
  {"\u00F0", PhoneClass::kFricative, Source::kMixed, 300, 1400, 2400, kBc1, kBc2, kBc3,  70,   0,   0, 2770, 0.25f,0.14f,0}, // ð
  {"s",  PhoneClass::kFricative, Source::kVoiceless, 300, 1700, 2600, kBc1, kBc2, kBc3, 120,   0,   0, 5344, 0.0f, 0.7f, 0},
  {"z",  PhoneClass::kFricative, Source::kMixed,     300, 1700, 2600, kBc1, kBc2, kBc3,  90,   0,   0, 5344, 0.3f, 0.5f, 0},
  {"\u0283", PhoneClass::kFricative, Source::kVoiceless, 300, 1800, 2500, kBc1, kBc2, kBc3, 120, 0, 0, 2939, 0.0f, 0.75f,0}, // ʃ
  {"\u0292", PhoneClass::kFricative, Source::kMixed, 300, 1800, 2500, kBc1, kBc2, kBc3,  90,   0,   0, 2939, 0.3f, 0.55f,0}, // ʒ
  {"h",  PhoneClass::kFricative, Source::kVoiceless, 500, 1500, 2500, kBc1, kBc2, kBc3,  70,   0,   0,    0, 0.0f, 0.0f, 0.5f},

  // --- Approximants & lateral ----------------------------------------------
  {"\u0279", PhoneClass::kApproximant, Source::kVoiced, 330, 1100, 1600, kBv1, kBv2, kBv3, 80, 0, 0, 0, 1.0f, 0, 0}, // ɹ
  {"j",  PhoneClass::kApproximant, Source::kVoiced,    250, 2300, 3000, kBv1, kBv2, kBv3,  70,   0,   0,    0,  1.0f, 0, 0},
  {"w",  PhoneClass::kApproximant, Source::kVoiced,    290,  610, 2150, kBv1, kBv2, kBv3,  80,   0,   0,    0,  1.0f, 0, 0},
  {"l",  PhoneClass::kLateral,     Source::kVoiced,    360, 1300, 2700, kBv1, kBv2, kBv3,  80,   0,   0,    0,  1.0f, 0, 0},

  // --- Silence (used for pauses) -------------------------------------------
  {" ",  PhoneClass::kSilence,  Source::kSilence,   500, 1500, 2500, kBv1, kBv2, kBv3,  60,   0,   0,    0,  0.0f, 0, 0},
  {".",  PhoneClass::kSilence,  Source::kSilence,   500, 1500, 2500, kBv1, kBv2, kBv3, 220,   0,   0,    0,  0.0f, 0, 0},
}};
// clang-format on

const std::unordered_map<std::string, const Phone*>& PhoneIndex() {
  static const std::unordered_map<std::string, const Phone*> index = [] {
    std::unordered_map<std::string, const Phone*> m;
    for (const Phone& p : kPhones) {
      m.emplace(p.ipa, &p);
    }
    return m;
  }();
  return index;
}

// Ordered (longest-first) rewrite rules used by the tokenizer. Each pattern,
// when it prefixes the remaining input, emits zero or more base-phone keys.
struct Rule {
  const char* pattern;
  std::array<const char*, 2> emit;  // nullptr entries are ignored
};

// clang-format off
const std::array<Rule, 42> kRules = {{
  // Diphthongs -> two vowel targets (the smoother renders the glide).
  {"e\u026A",      {"e", "\u026A"}},        // eɪ
  {"\u0251\u026A", {"\u0251", "\u026A"}},    // aɪ -> ɑɪ
  {"a\u026A",      {"\u0251", "\u026A"}},    // aɪ (ascii a)
  {"\u0251\u028A", {"\u0251", "\u028A"}},    // aʊ -> ɑʊ
  {"a\u028A",      {"\u0251", "\u028A"}},    // aʊ (ascii a)
  {"\u0254\u026A", {"\u0254", "\u026A"}},    // ɔɪ
  {"o\u028A",      {"o", "\u028A"}},         // oʊ
  {"\u0259\u028A", {"o", "\u028A"}},         // əʊ -> oʊ
  // Affricates -> stop + fricative.
  {"t\u0283",      {"t", "\u0283"}},         // tʃ
  {"d\u0292",      {"d", "\u0292"}},         // dʒ
  // Length mark -> dropped. Stress marks are kept as tokens so the synth can
  // place pitch accents on the following vowel (see Synthesize); they are not
  // phones, so Lookup ignores them during segment building.
  {"\u02D0",       {nullptr, nullptr}},      // ː
  {"\u02C8",       {"\u02C8", nullptr}},     // ˈ primary stress
  {"\u02CC",       {"\u02CC", nullptr}},     // ˌ secondary stress
  // Folded alternates.
  {"\u0261",       {"g", nullptr}},          // ɡ (script g) -> g
  {"\u025A",       {"\u025D", nullptr}},     // ɚ -> ɝ
  {"\u0258",       {"\u0259", nullptr}},     // ɘ -> ə
  {"\u0250",       {"\u028C", nullptr}},     // ɐ -> ʌ
  {"\u025C",       {"\u025D", nullptr}},     // ɜ -> ɝ
  {"\u0252",       {"\u0254", nullptr}},     // ɒ -> ɔ
  {"\u027E",       {"d", nullptr}},          // ɾ flap -> d
  // Single IPA codepoints already in the table.
  {"\u026A", {"\u026A", nullptr}},  // ɪ
  {"\u025B", {"\u025B", nullptr}},  // ɛ
  {"\u00E6", {"\u00E6", nullptr}},  // æ
  {"\u0251", {"\u0251", nullptr}},  // ɑ
  {"\u0254", {"\u0254", nullptr}},  // ɔ
  {"\u028A", {"\u028A", nullptr}},  // ʊ
  {"\u028C", {"\u028C", nullptr}},  // ʌ
  {"\u025D", {"\u025D", nullptr}},  // ɝ
  {"\u0259", {"\u0259", nullptr}},  // ə
  {"\u014B", {"\u014B", nullptr}},  // ŋ
  {"\u03B8", {"\u03B8", nullptr}},  // θ
  {"\u00F0", {"\u00F0", nullptr}},  // ð
  {"\u0283", {"\u0283", nullptr}},  // ʃ
  {"\u0292", {"\u0292", nullptr}},  // ʒ
  {"\u0279", {"\u0279", nullptr}},  // ɹ
  // ASCII letters.
  {"a", {"\u0251", nullptr}},  // a -> ɑ
  {"r", {"\u0279", nullptr}},  // r -> ɹ
  {"y", {"j", nullptr}},       // y -> j (some G2Ps use y)
  {"g", {"g", nullptr}},
  // Whitespace -> pause token.
  {" ", {" ", nullptr}},
  {"\t", {" ", nullptr}},
  {"\n", {" ", nullptr}},
}};
// clang-format on

// Single ASCII phones that map straight to a table key (and aren't covered by
// the rule table above).
bool IsDirectAscii(char c) {
  switch (c) {
    case 'i':
    case 'e':
    case 'o':
    case 'u':
    case 'p':
    case 'b':
    case 't':
    case 'd':
    case 'k':
    case 'm':
    case 'n':
    case 'f':
    case 'v':
    case 's':
    case 'z':
    case 'h':
    case 'w':
    case 'j':
    case 'l':
      return true;
    default:
      return false;
  }
}

size_t Utf8Len(unsigned char lead) {
  if (lead < 0x80) return 1;
  if ((lead >> 5) == 0x6) return 2;
  if ((lead >> 4) == 0xE) return 3;
  if ((lead >> 3) == 0x1E) return 4;
  return 1;
}

}  // namespace

const Phone* LookupPhone(const std::string& ipa) {
  const auto& idx = PhoneIndex();
  auto it = idx.find(ipa);
  return it == idx.end() ? nullptr : it->second;
}

std::vector<Phone> DefaultPhoneTable() {
  return std::vector<Phone>(kPhones.begin(), kPhones.end());
}

std::vector<std::string> TokenizeIpa(const std::string& ipa) {
  std::vector<std::string> out;
  size_t i = 0;
  const size_t n = ipa.size();
  while (i < n) {
    bool matched = false;

    // Try the rewrite rules (already ordered longest-first for the multi-byte
    // clusters that must win over their single-codepoint prefixes).
    for (const Rule& r : kRules) {
      const size_t plen = std::char_traits<char>::length(r.pattern);
      if (plen == 0 || i + plen > n) continue;
      if (ipa.compare(i, plen, r.pattern) == 0) {
        for (const char* e : r.emit) {
          if (e != nullptr) out.emplace_back(e);
        }
        i += plen;
        matched = true;
        break;
      }
    }
    if (matched) continue;

    // Direct single-byte ASCII phones.
    const unsigned char c = static_cast<unsigned char>(ipa[i]);
    if (c < 0x80 && IsDirectAscii(static_cast<char>(c))) {
      out.emplace_back(1, static_cast<char>(c));
      ++i;
      continue;
    }

    // Unknown codepoint: skip the whole UTF-8 sequence.
    i += Utf8Len(c);
  }
  return out;
}

}  // namespace tts
