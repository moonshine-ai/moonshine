// IPA string -> base-phone token stream (TokenizeIpa). Moved here from the
// Klatt synth's phonemes.cc so both TTS back ends share one tokenizer; the
// token set matches the synth phone tables and models/neural_tts/
// phone_vocab.json.

#include <array>
#include <cstddef>
#include <string>

#include "g2p.h"

namespace g2p {

namespace {

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
  // place pitch accents on the following vowel; they are not phones, so
  // segment building skips them.
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

}  // namespace g2p
