#include "g2p.h"

#include <cctype>
#include <string>

#include "g2p_dict.h"
#include "g2p_numbers.h"
#include "g2p_rules.h"
#include "phonemes.h"

namespace tts {

namespace {

bool HasDigit(const std::string& s) {
  for (unsigned char c : s) {
    if (std::isdigit(c)) return true;
  }
  return false;
}

// Resolve a single token to an IPA string via the lookup pipeline.
std::string ResolveToken(const std::string& tok, const Lexicon* overrides) {
  std::string ipa;
  if (overrides != nullptr && overrides->Lookup(tok, &ipa)) return ipa;
  if (HasDigit(tok) && NumberWordToIpa(tok, &ipa)) return ipa;
  if (DictLookup(tok, &ipa)) return ipa;
  return RulesWordToIpa(tok);
}

}  // namespace

std::vector<std::string> TextToPhones(const std::string& text,
                                      const Lexicon* overrides) {
  std::vector<std::string> out;
  std::string tok;

  auto flush = [&]() {
    if (tok.empty()) return;
    for (std::string& t : TokenizeIpa(ResolveToken(tok, overrides))) {
      out.push_back(std::move(t));
    }
    tok.clear();
  };

  const size_t n = text.size();
  for (size_t i = 0; i < n; ++i) {
    const char c = text[i];
    const unsigned char uc = static_cast<unsigned char>(c);
    bool is_tok = uc < 0x80 && (std::isalnum(uc) || c == '\'');
    // A '.' or ',' is part of a number (decimal point / grouping) only when it
    // sits between two digits ("3.5", "1,000"); otherwise it's punctuation.
    if (!is_tok && (c == '.' || c == ',')) {
      const char prev = tok.empty() ? '\0' : tok.back();
      const char next = (i + 1 < n) ? text[i + 1] : '\0';
      if (std::isdigit(static_cast<unsigned char>(prev)) &&
          std::isdigit(static_cast<unsigned char>(next))) {
        is_tok = true;
      }
    }
    if (is_tok) {
      tok.push_back(c);
      continue;
    }
    flush();
    switch (c) {
      case '.':
      case '!':
      case '?':
        out.emplace_back(".");  // sentence pause
        break;
      case ',':
      case ';':
      case ':':
      case ' ':
      case '\t':
      case '\n':
      case '\r':
        if (!out.empty() && out.back() != " " && out.back() != ".") {
          out.emplace_back(" ");  // word gap / short pause
        }
        break;
      default:
        break;
    }
  }
  flush();
  return out;
}

}  // namespace tts
