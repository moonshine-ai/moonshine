#include "g2p/g2p_phones.h"

#include <cctype>
#include <cstring>
#include <string>

#include "g2p/g2p.h"
#include "g2p_dict.h"
#include "g2p_numbers.h"
#include "g2p_rules.h"

namespace g2p {

namespace {

bool HasDigit(const char* s) {
  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p;
       ++p) {
    if (std::isdigit(*p)) return true;
  }
  return false;
}

// Resolve one word token to IPA in `out` (capacity includes NUL). Peak heap
// use is one short std::string inside the dictionary/rules path, not a token
// vector.
bool ResolveTokenBuf(const char* tok, char* out, std::size_t cap,
                     const Lexicon* overrides) {
  if (tok == nullptr || out == nullptr || cap == 0) return false;
  std::string ipa;
  if (overrides != nullptr && overrides->Lookup(tok, &ipa)) {
  } else if (HasDigit(tok) && NumberWordToIpa(tok, &ipa)) {
  } else if (DictLookup(tok, &ipa)) {
  } else if (LetterHomophoneToIpa(tok, &ipa)) {
  } else {
    ipa = RulesWordToIpa(tok);
  }
  if (ipa.empty() || ipa.size() + 1 > cap) return false;
  std::memcpy(out, ipa.c_str(), ipa.size() + 1);
  return true;
}

}  // namespace

bool PhoneTokenList::push(const char* tok) {
  if (tok == nullptr || count >= kMaxTokens) return false;
  std::strncpy(tokens[count], tok, kMaxLen - 1);
  tokens[count][kMaxLen - 1] = '\0';
  ++count;
  return true;
}

bool TokenizeIpaToList(const char* ipa, PhoneTokenList* out) {
  if (ipa == nullptr || out == nullptr) return false;
  for (const std::string& t : TokenizeIpa(std::string(ipa))) {
    if (!out->push(t.c_str())) return false;
  }
  return true;
}

bool TextToPhoneList(const char* text, PhoneTokenList* out,
                     const Lexicon* overrides) {
  if (text == nullptr || out == nullptr) return false;
  out->count = 0;

  char tok[64];
  int tlen = 0;
  char ipa[128];

  auto flush = [&]() -> bool {
    if (tlen <= 0) return true;
    tok[tlen] = '\0';
    if (!ResolveTokenBuf(tok, ipa, sizeof(ipa), overrides)) return false;
    if (!TokenizeIpaToList(ipa, out)) return false;
    tlen = 0;
    return true;
  };

  for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text);
       *p != '\0'; ++p) {
    const char c = static_cast<char>(*p);
    const bool is_tok =
        (*p < 0x80) && (std::isalnum(*p) || c == '\'');
    bool extend_tok = is_tok;
    if (!extend_tok && (c == '.' || c == ',')) {
      const char prev = tlen > 0 ? tok[tlen - 1] : '\0';
      const char next = p[1];
      if (std::isdigit(static_cast<unsigned char>(prev)) &&
          std::isdigit(static_cast<unsigned char>(next))) {
        extend_tok = true;
      }
    }
    if (extend_tok) {
      if (tlen + 1 >= static_cast<int>(sizeof(tok))) return false;
      tok[tlen++] = c;
      continue;
    }
    if (!flush()) return false;
    switch (c) {
      case '.':
      case '!':
      case '?':
        if (!out->push(".")) return false;
        break;
      case ',':
      case ';':
      case ':':
      case ' ':
      case '\t':
      case '\n':
      case '\r':
        if (out->count > 0 && std::strcmp(out->tokens[out->count - 1], " ") != 0 &&
            std::strcmp(out->tokens[out->count - 1], ".") != 0) {
          if (!out->push(" ")) return false;
        }
        break;
      default:
        break;
    }
  }
  return flush();
}

}  // namespace g2p
