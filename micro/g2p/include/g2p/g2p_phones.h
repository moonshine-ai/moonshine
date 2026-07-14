// Heap-free phone-token list for on-device TTS (CYW43 leaves little malloc headroom).
//
// PhoneTokenList stores the flat token stream produced by TextToPhones; each
// token is a short UTF-8 base-phone key (same vocabulary as TokenizeIpa).

#ifndef G2P_G2P_PHONES_H_
#define G2P_G2P_PHONES_H_

#include <cstddef>

#include "g2p_dict.h"  // Lexicon

namespace g2p {

struct PhoneTokenList {
  static constexpr int kMaxTokens = 192;
  static constexpr int kMaxLen = 8;
  char tokens[kMaxTokens][kMaxLen];
  int count = 0;

  bool push(const char* tok);
  const char* operator[](int i) const { return tokens[i]; }
};

// Plain-text -> base-phone tokens without std::vector/std::string output.
// Returns false if the stream overflows kMaxTokens.
bool TextToPhoneList(const char* text, PhoneTokenList* out,
                     const Lexicon* overrides = nullptr);

// IPA string -> base-phone tokens (heap-free output).
bool TokenizeIpaToList(const char* ipa, PhoneTokenList* out);

}  // namespace g2p

#endif  // G2P_G2P_PHONES_H_
