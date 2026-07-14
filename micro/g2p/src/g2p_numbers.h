// English number-token normalization (numeral -> IPA).
//
// Cardinals up to the trillions, decimals ("point"), negatives, and
// digit-by-digit reading for leading-zero strings. Returns IPA compatible with
// TokenizeIpa.

#ifndef G2P_NUMBERS_H_
#define G2P_NUMBERS_H_

#include <string>
#include <string_view>

namespace g2p {

// If `token` is a supported plain numeral (e.g. "123", "-4", "3.5", "007"),
// writes its IPA to *ipa and returns true; otherwise returns false.
bool NumberWordToIpa(std::string_view token, std::string* ipa);

}  // namespace g2p

#endif  // G2P_NUMBERS_H_
