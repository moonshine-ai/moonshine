#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "english-hand-oov.h"
#include "english-numbers.h"

#include <string>

using moonshine_tts::english_hand_oov_rules_ipa;
using moonshine_tts::english_number_token_ipa;

TEST_CASE("english_number_token_ipa") {
  CHECK(english_number_token_ipa("42").value() == "fˈɔɹtiˌtˈu");
  CHECK(english_number_token_ipa("007").value().find("ˈzɪroʊ") != std::string::npos);
  CHECK_FALSE(english_number_token_ipa("hello").has_value());
}

TEST_CASE("english_hand_oov_rules_ipa nonempty") {
  const std::string s = english_hand_oov_rules_ipa("xyzqx");
  CHECK(s.size() > 2);
}
