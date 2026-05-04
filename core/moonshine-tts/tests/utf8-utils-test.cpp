#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "utf8-utils.h"

using namespace moonshine_tts;

TEST_CASE("utf8_split_codepoints ascii") {
  const auto v = utf8_split_codepoints("abc");
  REQUIRE(v.size() == 3);
  CHECK(v[0] == "a");
  CHECK(v[1] == "b");
  CHECK(v[2] == "c");
}

TEST_CASE("utf8_split_codepoints two-byte") {
  const std::string s = "\xC3\xA9";  // é
  const auto v = utf8_split_codepoints(s);
  REQUIRE(v.size() == 1);
  CHECK(v[0] == s);
}

TEST_CASE("utf8_find_token_codepoints") {
  const std::string text = "I read";
  const auto hit = utf8_find_token_codepoints(text, "read", 0);
  REQUIRE(hit);
  CHECK(hit->first == 2);
  CHECK(hit->second == 6);
}

TEST_CASE("digit_ascii_span_expandable_python_w") {
  // Space-delimited digit expands; digit glued to katakana (U+30EA リ) does not.
  const std::string glued = "\xe3\x83\xaa" "3";  // リ + ASCII 3
  REQUIRE(glued.size() == 4);
  CHECK_FALSE(digit_ascii_span_expandable_python_w(glued, 3, 4));

  const std::string spaced = "x 3 , y";
  const size_t q = spaced.find('3');
  REQUIRE(q != std::string::npos);
  CHECK(digit_ascii_span_expandable_python_w(spaced, q, q + 1));
}
