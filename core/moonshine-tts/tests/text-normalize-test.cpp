#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "text-normalize.h"

using namespace moonshine_tts;

TEST_CASE("split_text_to_words") {
  const auto w = split_text_to_words("  a  b\tc\n");
  REQUIRE(w.size() == 3);
  CHECK(w[0] == "a");
  CHECK(w[1] == "b");
  CHECK(w[2] == "c");
}

TEST_CASE("normalize_word_for_lookup") {
  CHECK(normalize_word_for_lookup("Hello,") == "hello");
  CHECK(normalize_word_for_lookup("") == "");
}

TEST_CASE("normalize_grapheme_key strips alternate suffix") {
  CHECK(normalize_grapheme_key("hello(2)") == "hello");
  CHECK(normalize_grapheme_key("Hello") == "hello");
}
