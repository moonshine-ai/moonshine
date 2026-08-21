#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "sentence-splitter.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

using namespace moonshine_tts;

namespace {

std::vector<std::string> split(const std::string& text,
                               const std::string& language = "en") {
  SentenceSplitOptions options;
  options.language = language;
  return split_sentences(text, options);
}

}  // namespace

TEST_CASE("plain sentences split on terminator plus whitespace") {
  const auto parts = split("Hello there. How are you? Fine!");
  REQUIRE(parts.size() == 3);
  CHECK(parts[0] == "Hello there.");
  CHECK(parts[1] == "How are you?");
  CHECK(parts[2] == "Fine!");
}

TEST_CASE("empty and whitespace-only input yields nothing") {
  CHECK(split("").empty());
  CHECK(split("   \n\t ").empty());
}

TEST_CASE("text with no terminator stays one unit") {
  const auto parts = split("no terminator here");
  REQUIRE(parts.size() == 1);
  CHECK(parts[0] == "no terminator here");
}

TEST_CASE("colon splits by default and can be turned off") {
  const auto on = split("Warning: the core is hot.");
  REQUIRE(on.size() == 2);
  CHECK(on[0] == "Warning:");
  CHECK(on[1] == "the core is hot.");

  SentenceSplitOptions options;
  options.split_on_colon = false;
  const auto off = split_sentences("Warning: the core is hot.", options);
  REQUIRE(off.size() == 1);
  CHECK(off[0] == "Warning: the core is hot.");
}

TEST_CASE("decimals and times are never boundaries") {
  const auto pi = split("Pi is 3.14 exactly.");
  REQUIRE(pi.size() == 1);
  CHECK(pi[0] == "Pi is 3.14 exactly.");

  const auto clock = split("Meet at 12:30 sharp.");
  REQUIRE(clock.size() == 1);
}

TEST_CASE("titles do not end a sentence") {
  const auto parts = split("Dr. Smith saw Mrs. Jones. They left.");
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "Dr. Smith saw Mrs. Jones.");
  CHECK(parts[1] == "They left.");
}

TEST_CASE("single initials do not end a sentence") {
  const auto parts = split("J. R. R. Tolkien wrote it. We read it.");
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "J. R. R. Tolkien wrote it.");
  CHECK(parts[1] == "We read it.");
}

TEST_CASE("e.g. and i.e. survive as one unit") {
  const auto parts = split("Bring gear, e.g. rope. Then climb.");
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "Bring gear, e.g. rope.");
}

TEST_CASE("a following lowercase word suppresses an unknown abbreviation") {
  const auto parts = split("The Ph.D. candidates arrived.");
  REQUIRE(parts.size() == 1);
  CHECK(parts[0] == "The Ph.D. candidates arrived.");
}

TEST_CASE("German abbreviations are language-gated") {
  const auto de = split("Wir kaufen Obst, z.B. Äpfel. Dann gehen wir.", "de");
  REQUIRE(de.size() == 2);
  CHECK(de[0] == "Wir kaufen Obst, z.B. Äpfel.");
}

TEST_CASE("closing punctuation stays with the unit it ends") {
  const auto parts = split("He said \"Stop!\" Then he left.");
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "He said \"Stop!\"");
  CHECK(parts[1] == "Then he left.");
}

TEST_CASE("no split inside brackets") {
  const auto parts = split("Check it (see fig. 2. it is clear) now. Done.");
  REQUIRE(parts.size() == 2);
  CHECK(parts[1] == "Done.");
}

TEST_CASE("CJK terminators need no whitespace") {
  const auto parts = split("こんにちは。元気ですか？はい。", "ja");
  REQUIRE(parts.size() == 3);
  CHECK(parts[0] == "こんにちは。");
  CHECK(parts[1] == "元気ですか？");
  CHECK(parts[2] == "はい。");
}

TEST_CASE("CJK closing quote stays with its unit") {
  const auto parts = split("「やめて。」と言った。", "ja");
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "「やめて。」");
  CHECK(parts[1] == "と言った。");
}

TEST_CASE("Devanagari danda splits") {
  const auto parts = split("नमस्ते। आप कैसे हैं।", "hi");
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "नमस्ते।");
}

TEST_CASE("Arabic question mark splits") {
  const auto parts = split("كيف حالك؟ أنا بخير.", "ar");
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "كيف حالك؟");
}

TEST_CASE("Greek semicolon is a question mark only for Greek") {
  const auto el = split("Τι κάνεις; Καλά.", "el");
  REQUIRE(el.size() == 2);
  CHECK(el[0] == "Τι κάνεις;");

  const auto en = split("First part; second part.");
  REQUIRE(en.size() == 1);
}

TEST_CASE("incremental split holds back an unconfirmed tail") {
  SentenceSplitOptions options;
  options.language = "en";
  const SentenceSplit partial =
      split_sentences_incremental("Hello. 3", options);
  REQUIRE(partial.units.size() == 1);
  CHECK(partial.units[0] == "Hello.");
  CHECK(partial.remainder == "3");

  // A terminator at the very end could still turn into "3.14", so it waits.
  const SentenceSplit pending =
      split_sentences_incremental("Hello. 3.", options);
  REQUIRE(pending.units.size() == 1);
  CHECK(pending.remainder == "3.");
}

TEST_CASE("incremental split closes CJK units immediately") {
  SentenceSplitOptions options;
  options.language = "ja";
  const SentenceSplit s =
      split_sentences_incremental("こんにちは。あ", options);
  REQUIRE(s.units.size() == 1);
  CHECK(s.units[0] == "こんにちは。");
  CHECK(s.remainder == "あ");
}

TEST_CASE("short units merge forward when a minimum is set") {
  SentenceSplitOptions options;
  options.language = "en";
  options.min_codepoints = 12;
  const auto parts = split_sentences("Hi. Welcome to the show. Bye.", options);
  REQUIRE(parts.size() == 2);
  CHECK(parts[0] == "Hi. Welcome to the show.");
  CHECK(parts[1] == "Bye.");
}

TEST_CASE("ellipsis ends a unit but a bare one mid-word does not") {
  const auto spaced = split("Wait... what? Nothing.");
  REQUIRE(spaced.size() == 2);
  CHECK(spaced[0] == "Wait... what?");

  const auto tight = split("a…b is one token.");
  REQUIRE(tight.size() == 1);
}
