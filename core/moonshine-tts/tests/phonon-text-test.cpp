#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "phonon-text.h"

#include <doctest/doctest.h>

#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace moonshine_tts;

namespace {

// One token per whitespace word, so chunk budgets read as word counts.
int words(std::string_view text) {
  int n = 0;
  bool in_word = false;
  for (const char c : text) {
    const bool space = c == ' ' || c == '\n' || c == '\t';
    if (!space && !in_word) {
      ++n;
    }
    in_word = !space;
  }
  return n;
}

}  // namespace

TEST_CASE("prepare_prompt shapes a prompt the way training did") {
  // The reference runtime's own examples.
  PhononPrompt p = phonon_prepare_prompt("not a thing");
  CHECK(p.text == "Not a thing.");
  CHECK(p.words == 3);
  p = phonon_prepare_prompt("one two three four five");
  CHECK(p.text == "One two three four five.");
  CHECK(p.words == 5);

  // Whitespace is collapsed, existing punctuation is kept.
  CHECK(phonon_prepare_prompt("  hello,\n  world!  ").text == "Hello, world!");
  CHECK(phonon_prepare_prompt("").text.empty());
  CHECK(phonon_prepare_prompt("   ").words == 0);
  // A prompt ending in a digit gets its stop too.
  CHECK(phonon_prepare_prompt("room 101").text == "Room 101.");
  // One ending in a closing quote does not.
  CHECK(phonon_prepare_prompt("she said \"go\"").text == "She said \"go\"");
}

TEST_CASE("normalize_text folds typography") {
  CHECK(phonon_normalize_text("“Hello” — it’s ‘me’") == "\"Hello\" it's 'me'");
  CHECK(phonon_normalize_text("wait… what") == "wait. what");
  CHECK(phonon_normalize_text("a (quick) note; really: yes") ==
        "a, quick, note, really, yes");
  CHECK(phonon_normalize_text("at 10:30 sharp") == "at 10:30 sharp");
  CHECK(phonon_normalize_text("x = 2") == "x equals 2");
  CHECK(phonon_normalize_text("great 🎉 news") == "great news");
  CHECK(phonon_normalize_text("[note] {aside}") == "note aside");
  CHECK(phonon_normalize_text("• one\n• two") == "one two");
  // Letters the tokenizer has pieces for stay; the ones it would hand the
  // model as raw bytes fold to their base letter.
  CHECK(phonon_normalize_text("résumé naïve Łódź") == "résumé naïve Łódz");
  CHECK(phonon_normalize_text("2 × 3 ≈ 6") == "2 times 3 approximately 6");
  CHECK(phonon_normalize_text("20°C today") == "20 degrees Celsius today");
  CHECK(phonon_normalize_text("1° of arc") == "1 degree of arc");
  CHECK(phonon_normalize_text("~5 minutes") == "about 5 minutes");
}

TEST_CASE("normalize_text reads numbers out") {
  // The reference runtime's own table, which pins the readings the serving
  // stack speaks (including its quirks: commas are stripped wherever they
  // fall, so "1,23" is 123).
  const std::vector<std::pair<std::string, std::string>> cases = {
      {"0", "0"},
      {"123", "123"},
      {"2024", "2024"},
      {"1234", "1 thousand 234"},
      {"1,234.56...", "1 thousand 234 point 56..."},
      {"3.14", "3 point 14"},
      {"12.10", "12 point 10"},
      {"-12.00", "minus 12 point 00"},
      {"1000000", "1 million"},
      {"2500000", "2 million 500 thousand"},
      {"2,500,000,", "2 million 500 thousand,"},
      {"1002003004", "1 billion 2 million 3 thousand 4"},
      {"-4500", "minus 4 thousand 500"},
      {"-12.", "minus 12."},
      {"007", "zero zero seven"},
      {"0600.", "zero six zero zero."},
      {"7,000-ton", "7 thousand-ton"},
      {"20-year-old", "20-year-old"},
      {"20-25", "20 to 25"},
      {"10-5", "10-5"},
      {"22-22", "22-22"},
      {"3-5", "3-5"},
      {"12:30", "12:30"},
      {"10-15", "10 to 15"},
      {"1445-1450", "1445 to 1450"},
      {"2024-25", "2024 to 25"},
      {"1999-1998", "1999-1998"},
      {"3.5%", "3.5%"},
      {"1234567890123", "1234567890123"},
      {"hello", "hello"},
      {"12,345", "12 thousand 345"},
      {"1,2,3", "123"},
      {"1,23", "123"},
      {"1234,567", "1 million 234 thousand 567"},
      {"1,999", "1999"},
      {"0,123", "123"},
  };
  for (const auto& [input, expected] : cases) {
    CAPTURE(input);
    CHECK(phonon_normalize_text(input) == expected);
  }
  CHECK(phonon_normalize_text("$1,500 is 1,500.") ==
        "$1,500 is 1 thousand 500.");
  CHECK(phonon_normalize_text("in 2024, 999 came") == "in 2024, 999 came");
  CHECK(phonon_normalize_text("a 1500-strong crowd") ==
        "a 1 thousand 500-strong crowd");
  // Digit runs are read group by group.
  CHECK(phonon_normalize_text("call 555 123 4567 now") ==
        "call five five five, one two three, four five six seven now");
  // @ and + are spoken, and the pieces get their own numbers turn.
  CHECK(phonon_normalize_text("me@example.com") == "me at example.com");
  CHECK(phonon_normalize_text("1500+20") == "1 thousand 500 plus 20");
}

TEST_CASE("plan_chunks groups whole sentences up to the budget") {
  const std::string text =
      "One two three. Four five six. Seven eight nine ten eleven. Twelve.";
  // Six words per chunk.
  std::vector<std::string> chunks = phonon_plan_chunks(text, words, 6, 200);
  REQUIRE(chunks.size() == 2);
  CHECK(chunks[0] == "One two three. Four five six.");
  CHECK(chunks[1] == "Seven eight nine ten eleven. Twelve.");
  chunks = phonon_plan_chunks(text, words, 5, 200);
  REQUIRE(chunks.size() == 4);
  CHECK(chunks[0] == "One two three.");
  CHECK(chunks[1] == "Four five six.");
  CHECK(chunks[2] == "Seven eight nine ten eleven.");
  CHECK(chunks[3] == "Twelve.");

  // A sentence over the soft budget stays whole.
  chunks =
      phonon_plan_chunks("Seven eight nine ten eleven twelve.", words, 3, 200);
  REQUIRE(chunks.size() == 1);

  // Over the hard budget it is cut, preferring a clause mark near the middle.
  chunks = phonon_plan_chunks(
      "Alpha beta gamma delta, epsilon zeta eta theta iota kappa.", words, 3,
      6);
  REQUIRE(chunks.size() == 2);
  CHECK(chunks[0] == "Alpha beta gamma delta,");
  CHECK(chunks[1] == "epsilon zeta eta theta iota kappa.");

  // A single enormous word is cut between its characters.
  const auto long_word = [](std::string_view t) {
    return static_cast<int>(t.size()) / 4;
  };
  chunks = phonon_plan_chunks(std::string(40, 'a'), long_word, 3, 6);
  CHECK(chunks.size() >= 2);
  CHECK(phonon_plan_chunks("", words, 50, 200).empty());
}
