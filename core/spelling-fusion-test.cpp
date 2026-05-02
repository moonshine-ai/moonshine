#include "spelling-fusion.h"

#include <string>

#include "spelling-fusion-data.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

// Helper: shorthand for "matcher said this character".
SpellingMatch char_match(const std::string &c) {
  SpellingMatch m;
  m.type = SpellingMatchType::CHARACTER;
  m.character = c;
  return m;
}

// Helper: shorthand for "no match".
SpellingMatch no_match() { return SpellingMatch{}; }

}  // namespace

TEST_CASE("spelling-fusion: normalize") {
  CHECK(spelling_normalize("") == "");
  CHECK(spelling_normalize("Hello") == "hello");
  CHECK(spelling_normalize("\"Aww,\"") == "aww");
  CHECK(spelling_normalize("That's it.") == "thats it");
  CHECK(spelling_normalize("  upper   case  ") == "upper case");
  // Curly quotes (U+2018, U+2019, U+201C, U+201D) are stripped.
  // Use string-literal concatenation to break the hex-escape run so the
  // compiler doesn't read ``\x9CAww`` as a single oversized escape.
  CHECK(spelling_normalize("\xE2\x80\x9C" "Aww." "\xE2\x80\x9D") == "aww");
}

TEST_CASE("spelling-fusion: matcher classifies plain letters") {
  SpellingMatcher matcher;
  CHECK(matcher.classify("a").character == "a");
  CHECK(matcher.classify("hey").character == "a");
  CHECK(matcher.classify("Bee").character == "b");
  CHECK(matcher.classify("see").character == "c");
  CHECK(matcher.classify("aitch").character == "h");
  // Trailing punctuation gets stripped before lookup.
  CHECK(matcher.classify("Bee.").character == "b");
}

TEST_CASE("spelling-fusion: matcher classifies NATO codewords") {
  SpellingMatcher matcher;
  CHECK(matcher.classify("alpha").character == "a");
  CHECK(matcher.classify("Alfa").character == "a");
  CHECK(matcher.classify("bravo").character == "b");
  CHECK(matcher.classify("charlie").character == "c");
  CHECK(matcher.classify("whiskey").character == "w");
  CHECK(matcher.classify("x-ray").character == "x");
  CHECK(matcher.classify("zulu").character == "z");
}

TEST_CASE("spelling-fusion: matcher classifies digits") {
  SpellingMatcher matcher;
  CHECK(matcher.classify("zero").character == "0");
  CHECK(matcher.classify("five").character == "5");
  CHECK(matcher.classify("9").character == "9");
  CHECK(matcher.classify("nine").character == "9");
  // Digit homophones.
  CHECK(matcher.classify("won").character == "1");
  CHECK(matcher.classify("too").character == "2");
  CHECK(matcher.classify("eight").character == "8");
}

TEST_CASE("spelling-fusion: matcher parses 10..1000 number words") {
  SpellingMatcher matcher;
  CHECK(matcher.classify("ten").character == "10");
  CHECK(matcher.classify("eleven").character == "11");
  CHECK(matcher.classify("twenty three").character == "23");
  CHECK(matcher.classify("thirty-five").character == "35");
  CHECK(matcher.classify("one hundred").character == "100");
  CHECK(matcher.classify("a hundred").character == "100");
  CHECK(matcher.classify("one hundred and ten").character == "110");
  CHECK(matcher.classify("one thousand").character == "1000");
}

TEST_CASE("spelling-fusion: matcher applies upper-case modifier") {
  SpellingMatcher matcher;
  CHECK(matcher.classify("capital h").character == "H");
  CHECK(matcher.classify("upper case bravo").character == "B");
  CHECK(matcher.classify("uppercase z").character == "Z");
  // Bare modifier is not a hit.
  CHECK_FALSE(matcher.classify("capital").is_recognized());
}

TEST_CASE("spelling-fusion: matcher recognizes speller patterns") {
  SpellingMatcher matcher;
  CHECK(matcher.classify("a as in alpha").character == "a");
  CHECK(matcher.classify("b for bravo").character == "b");
  CHECK(matcher.classify("m for mary").character == "m");
  // Wrong starting letter — should fall through to NONE.
  CHECK_FALSE(matcher.classify("a for bravo").is_character());
}

TEST_CASE("spelling-fusion: matcher classifies command words") {
  SpellingMatcher matcher;
  CHECK(matcher.classify("stop").type == SpellingMatchType::STOPPED);
  CHECK(matcher.classify("Done.").type == SpellingMatchType::STOPPED);
  CHECK(matcher.classify("that's it").type == SpellingMatchType::STOPPED);
  CHECK(matcher.classify("clear").type == SpellingMatchType::CLEAR);
  CHECK(matcher.classify("clear all").type == SpellingMatchType::CLEAR);
  CHECK(matcher.classify("undo").type == SpellingMatchType::UNDO);
  CHECK(matcher.classify("backspace").type == SpellingMatchType::UNDO);
}

TEST_CASE("spelling-fusion: matcher classifies special characters") {
  SpellingMatcher matcher;
  // Spot-check every spoken phrase that maps to a unique special char.
  CHECK(matcher.classify("dollar sign").character == "$");
  CHECK(matcher.classify("at sign").character == "@");
  CHECK(matcher.classify("hash").character == "#");
  CHECK(matcher.classify("hashtag").character == "#");
  CHECK(matcher.classify("pound sign").character == "#");
  CHECK(matcher.classify("percent").character == "%");
  CHECK(matcher.classify("ampersand").character == "&");
  CHECK(matcher.classify("asterisk").character == "*");
  CHECK(matcher.classify("hyphen").character == "-");
  CHECK(matcher.classify("underscore").character == "_");
  CHECK(matcher.classify("plus").character == "+");
  CHECK(matcher.classify("equals").character == "=");
  CHECK(matcher.classify("pipe").character == "|");
  CHECK(matcher.classify("backslash").character == "\\");
  CHECK(matcher.classify("forward slash").character == "/");
  CHECK(matcher.classify("tilde").character == "~");
  CHECK(matcher.classify("backtick").character == "`");
  CHECK(matcher.classify("apostrophe").character == "'");
  CHECK(matcher.classify("quote").character == "\"");
  CHECK(matcher.classify("space").character == " ");
  CHECK(matcher.classify("open paren").character == "(");
  CHECK(matcher.classify("close paren").character == ")");
  CHECK(matcher.classify("open bracket").character == "[");
  CHECK(matcher.classify("right brace").character == "}");
  CHECK(matcher.classify("period").character == ".");
  CHECK(matcher.classify("comma").character == ",");
  CHECK(matcher.classify("colon").character == ":");
  CHECK(matcher.classify("semicolon").character == ";");
  CHECK(matcher.classify("question mark").character == "?");
  CHECK(matcher.classify("exclamation mark").character == "!");
}

TEST_CASE("spelling-fusion: weak-homonym detection") {
  SpellingMatcher matcher;
  CHECK(matcher.is_weak_homonym("okay"));
  CHECK(matcher.is_weak_homonym("Ok."));
  CHECK(matcher.is_weak_homonym("you"));
  CHECK_FALSE(matcher.is_weak_homonym("alpha"));
  CHECK_FALSE(matcher.is_weak_homonym(""));
}

TEST_CASE("spelling-fusion: fuse without prediction") {
  SpellingMatcher matcher;
  // No prediction → matcher's character passes through.
  FusedResult r = fuse_default("alpha", char_match("a"), nullptr, matcher);
  CHECK(r.is_character());
  CHECK(r.character == "a");
}

TEST_CASE("spelling-fusion: fuse drops unrecognized + no prediction") {
  SpellingMatcher matcher;
  FusedResult r = fuse_default("nonsense", no_match(), nullptr, matcher);
  CHECK_FALSE(r.is_character());
  CHECK(r.type == SpellingMatchType::NONE);
}

TEST_CASE("spelling-fusion: fuse passes through command words") {
  SpellingMatcher matcher;
  SpellingPrediction pred = {"a", 0.99f, "a"};
  SpellingMatch stop;
  stop.type = SpellingMatchType::STOPPED;
  FusedResult r = fuse_default("stop", stop, &pred, matcher);
  CHECK(r.type == SpellingMatchType::STOPPED);
  CHECK(r.character.empty());
}

TEST_CASE(
    "spelling-fusion: special-character match is preserved when the "
    "spelling model is not confident (dollar-sign regression)") {
  SpellingMatcher matcher;

  // Below the same-class threshold (0.5) the matcher's "$" wins. The
  // spelling-CNN has no special-character class, so on real audio the
  // top-1 probability for utterances like "dollar sign" is empirically
  // low and this is the branch that fires in production.
  SpellingPrediction weak = {"a", 0.20f, "a"};
  FusedResult r1 =
      fuse_default("dollar sign", char_match("$"), &weak, matcher);
  CHECK(r1.is_character());
  CHECK(r1.character == "$");

  // No prediction at all (spelling model not loaded) → "$" survives.
  FusedResult r2 =
      fuse_default("dollar sign", char_match("$"), nullptr, matcher);
  CHECK(r2.is_character());
  CHECK(r2.character == "$");
}

TEST_CASE("spelling-fusion: weak-homonym demotion") {
  SpellingMatcher matcher;
  // "okay" + confident spelling("k", 0.6) → matcher demoted, but the
  // spelling model agrees, so we still emit "k".
  SpellingPrediction agree = {"k", 0.6f, "k"};
  FusedResult r1 = fuse_default("okay", char_match("k"), &agree, matcher);
  CHECK(r1.character == "k");

  // "okay" + confident spelling("z", 0.7) → matcher demoted, fall back
  // to the spelling model's "z".
  SpellingPrediction disagree = {"z", 0.7f, "z"};
  FusedResult r2 = fuse_default("okay", char_match("k"), &disagree, matcher);
  CHECK(r2.character == "z");

  // "okay" + LOW-confidence spelling("z", 0.1): below the demotion
  // threshold (0.3), so the matcher's "k" is kept.
  SpellingPrediction weak = {"z", 0.1f, "z"};
  FusedResult r3 = fuse_default("okay", char_match("k"), &weak, matcher);
  CHECK(r3.character == "k");
}

TEST_CASE("spelling-fusion: cross-class routing") {
  SpellingMatcher matcher;
  // Matcher: digit, spelling: letter -> trust the matcher.
  SpellingPrediction letter = {"o", 0.95f, "o"};
  FusedResult r1 = fuse_default("five", char_match("5"), &letter, matcher);
  CHECK(r1.character == "5");

  // Matcher: letter, spelling: digit -> trust the spelling model.
  SpellingPrediction digit = {"0", 0.95f, "zero"};
  FusedResult r2 = fuse_default("oh", char_match("o"), &digit, matcher);
  CHECK(r2.character == "0");
}

TEST_CASE("spelling-fusion: same-class disagreement uses threshold") {
  SpellingMatcher matcher;
  // Matcher: "b", spelling: "d" with 0.6 -> spelling wins.
  SpellingPrediction strong = {"d", 0.6f, "d"};
  FusedResult r1 = fuse_default("bee", char_match("b"), &strong, matcher);
  CHECK(r1.character == "d");

  // Matcher: "b", spelling: "d" with 0.4 -> matcher wins.
  SpellingPrediction weak = {"d", 0.4f, "d"};
  FusedResult r2 = fuse_default("bee", char_match("b"), &weak, matcher);
  CHECK(r2.character == "b");
}

TEST_CASE("spelling-fusion: multi-digit ASR vs single-digit spelling") {
  // Regression: the matcher's digit-string fallback returns multi-digit
  // strings like "1944" or "23". Before the fix the predicate
  // ``char_is_digit`` only recognised single digits, so ``"1944"`` was
  // routed through the special-character pass-through and the spelling
  // model's ``"4"`` was discarded. Mirror Python's smart-router and
  // treat any all-digit string as the digit class, then break the tie
  // on the spelling probability.
  SpellingMatcher matcher;
  SpellingPrediction strong = {"4", 0.95f, "four"};
  FusedResult r1 = fuse_default("1944.", char_match("1944"), &strong, matcher);
  CHECK(r1.character == "4");

  SpellingPrediction weak = {"4", 0.10f, "four"};
  FusedResult r2 = fuse_default("1944.", char_match("1944"), &weak, matcher);
  CHECK(r2.character == "1944");

  // Cross-class still works for multi-digit ASR: spelling says letter,
  // ASR says digit-string -> trust ASR.
  SpellingPrediction letter = {"o", 0.95f, "o"};
  FusedResult r3 = fuse_default("23.", char_match("23"), &letter, matcher);
  CHECK(r3.character == "23");
}

TEST_CASE("spelling-fusion: agreement preserves matcher casing") {
  SpellingMatcher matcher;
  // Matcher returned upper-case "B" via the modifier; the spelling
  // model agrees (case-blind). Output stays upper-case.
  SpellingPrediction agree = {"b", 0.99f, "b"};
  FusedResult r =
      fuse_default("capital b", char_match("B"), &agree, matcher);
  CHECK(r.character == "B");
}

TEST_CASE("spelling-fusion: spelling-only when matcher misses") {
  SpellingMatcher matcher;
  // Matcher returned NONE (raw text doesn't resolve), spelling fired —
  // emit the spelling model's character.
  SpellingPrediction pred = {"f", 0.85f, "f"};
  FusedResult r = fuse_default("hmmm", no_match(), &pred, matcher);
  CHECK(r.character == "f");
}

TEST_CASE("spelling-fusion: data tables are non-empty") {
  // Sanity check — guards against a stale build (or a bad rebase
  // accidentally clearing one of the tables).
  CHECK(spelling_fusion_data::lookup_table().size() > 100);
  CHECK(spelling_fusion_data::stop_words().size() >= 5);
  CHECK(spelling_fusion_data::default_weak_homonyms().size() == 3);
  CHECK(spelling_fusion_data::default_meta().classes.size() == 36);
  CHECK(spelling_fusion_data::default_meta().sample_rate == 16000);
  CHECK(spelling_fusion_data::default_meta().clip_seconds ==
        doctest::Approx(1.0f));
}
