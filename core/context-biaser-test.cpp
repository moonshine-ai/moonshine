#include "context-biaser.h"

#include <cmath>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

// Stand-in vocabulary size; the biaser only uses it to reject out-of-range
// token IDs.
const int kVocabSize = 64;

std::vector<float> zero_logits() {
  return std::vector<float>(kVocabSize, 0.0f);
}

}  // namespace

TEST_CASE("context-biaser") {
  SUBCASE("empty-biaser-changes-nothing") {
    ContextBiaser biaser;
    CHECK(biaser.empty());
    std::vector<float> logits = zero_logits();
    biaser.apply(logits.data(), kVocabSize);
    for (const float logit : logits) {
      CHECK(logit == 0.0f);
    }
  }

  SUBCASE("zero-boost-adds-nothing") {
    // The guarantee that a zero boost is an exact no-op lives here rather than
    // in an end-to-end test, because transcription output is not bit-stable
    // enough across process states to compare text byte-for-byte.
    ContextBiaser biaser;
    biaser.set_boost(0.0f);
    biaser.add_token_sequence({10, 11, 12});

    std::vector<float> logits = zero_logits();
    biaser.apply(logits.data(), kVocabSize);
    biaser.advance(10);
    biaser.apply(logits.data(), kVocabSize);
    biaser.advance(11);
    biaser.apply(logits.data(), kVocabSize);
    for (const float logit : logits) {
      CHECK(logit == 0.0f);
    }
  }

  SUBCASE("boosts-only-the-first-token-of-a-term") {
    ContextBiaser biaser;
    biaser.set_boost(2.0f);
    biaser.add_token_sequence({10, 11, 12});
    CHECK_FALSE(biaser.empty());

    std::vector<float> logits = zero_logits();
    biaser.apply(logits.data(), kVocabSize);
    // Only the entry token is a candidate before anything has been emitted.
    CHECK(logits.at(10) == doctest::Approx(2.0f));
    CHECK(logits.at(11) == 0.0f);
    CHECK(logits.at(12) == 0.0f);
  }

  SUBCASE("bonus-grows-with-depth-along-the-path") {
    ContextBiaser biaser;
    biaser.set_boost(2.0f);
    biaser.add_token_sequence({10, 11, 12});

    CHECK(biaser.bonus_for_token(10) == doctest::Approx(2.0f));
    biaser.advance(10);
    // Depth 2: boost * (1 + ln 2).
    CHECK(biaser.bonus_for_token(11) ==
          doctest::Approx(2.0f * (1.0f + std::log(2.0f))));
    biaser.advance(11);
    CHECK(biaser.bonus_for_token(12) ==
          doctest::Approx(2.0f * (1.0f + std::log(3.0f))));
    // The completion is rewarded more than the entry, which is what keeps
    // greedy decoding from firing on a bare prefix.
    CHECK(biaser.bonus_for_token(12) > biaser.bonus_for_token(10));
  }

  SUBCASE("abandoning-a-path-drops-back-to-the-root") {
    ContextBiaser biaser;
    biaser.add_token_sequence({10, 11, 12});

    biaser.advance(10);
    CHECK(biaser.bonus_for_token(11) > 0.0f);
    // A token that is not part of the term ends the partial match.
    biaser.advance(42);
    CHECK(biaser.bonus_for_token(11) == 0.0f);
    // ...but the term can still start again straight away.
    CHECK(biaser.bonus_for_token(10) > 0.0f);
  }

  SUBCASE("a-term-can-start-part-way-through-another") {
    ContextBiaser biaser;
    biaser.add_token_sequence({10, 11});
    biaser.add_token_sequence({11, 20});

    biaser.advance(10);
    biaser.advance(11);
    // Matching "10 11" fully should not stop "11 20" from being live.
    CHECK(biaser.bonus_for_token(20) > 0.0f);
  }

  SUBCASE("shared-prefixes-are-not-stacked") {
    ContextBiaser biaser;
    biaser.set_boost(3.0f);
    biaser.add_token_sequence({10, 11});
    biaser.add_token_sequence({10, 12});

    std::vector<float> logits = zero_logits();
    biaser.apply(logits.data(), kVocabSize);
    // Both terms start with token 10, but it is boosted once.
    CHECK(logits.at(10) == doctest::Approx(3.0f));
  }

  SUBCASE("a-token-two-paths-propose-takes-the-larger-bonus") {
    ContextBiaser biaser;
    biaser.set_boost(3.0f);
    biaser.add_token_sequence({10, 11});
    biaser.add_token_sequence({11, 20});

    // After token 10 both the root and the node inside "10 11" are live, and
    // both offer token 11: the root as the start of "11 20" (depth 1), the
    // deeper node as the completion of "10 11" (depth 2).
    biaser.advance(10);
    std::vector<float> logits = zero_logits();
    biaser.apply(logits.data(), kVocabSize);
    CHECK(logits.at(11) == doctest::Approx(3.0f * (1.0f + std::log(2.0f))));
    CHECK(logits.at(10) == doctest::Approx(3.0f));
  }

  SUBCASE("reset-discards-a-partial-match") {
    ContextBiaser biaser;
    biaser.add_token_sequence({10, 11});

    biaser.advance(10);
    CHECK(biaser.bonus_for_token(11) > 0.0f);
    biaser.reset();
    CHECK(biaser.bonus_for_token(11) == 0.0f);
    CHECK(biaser.bonus_for_token(10) > 0.0f);
  }

  SUBCASE("out-of-range-tokens-are-ignored") {
    ContextBiaser biaser;
    biaser.add_token_sequence({kVocabSize + 5});
    std::vector<float> logits = zero_logits();
    // Must not write past the end of the logits buffer.
    biaser.apply(logits.data(), kVocabSize);
    for (const float logit : logits) {
      CHECK(logit == 0.0f);
    }
  }

  SUBCASE("clear-removes-every-term") {
    ContextBiaser biaser;
    biaser.add_token_sequence({10, 11});
    CHECK_FALSE(biaser.empty());
    biaser.clear();
    CHECK(biaser.empty());
    CHECK(biaser.bonus_for_token(10) == 0.0f);
  }

  SUBCASE("empty-sequences-are-not-registered") {
    ContextBiaser biaser;
    biaser.add_token_sequence({});
    CHECK(biaser.empty());
  }

  SUBCASE("variants-cover-mid-sentence-and-initial-forms") {
    const std::vector<std::string> variants =
        ContextBiaser::variants_for_term("Kubernetes");
    REQUIRE(variants.size() == 2);
    CHECK(variants.at(0) == "Kubernetes");
    CHECK(variants.at(1) == " Kubernetes");

    // Surrounding whitespace is not meaningful.
    CHECK(ContextBiaser::variants_for_term("  Kubernetes  ").at(0) ==
          "Kubernetes");
    CHECK(ContextBiaser::variants_for_term("   ").empty());
    // A caller that anchors the term itself gets exactly what they asked for.
    const std::vector<std::string> anchored =
        ContextBiaser::variants_for_term("\xe2\x96\x81Kubernetes");
    REQUIRE(anchored.size() == 1);
  }
}
