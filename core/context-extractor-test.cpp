#include "context-extractor.h"

#include <algorithm>
#include <set>
#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include "string-utils.h"

namespace {

// Words a stand-in frequency-ordered vocabulary has a token to itself for.
// Everything outside this set is spelled a few characters at a time, which is
// how a real tokenizer behaves on a word it was not built around.
const std::set<std::string> kCommonWords = {
    "the",    "and",    "for",     "with",  "about", "some", "will",
    "have",   "team",   "meeting", "notes", "word",  "here", "chapter",
    "doctor", "prison", "wine",    "shop",  "spy",   "said", "very",
};

// Stands in for MoonshineStreamingModel::text_to_tokens. Takes the leading
// space that ContextExtractor adds, the same as the real tokenizer would.
size_t stub_subword_count(const std::string &word) {
  const std::string bare = to_lowercase(trim(word));
  if (bare.empty()) {
    return 0;
  }
  if (kCommonWords.count(bare) == 1) {
    return 1;
  }
  return (bare.size() + 2) / 3;
}

bool contains(const std::vector<std::string> &terms, const std::string &term) {
  return std::find(terms.begin(), terms.end(), term) != terms.end();
}

size_t position_of(const std::vector<std::string> &terms,
                   const std::string &term) {
  const auto found = std::find(terms.begin(), terms.end(), term);
  REQUIRE(found != terms.end());
  return static_cast<size_t>(found - terms.begin());
}

}  // namespace

TEST_CASE("context-extractor-words") {
  SUBCASE("splits-on-punctuation-and-keeps-case") {
    const std::vector<std::string> words =
        ContextExtractor::candidate_words("The meeting, with Defarge: ready?");
    CHECK(words == std::vector<std::string>{"The", "meeting", "with", "Defarge",
                                            "ready"});
  }

  SUBCASE("keeps-hyphens-and-apostrophes-inside-a-word") {
    const std::vector<std::string> words =
        ContextExtractor::candidate_words("a wine-shop, and Marie's can't");
    CHECK(words ==
          std::vector<std::string>{"wine-shop", "and", "Marie", "can't"});
  }

  SUBCASE("a-dash-between-words-does-not-join-them") {
    // A hyphen only stays inside a word that has already started, so a dash
    // used as punctuation cannot glue two candidates into one.
    const std::vector<std::string> words =
        ContextExtractor::candidate_words("Defarge -- Manette");
    CHECK(words == std::vector<std::string>{"Defarge", "Manette"});
  }

  SUBCASE("folds-the-unicode-punctuation-that-real-prose-carries") {
    // Curly apostrophe and em dash, as any word processor or web page emits
    // them. Untouched, the first would split a word and the second would join
    // two.
    const std::vector<std::string> words = ContextExtractor::candidate_words(
        "Tellson\xe2\x80\x99s bank\xe2\x80\x94Lorry waited");
    CHECK(words ==
          std::vector<std::string>{"Tellson", "bank", "Lorry", "waited"});
  }

  SUBCASE("drops-short-words-and-anything-with-a-digit") {
    // "IPv6" is collected whole and then dropped rather than being handed on
    // as a truncated "IPv", which would bias towards a fragment.
    const std::vector<std::string> words = ContextExtractor::candidate_words(
        "an IPv6 route to 10 hosts in Kubernetes");
    CHECK(words == std::vector<std::string>{"route", "hosts", "Kubernetes"});
  }

  SUBCASE("keeps-non-ascii-letters") {
    const std::vector<std::string> words = ContextExtractor::candidate_words(
        "the Marquis St. \xc3\x89vr\xc3\xa9monde");
    CHECK(contains(words, "Marquis"));
    CHECK(contains(words, "\xc3\x89vr\xc3\xa9monde"));
  }

  SUBCASE("counts-characters-not-bytes-for-the-length-floor") {
    // Three characters of two bytes each: long enough, though a byte count
    // would have let it through for the wrong reason and a shorter one would
    // have been kept.
    const std::vector<std::string> words = ContextExtractor::candidate_words(
        "\xd0\xb4\xd0\xb2\xd0\xb0 \xd0\xb4\xd0\xb2");
    CHECK(words == std::vector<std::string>{"\xd0\xb4\xd0\xb2\xd0\xb0"});
  }

  SUBCASE("strips-possessives") {
    CHECK(ContextExtractor::strip_possessive("Tellson's") == "Tellson");
    CHECK(ContextExtractor::strip_possessive("Jones'") == "Jones");
    CHECK(ContextExtractor::strip_possessive("Defarge") == "Defarge");
    const std::vector<std::string> words =
        ContextExtractor::candidate_words("Tellson's clerk and Jones' desk");
    CHECK(words ==
          std::vector<std::string>{"Tellson", "clerk", "and", "Jones", "desk"});
  }
}

TEST_CASE("context-extractor-extract") {
  SUBCASE("keeps-unusual-words-and-leaves-everyday-ones-alone") {
    const std::vector<std::string> terms = ContextExtractor::extract(
        "The team meeting notes said very little about Kubernetes.", 0,
        stub_subword_count);
    CHECK(contains(terms, "Kubernetes"));
    CHECK_FALSE(contains(terms, "meeting"));
    CHECK_FALSE(contains(terms, "notes"));
    CHECK_FALSE(contains(terms, "The"));
  }

  SUBCASE("ranks-by-how-often-the-passage-says-a-word") {
    const std::vector<std::string> terms = ContextExtractor::extract(
        "Defarge and Defarge and Defarge, with Manette.", 0,
        stub_subword_count);
    CHECK(terms.size() == 2);
    CHECK(position_of(terms, "Defarge") < position_of(terms, "Manette"));
  }

  SUBCASE("breaks-a-tie-on-how-unusual-the-word-is") {
    // Everything here is said once, which is the normal case for a short
    // passage, so the ordering has to come from the tokenizer instead.
    const std::vector<std::string> terms = ContextExtractor::extract(
        "Ceph and glomerulonephritis.", 0, stub_subword_count);
    CHECK(position_of(terms, "glomerulonephritis") <
          position_of(terms, "Ceph"));
  }

  SUBCASE("merges-case-variants-and-asks-for-the-majority-spelling") {
    const std::vector<std::string> terms = ContextExtractor::extract(
        "Madame Defarge, madame Defarge, Madame Defarge, and MADAME.", 0,
        stub_subword_count);
    CHECK(contains(terms, "Madame"));
    CHECK_FALSE(contains(terms, "madame"));
    CHECK_FALSE(contains(terms, "MADAME"));
    // Both spellings counted towards the same term, so it outranks the word
    // said fewer times overall.
    CHECK(position_of(terms, "Madame") < position_of(terms, "Defarge"));
  }

  SUBCASE("honors-the-cap-and-keeps-the-most-said-terms") {
    const std::vector<std::string> terms = ContextExtractor::extract(
        "Defarge Defarge Defarge Manette Manette Cruncher", 2,
        stub_subword_count);
    CHECK(terms == std::vector<std::string>{"Defarge", "Manette"});
  }

  SUBCASE("zero-or-negative-max-terms-asks-for-the-default") {
    // A passage carrying more unusual words than the default cap allows, built
    // out of letters alone since a digit anywhere disqualifies a candidate.
    std::string context;
    for (int first = 'a'; first <= 'z'; first++) {
      for (int second = 'a'; second <= 'z'; second++) {
        context += "zq";
        context += static_cast<char>(first);
        context += static_cast<char>(second);
        context += "vx ";
      }
    }
    const size_t expected =
        static_cast<size_t>(ContextExtractor::kDefaultMaxTerms);
    CHECK(ContextExtractor::extract(context, 0, stub_subword_count).size() ==
          expected);
    CHECK(ContextExtractor::extract(context, -1, stub_subword_count).size() ==
          expected);
  }

  SUBCASE("an-empty-passage-selects-nothing") {
    CHECK(ContextExtractor::extract("", 0, stub_subword_count).empty());
    CHECK(
        ContextExtractor::extract("   ,,,   ", 0, stub_subword_count).empty());
  }

  SUBCASE("a-word-the-tokenizer-cannot-spell-is-skipped") {
    // The real tokenizer throws on bytes it has no token for, which the caller
    // turns into a zero count. Zero must not read as "unusual enough".
    const auto refuses_everything = [](const std::string &) -> size_t {
      return 0;
    };
    CHECK(
        ContextExtractor::extract("Defarge and Manette", 0, refuses_everything)
            .empty());
  }

  SUBCASE("repeats-the-same-choice-for-the-same-passage") {
    // The grouping runs through hash maps, so the ordering has to come from
    // the passage rather than from their iteration order.
    const std::string context =
        "Ceph and etcd and Kubernetes, Ceph and Grafana and Prometheus.";
    const std::vector<std::string> first =
        ContextExtractor::extract(context, 0, stub_subword_count);
    CHECK_FALSE(first.empty());
    for (int attempt = 0; attempt < 5; attempt++) {
      CHECK(ContextExtractor::extract(context, 0, stub_subword_count) == first);
    }
  }
}
