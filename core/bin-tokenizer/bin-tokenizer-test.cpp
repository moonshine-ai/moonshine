#include "bin-tokenizer.h"

#include <cstdio>
#include <filesystem>

#include "debug-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

// Builds a tokenizer.bin blob laid out the way the shipped ones are: the
// control tokens, then the 256 single-byte fallbacks in order, then the learned
// pieces. Byte-pair encoding needs that byte block to exist, and needs the
// pieces to come after it.
struct BpeVocabulary {
  std::vector<uint8_t> data;

  BpeVocabulary() {
    for (const char *control : {"<unk>", "<s>", "</s>"}) {
      this->add_piece(control);
    }
    for (int value = 0; value < 256; value++) {
      this->add_piece(std::string(1, static_cast<char>(value)));
    }
  }

  // Appends an entry and returns its id. Entries here are always shorter than
  // 128 bytes, so the length is always a single prefix byte.
  int32_t add_piece(const std::string &spelling) {
    this->data.push_back(static_cast<uint8_t>(spelling.size()));
    this->data.insert(this->data.end(), spelling.begin(), spelling.end());
    return this->next_id++;
  }

  // The fallback entry for a raw byte. The block starts at id 3, after the
  // three control tokens.
  int32_t byte_id(int value) const { return 3 + value; }

 private:
  int32_t next_id = 0;
};

}  // namespace

TEST_CASE("bin-tokenizer") {
  SUBCASE("constructor-from-path") {
    std::vector<uint8_t> data = {0, 2, 2, 3, 4, 1, 2, 3, 4};
    save_memory_to_file("tokenizer.bin", data);
    REQUIRE(std::filesystem::exists("tokenizer.bin"));

    BinTokenizer tokenizer("tokenizer.bin");
    CHECK(tokenizer.tokens_to_bytes.size() == 3);
    CHECK(tokenizer.tokens_to_bytes[0].size() == 0);
    CHECK(tokenizer.tokens_to_bytes[1].size() == 2);
    CHECK(tokenizer.tokens_to_bytes[1] == std::vector<uint8_t>({2, 3}));
    CHECK(tokenizer.tokens_to_bytes[2].size() == 4);
    CHECK(tokenizer.tokens_to_bytes[2] == std::vector<uint8_t>({1, 2, 3, 4}));
    std::remove("tokenizer.bin");
  }
  SUBCASE("constructor-from-data") {
    std::vector<uint8_t> data = {0, 2, 2, 3, 4, 1, 2, 3, 4};
    BinTokenizer tokenizer(data.data(), data.size());
    CHECK(tokenizer.tokens_to_bytes.size() == 3);
    CHECK(tokenizer.tokens_to_bytes[0].size() == 0);
    CHECK(tokenizer.tokens_to_bytes[1].size() == 2);
    CHECK(tokenizer.tokens_to_bytes[1] == std::vector<uint8_t>({2, 3}));
    CHECK(tokenizer.tokens_to_bytes[2].size() == 4);
    CHECK(tokenizer.tokens_to_bytes[2] == std::vector<uint8_t>({1, 2, 3, 4}));
  }
  SUBCASE("text-to-tokens-takes-the-longest-match") {
    // Vocabulary: 0 empty, 1 "a", 2 "ab", 3 "b", 4 "abc", 5 "ab" again. The
    // duplicate is there to pin the tie-break between two equally long matches.
    std::vector<uint8_t> data = {0, 1,   'a', 2,   'a', 'b', 1,  'b',
                                 3, 'a', 'b', 'c', 2,   'a', 'b'};
    BinTokenizer tokenizer(data.data(), data.size());
    REQUIRE(tokenizer.tokens_to_bytes.size() == 6);

    CHECK(tokenizer.text_to_tokens<int32_t>("abc") ==
          std::vector<int32_t>({4}));
    // Longest wins over the two single-byte tokens, and the lower id wins over
    // the identical entry at index 5.
    CHECK(tokenizer.text_to_tokens<int32_t>("ab") == std::vector<int32_t>({2}));
    CHECK(tokenizer.text_to_tokens<int32_t>("aba") ==
          std::vector<int32_t>({2, 1}));
    CHECK(tokenizer.text_to_tokens<int32_t>("ba") ==
          std::vector<int32_t>({3, 1}));
    CHECK(tokenizer.text_to_tokens<int64_t>("abc") ==
          std::vector<int64_t>({4}));
    CHECK_THROWS(tokenizer.text_to_tokens<int32_t>("z"));
  }
  SUBCASE("bpe-encoding-replays-the-merges-in-id-order") {
    // A vocabulary shaped like the ones the models ship with: three control
    // tokens, then the 256 byte fallbacks, then the learned pieces in merge
    // order. "cd" is learned before "ab", and "abcd" was never learned at all,
    // which is the shape that makes the two encodings disagree.
    BpeVocabulary vocabulary;
    const int32_t cd = vocabulary.add_piece("cd");
    const int32_t ab = vocabulary.add_piece("ab");
    const int32_t abc = vocabulary.add_piece("abc");
    BinTokenizer tokenizer(vocabulary.data.data(), vocabulary.data.size(), "_",
                           BinTokenizerEncoding::kBpe);
    REQUIRE(tokenizer.encoding_for_test() == BinTokenizerEncoding::kBpe);

    // "cd" is the earliest merge, so it forms before "ab" is even considered,
    // and "abc" never forms because "c" is spoken for by the time "ab" exists.
    CHECK(tokenizer.text_to_tokens<int32_t>("abcd") ==
          std::vector<int32_t>({ab, cd}));
    CHECK(tokenizer.text_to_tokens<int64_t>("abcd") ==
          std::vector<int64_t>({ab, cd}));
    // The same vocabulary read as longest-match spells it differently, which is
    // the whole reason the encoding has to be chosen by the vocabulary.
    BinTokenizer longest_match(vocabulary.data.data(), vocabulary.data.size(),
                               "_");
    CHECK(longest_match.text_to_tokens<int32_t>("abcd") ==
          std::vector<int32_t>({abc, vocabulary.byte_id('d')}));

    // With nothing to compete for the "c", merging does run all the way up.
    CHECK(tokenizer.text_to_tokens<int32_t>("abc") ==
          std::vector<int32_t>({abc}));
    CHECK(tokenizer.text_to_tokens<int32_t>("ab") ==
          std::vector<int32_t>({ab}));
    CHECK(tokenizer.text_to_tokens<int32_t>("cd") ==
          std::vector<int32_t>({cd}));
    // Characters no merge can spell fall back to their raw bytes rather than
    // throwing the way longest-match encoding does.
    CHECK(tokenizer.text_to_tokens<int32_t>("z") ==
          std::vector<int32_t>({vocabulary.byte_id('z')}));
    // A multi-byte character with no piece of its own becomes one token per
    // byte: U+00E9 is 0xC3 0xA9.
    CHECK(tokenizer.text_to_tokens<int32_t>("é") ==
          std::vector<int32_t>(
              {vocabulary.byte_id(0xC3), vocabulary.byte_id(0xA9)}));
    // Spaces become the marker before anything merges, so a word start is just
    // another character as far as the merges are concerned.
    CHECK(tokenizer.text_to_tokens<int32_t>(" ab") ==
          std::vector<int32_t>({vocabulary.byte_id('_'), ab}));
    // Control tokens sit before the byte block and outside the merge index, so
    // encoding never produces one, but they can still be looked up whole.
    CHECK(tokenizer.text_to_special_token<int32_t>("</s>") == 2);
    CHECK(tokenizer.tokens_to_text<int32_t>({ab, cd}, false) == "abcd");
  }
  SUBCASE("bpe-falls-back-when-there-is-no-byte-block") {
    std::vector<uint8_t> data = {0, 1, 'a', 2, 'a', 'b'};
    BinTokenizer tokenizer(data.data(), data.size(), "_",
                           BinTokenizerEncoding::kBpe);
    CHECK(tokenizer.encoding_for_test() == BinTokenizerEncoding::kLongestMatch);
    CHECK(tokenizer.text_to_tokens<int32_t>("ab") == std::vector<int32_t>({2}));
  }
}