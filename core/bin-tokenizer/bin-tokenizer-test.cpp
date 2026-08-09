#include "bin-tokenizer.h"

#include <cstdio>
#include <filesystem>

#include "debug-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

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
}