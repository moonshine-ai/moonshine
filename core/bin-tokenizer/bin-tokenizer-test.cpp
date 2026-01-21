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
}