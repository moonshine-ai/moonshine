#include "file-utils.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {
// Writes `bytes` to `path` for the read-back tests below.
void write_file(const char *path, const std::vector<uint8_t> &bytes) {
  std::FILE *file = std::fopen(path, "wb");
  REQUIRE(file != nullptr);
  if (!bytes.empty()) {
    const size_t written = std::fwrite(bytes.data(), 1, bytes.size(), file);
    REQUIRE(written == bytes.size());
  }
  std::fclose(file);
}
}  // namespace

TEST_CASE("fread_exact") {
  const char *path = "file-utils-test.bin";

  SUBCASE("reads the full requested amount") {
    const std::vector<uint8_t> contents = {1, 2, 3, 4, 5, 6, 7, 8};
    write_file(path, contents);
    std::FILE *file = std::fopen(path, "rb");
    REQUIRE(file != nullptr);

    std::vector<uint8_t> buffer(contents.size());
    const size_t count =
        fread_exact(buffer.data(), 1, buffer.size(), file, "byte buffer");
    CHECK(count == contents.size());
    CHECK(buffer == contents);
    std::fclose(file);
  }

  SUBCASE("reads multi-byte elements and preserves values") {
    const uint32_t values[2] = {0x01020304u, 0x05060708u};
    std::vector<uint8_t> contents(sizeof(values));
    std::memcpy(contents.data(), values, sizeof(values));
    write_file(path, contents);
    std::FILE *file = std::fopen(path, "rb");
    REQUIRE(file != nullptr);

    uint32_t read_values[2] = {0, 0};
    const size_t count =
        fread_exact(read_values, sizeof(uint32_t), 2, file, "uint32 pair");
    CHECK(count == 2);
    CHECK(read_values[0] == values[0]);
    CHECK(read_values[1] == values[1]);
    std::fclose(file);
  }

  SUBCASE("throws when fewer elements are available than requested") {
    const std::vector<uint8_t> contents = {1, 2, 3};
    write_file(path, contents);
    std::FILE *file = std::fopen(path, "rb");
    REQUIRE(file != nullptr);

    std::vector<uint8_t> buffer(8);
    CHECK_THROWS_AS(
        fread_exact(buffer.data(), 1, buffer.size(), file, "byte buffer"),
        std::runtime_error);
    std::fclose(file);
  }

  SUBCASE("throws on a partial trailing element") {
    // Three bytes cannot satisfy a request for two 2-byte elements.
    const std::vector<uint8_t> contents = {0xAA, 0xBB, 0xCC};
    write_file(path, contents);
    std::FILE *file = std::fopen(path, "rb");
    REQUIRE(file != nullptr);

    uint16_t words[2] = {0, 0};
    CHECK_THROWS_AS(fread_exact(words, sizeof(uint16_t), 2, file, "word pair"),
                    std::runtime_error);
    std::fclose(file);
  }

  SUBCASE("throws when reading past end of file") {
    const std::vector<uint8_t> contents = {42};
    write_file(path, contents);
    std::FILE *file = std::fopen(path, "rb");
    REQUIRE(file != nullptr);

    uint8_t byte = 0;
    CHECK(fread_exact(&byte, 1, 1, file, "first byte") == 1);
    CHECK(byte == 42);
    // Nothing left to read, so the next element must throw.
    CHECK_THROWS_AS(fread_exact(&byte, 1, 1, file, "second byte"),
                    std::runtime_error);
    std::fclose(file);
  }

  SUBCASE("zero count is a no-op that returns count") {
    const std::vector<uint8_t> contents = {1, 2, 3};
    write_file(path, contents);
    std::FILE *file = std::fopen(path, "rb");
    REQUIRE(file != nullptr);
    CHECK(fread_exact(nullptr, 1, 0, file, "nothing") == 0);
    std::fclose(file);
  }

  SUBCASE("zero size is a no-op that returns count") {
    const std::vector<uint8_t> contents = {1, 2, 3};
    write_file(path, contents);
    std::FILE *file = std::fopen(path, "rb");
    REQUIRE(file != nullptr);
    CHECK(fread_exact(nullptr, 0, 4, file, "nothing") == 4);
    std::fclose(file);
  }

  std::remove(path);
}
