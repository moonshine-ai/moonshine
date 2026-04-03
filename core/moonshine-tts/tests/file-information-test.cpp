#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "file-information.h"

#include <vector>

using moonshine_tts::FileInformation;
using moonshine_tts::FileInformationMap;

TEST_CASE("FileInformation default memory fields") {
  FileInformation f{std::filesystem::path{"a/b.tsv"}, nullptr, 0};
  CHECK(f.memory == nullptr);
  CHECK(f.memory_size == 0);
  CHECK(f.path == std::filesystem::path{"a/b.tsv"});
}

TEST_CASE("FileInformationMap set_path and contains") {
  FileInformationMap m;
  CHECK_FALSE(m.contains("k1"));
  m.set_path("k1", std::filesystem::path{"rel/p.tsv"});
  CHECK(m.contains("k1"));
  REQUIRE(m.entries.count("k1") == 1);
  CHECK(m.entries.at("k1").path == std::filesystem::path{"rel/p.tsv"});
  CHECK(m.entries.at("k1").memory == nullptr);
}

TEST_CASE("FileInformationMap erase_key") {
  FileInformationMap m;
  m.set_path("x", "p");
  m.erase_key("x");
  CHECK_FALSE(m.contains("x"));
}

TEST_CASE("FileInformationMap::parse_file_list") {
  std::vector<std::pair<std::string, std::string>> keys{{"asset_a", "sub/a.txt"}, {"asset_b", "b.txt"}};
  std::vector<uint8_t*> ptrs;
  std::vector<size_t> sizes;
  uint8_t blob[] = {1, 2, 3};
  ptrs.push_back(blob);
  sizes.push_back(sizeof(blob));
  ptrs.push_back(nullptr);
  sizes.push_back(0);

  FileInformationMap m;
  m.parse_file_list(&keys, &ptrs, &sizes, std::filesystem::path{"/root"});

  REQUIRE(m.entries.count("asset_a") == 1);
  CHECK(m.entries["asset_a"].path == std::filesystem::path{"/root/sub/a.txt"});
  CHECK(m.entries["asset_a"].memory == blob);
  CHECK(m.entries["asset_a"].memory_size == 3);

  REQUIRE(m.entries.count("asset_b") == 1);
  CHECK(m.entries["asset_b"].path == std::filesystem::path{"/root/b.txt"});
  CHECK(m.entries["asset_b"].memory == nullptr);
  CHECK(m.entries["asset_b"].memory_size == 0);
}

TEST_CASE("FileInformationMap::parse_file_list null key_list throws") {
  FileInformationMap m;
  CHECK_THROWS_AS(m.parse_file_list(nullptr, nullptr, nullptr, "/x"), std::runtime_error);
}

TEST_CASE("FileInformationMap::parse_file_list memory size mismatch throws") {
  std::vector<std::pair<std::string, std::string>> keys{{"a", "a"}};
  std::vector<uint8_t*> ptrs{nullptr};
  std::vector<size_t> sizes{1, 2};
  FileInformationMap m;
  CHECK_THROWS_AS(m.parse_file_list(&keys, &ptrs, &sizes, "/x"), std::runtime_error);
}
