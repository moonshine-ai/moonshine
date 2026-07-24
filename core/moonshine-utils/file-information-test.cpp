#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "file-information.h"

#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Writes bytes to a unique temp file and returns its path. The caller is
// responsible for removing it (tests clean up in a RAII guard below).
std::filesystem::path write_temp_file(const std::string& name,
                                      const std::vector<uint8_t>& bytes) {
  const std::filesystem::path dir =
      std::filesystem::temp_directory_path() / "moonshine_file_info_test";
  std::filesystem::create_directories(dir);
  const std::filesystem::path p = dir / name;
  std::ofstream f(p, std::ios::binary);
  for (uint8_t b : bytes) {
    f.put(static_cast<char>(b));
  }
  f.close();
  return p;
}

struct TempDirGuard {
  ~TempDirGuard() {
    std::error_code ec;
    std::filesystem::remove_all(
        std::filesystem::temp_directory_path() / "moonshine_file_info_test",
        ec);
  }
};

}  // namespace

TEST_CASE("FileInformation default memory fields") {
  FileInformation f{std::filesystem::path{"a/b.tsv"}, nullptr, 0};
  CHECK(f.memory == nullptr);
  CHECK(f.memory_size == 0);
  CHECK_FALSE(f.has_memory());
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

TEST_CASE("FileInformationMap set_memory records buffer and default path") {
  FileInformationMap m;
  uint8_t blob[] = {9, 8, 7, 6};
  m.set_memory("enc", blob, sizeof(blob));
  REQUIRE(m.contains("enc"));
  const FileInformation& fi = m.entries.at("enc");
  CHECK(fi.memory == blob);
  CHECK(fi.memory_size == 4);
  CHECK(fi.has_memory());
  // With no explicit resolve path, the key doubles as the path.
  CHECK(fi.path == std::filesystem::path{"enc"});
}

TEST_CASE("FileInformationMap erase_key") {
  FileInformationMap m;
  m.set_path("x", "p");
  m.erase_key("x");
  CHECK_FALSE(m.contains("x"));
}

TEST_CASE("FileInformation::load returns client buffer without copying") {
  uint8_t blob[] = {1, 2, 3, 4, 5};
  FileInformation fi{std::filesystem::path{"ignored"}, blob, sizeof(blob)};
  const uint8_t* out = nullptr;
  size_t out_size = 0;
  fi.load(&out, &out_size);
  // Client memory is returned as-is (same pointer), never copied.
  CHECK(out == blob);
  CHECK(out_size == 5);
}

TEST_CASE("FileInformation::load reads bytes from disk when no buffer given") {
  TempDirGuard guard;
  const std::vector<uint8_t> bytes = {10, 20, 30, 40, 50, 60};
  const std::filesystem::path p = write_temp_file("disk.bin", bytes);

  FileInformation fi{p, nullptr, 0};
  const uint8_t* out = nullptr;
  size_t out_size = 0;
  fi.load(&out, &out_size);
  REQUIRE(out != nullptr);
  REQUIRE(out_size == bytes.size());
  for (size_t i = 0; i < bytes.size(); ++i) {
    CHECK(out[i] == bytes[i]);
  }
  // Now that it is loaded, it reports having memory, and a second load returns
  // the same owned pointer (idempotent, no re-read).
  CHECK(fi.has_memory());
  const uint8_t* out2 = nullptr;
  size_t out_size2 = 0;
  fi.load(&out2, &out_size2);
  CHECK(out2 == out);
  CHECK(out_size2 == out_size);
}

TEST_CASE("FileInformation copy keeps disk-loaded bytes alive independently") {
  TempDirGuard guard;
  const std::vector<uint8_t> bytes = {7, 7, 7, 42};
  const std::filesystem::path p = write_temp_file("copy.bin", bytes);

  FileInformation src{p, nullptr, 0};
  const uint8_t* src_out = nullptr;
  size_t src_size = 0;
  src.load(&src_out, &src_size);

  // Copy after load: the copy owns its own storage and points at it.
  FileInformation copy = src;
  const uint8_t* copy_out = nullptr;
  size_t copy_size = 0;
  copy.load(&copy_out, &copy_size);
  REQUIRE(copy_size == bytes.size());
  CHECK(copy_out != src_out);  // distinct owned buffers
  for (size_t i = 0; i < bytes.size(); ++i) {
    CHECK(copy_out[i] == bytes[i]);
  }

  // Dropping the source's bytes must not disturb the copy.
  src.free();
  CHECK_FALSE(src.has_memory());
  const uint8_t* copy_out2 = nullptr;
  size_t copy_size2 = 0;
  copy.load(&copy_out2, &copy_size2);
  CHECK(copy_size2 == bytes.size());
}

TEST_CASE("FileInformation::free drops disk bytes but not client buffers") {
  uint8_t blob[] = {1, 2, 3};
  FileInformation client{std::filesystem::path{"k"}, blob, sizeof(blob)};
  client.free();
  // Client-owned buffer is left intact by free().
  CHECK(client.memory == blob);
  CHECK(client.memory_size == 3);
}

TEST_CASE("FileInformation::load throws for empty path and no buffer") {
  FileInformation fi;
  const uint8_t* out = nullptr;
  size_t out_size = 0;
  CHECK_THROWS_AS(fi.load(&out, &out_size), std::runtime_error);
}

TEST_CASE("FileInformation::load throws for missing file") {
  FileInformation fi{std::filesystem::path{"/no/such/moonshine/file.bin"},
                     nullptr, 0};
  const uint8_t* out = nullptr;
  size_t out_size = 0;
  CHECK_THROWS_AS(fi.load(&out, &out_size), std::runtime_error);
}

TEST_CASE("FileInformationMap::load resolves by key and reports missing keys") {
  uint8_t blob[] = {5, 6};
  FileInformationMap m;
  m.set_memory("present", blob, sizeof(blob));

  const uint8_t* out = nullptr;
  size_t out_size = 0;
  m.load("present", &out, &out_size);
  CHECK(out == blob);
  CHECK(out_size == 2);

  CHECK_THROWS_AS(m.load("absent", &out, &out_size), std::runtime_error);
}

TEST_CASE(
    "FileInformationMap mixes memory and path entries (buffer-requirement "
    "variants)") {
  TempDirGuard guard;
  // Mirrors a real loader: some assets supplied in memory, others left on disk.
  const std::vector<uint8_t> disk_bytes = {100, 101, 102};
  const std::filesystem::path disk_path =
      write_temp_file("tokenizer.bin", disk_bytes);
  uint8_t encoder_blob[] = {1, 1, 2, 3, 5, 8};

  FileInformationMap m;
  m.set_memory("encoder_model.ort", encoder_blob, sizeof(encoder_blob));
  m.set_path("tokenizer.bin", disk_path);

  const uint8_t* enc = nullptr;
  size_t enc_size = 0;
  m.load("encoder_model.ort", &enc, &enc_size);
  CHECK(enc == encoder_blob);
  CHECK(enc_size == 6);

  const uint8_t* tok = nullptr;
  size_t tok_size = 0;
  m.load("tokenizer.bin", &tok, &tok_size);
  REQUIRE(tok_size == disk_bytes.size());
  for (size_t i = 0; i < disk_bytes.size(); ++i) {
    CHECK(tok[i] == disk_bytes[i]);
  }
}

TEST_CASE("FileInformationMap::parse_file_list") {
  std::vector<std::pair<std::string, std::string>> keys{
      {"asset_a", "sub/a.txt"}, {"asset_b", "b.txt"}};
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
  CHECK_THROWS_AS(m.parse_file_list(nullptr, nullptr, nullptr, "/x"),
                  std::runtime_error);
}

TEST_CASE("FileInformationMap::parse_file_list memory size mismatch throws") {
  std::vector<std::pair<std::string, std::string>> keys{{"a", "a"}};
  std::vector<uint8_t*> ptrs{nullptr};
  std::vector<size_t> sizes{1, 2};
  FileInformationMap m;
  CHECK_THROWS_AS(m.parse_file_list(&keys, &ptrs, &sizes, "/x"),
                  std::runtime_error);
}
