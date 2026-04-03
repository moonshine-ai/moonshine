#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "cmudict-tsv.h"

#include <filesystem>
#include <fstream>

using namespace moonshine_tts;

TEST_CASE("cmudict-tsv load and lookup") {
  const auto dir = std::filesystem::temp_directory_path();
  const auto path = dir / "moonshine_tts_test_dict.tsv";
  {
    std::ofstream out(path);
    out << "hello\thəˈloʊ\n";
    out << "hello\thəˈləʊ\n";
    out << "# comment\n";
  }
  CmudictTsv dict(path);
  const auto* alts = dict.lookup("hello");
  REQUIRE(alts != nullptr);
  REQUIRE(alts->size() == 2);
  CHECK((*alts)[0] < (*alts)[1]);
  std::filesystem::remove(path);
}
