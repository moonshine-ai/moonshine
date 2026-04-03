#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "turkish.h"
#include "rule-g2p-test-support.h"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace r = moonshine_tts::rule_g2p_test;

namespace {

void check_wiki_parity(const std::filesystem::path& wiki, const std::filesystem::path& golden) {
  constexpr std::size_t kWikiParityLines = 100;
  if (!std::filesystem::is_regular_file(wiki) || !std::filesystem::is_regular_file(golden)) {
    return;
  }
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(moonshine_tts::turkish_text_to_ipa(src[i]) == py[i]);
  }
}

}  // namespace

TEST_CASE("turkish: dağ and değer match reference IPA when golden exists") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path golden = r::tests_data_dir(repo) / "tr" / "rule_g2p_sample.txt";
  if (!std::filesystem::is_regular_file(golden)) {
    return;
  }
  const auto lines = r::load_ref_lines(golden);
  REQUIRE(lines.size() >= 1);
  CHECK(moonshine_tts::turkish_text_to_ipa("dağ değer") == lines[0]);
}

TEST_CASE("turkish: dialect ids include tr and tr-TR") {
  const auto ids = moonshine_tts::TurkishRuleG2p::dialect_ids();
  CHECK(std::find(ids.begin(), ids.end(), "tr") != ids.end());
  CHECK(std::find(ids.begin(), ids.end(), "tr-TR") != ids.end());
}

TEST_CASE("turkish: wiki-text first 100 lines match reference IPA when data and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path wiki = r::moonshine_tts_bundled_data_dir_relative() / "tr" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "tr" / "rule_g2p_wiki_100.txt";
  check_wiki_parity(wiki, golden);
}
