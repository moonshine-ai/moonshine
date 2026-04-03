#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "english.h"
#include "rule-g2p-test-support.h"

#include <filesystem>
#include <string>
#include <vector>

namespace r = moonshine_tts::rule_g2p_test;

namespace {

std::filesystem::path resolve_en_dict() {
  return r::moonshine_tts_bundled_data_dir_relative() / "en_us" / "dict_filtered_heteronyms.tsv";
}

std::filesystem::path en_homograph_json() {
  return r::moonshine_tts_bundled_data_dir_relative() / "en_us" / "heteronym" / "homograph_index.json";
}

}  // namespace

TEST_CASE("english: dialect_resolves_to_english_rules") {
  using moonshine_tts::dialect_resolves_to_english_rules;
  CHECK(dialect_resolves_to_english_rules("en_us"));
  CHECK(dialect_resolves_to_english_rules("en-US"));
  CHECK(dialect_resolves_to_english_rules("EN_US"));
  CHECK(dialect_resolves_to_english_rules("english"));
  CHECK_FALSE(dialect_resolves_to_english_rules("de"));
}

TEST_CASE("english: wiki-text first 100 lines match reference IPA when data and golden exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = resolve_en_dict();
  const std::filesystem::path wiki = r::moonshine_tts_bundled_data_dir_relative() / "en_us" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "en_us" / "rule_g2p_wiki_100.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  const std::filesystem::path homograph = en_homograph_json();
  moonshine_tts::EnglishRuleG2p g(dict, homograph, std::nullopt, std::nullopt);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
