#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "hindi-numbers.h"
#include "hindi.h"
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
  const std::filesystem::path hi_dict = wiki.parent_path() / "dict.tsv";
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(moonshine_tts::hindi_text_to_ipa(src[i], true, nullptr, true, hi_dict) == py[i]);
  }
}

}  // namespace

TEST_CASE("hindi: dialect_resolves_to_hindi_rules") {
  using moonshine_tts::dialect_resolves_to_hindi_rules;
  CHECK(dialect_resolves_to_hindi_rules("hi"));
  CHECK(dialect_resolves_to_hindi_rules("hi-IN"));
  CHECK(dialect_resolves_to_hindi_rules("Hindi"));
  CHECK_FALSE(dialect_resolves_to_hindi_rules("uk"));
}

TEST_CASE("hindi: कमल and मैं match reference IPA when golden exists") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path golden = r::tests_data_dir(repo) / "hi" / "rule_g2p_sample.txt";
  if (!std::filesystem::is_regular_file(golden)) {
    return;
  }
  const auto lines = r::load_ref_lines(golden);
  REQUIRE(lines.size() >= 1);
  const std::filesystem::path hi_dict = r::moonshine_tts_bundled_data_dir_relative() / "hi" / "dict.tsv";
  CHECK(moonshine_tts::hindi_text_to_ipa("कमल मैं", true, nullptr, true, hi_dict) == lines[0]);
}

TEST_CASE("hindi: expand_cardinal_digits_to_hindi_words") {
  using moonshine_tts::expand_cardinal_digits_to_hindi_words;
  CHECK(expand_cardinal_digits_to_hindi_words("0") == "शून्य");
  CHECK(expand_cardinal_digits_to_hindi_words("42") == "बयालीस");
}

TEST_CASE("hindi: wiki-text first 100 lines match reference IPA when data and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path wiki = r::moonshine_tts_bundled_data_dir_relative() / "hi" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "hi" / "rule_g2p_wiki_100.txt";
  check_wiki_parity(wiki, golden);
}
