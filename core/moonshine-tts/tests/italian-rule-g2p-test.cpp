#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "italian.h"
#include "rule-g2p-test-support.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace r = moonshine_tts::rule_g2p_test;

namespace {

std::filesystem::path make_temp_tsv(const char* contents) {
  const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
  std::filesystem::path p =
      std::filesystem::temp_directory_path() / ("moonshine_italian_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

}  // namespace

TEST_CASE("italian: dialect_resolves_to_italian_rules") {
  using moonshine_tts::dialect_resolves_to_italian_rules;
  CHECK(dialect_resolves_to_italian_rules("it"));
  CHECK(dialect_resolves_to_italian_rules("it-IT"));
  CHECK(dialect_resolves_to_italian_rules("IT_it"));
  CHECK(dialect_resolves_to_italian_rules("italian"));
  CHECK_FALSE(dialect_resolves_to_italian_rules("de"));
}

TEST_CASE("italian: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "Roma\twrong\n"
      "roma\tright\n");
  moonshine_tts::ItalianRuleG2p g(p);
  CHECK(g.word_to_ipa("roma") == "right");
  std::filesystem::remove(p);
}

TEST_CASE("italian: lexicon stress not shifted by vocoder") {
  static const std::string kPri{"\xCB\x88"};
  const auto p = make_temp_tsv("testlex\tt\xC9\xAA\xCB\x88st\n");
  moonshine_tts::ItalianRuleG2p g(p);
  const std::string ipa = g.word_to_ipa("testlex");
  CHECK(ipa.find(kPri) != std::string::npos);
  std::filesystem::remove(p);
}

TEST_CASE("italian: c'è matches reference IPA when data and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "it" / "dict.tsv";
  const std::filesystem::path g_ascii = r::tests_data_dir(repo) / "it" / "rule_g2p_ce_ascii.txt";
  const std::filesystem::path g_curly = r::tests_data_dir(repo) / "it" / "rule_g2p_ce_curly.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(g_ascii) ||
      !std::filesystem::is_regular_file(g_curly)) {
    return;
  }
  moonshine_tts::ItalianRuleG2p g(dict);
  CHECK(g.text_to_ipa("c'è") == r::load_ref_text_trimmed(g_ascii));
  CHECK(g.text_to_ipa("c\u2019\u00e8") == r::load_ref_text_trimmed(g_curly));
}

TEST_CASE("italian: wiki-text first 100 lines match reference IPA when data and golden exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "it" / "dict.tsv";
  const std::filesystem::path wiki = repo / "data" / "it" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "it" / "rule_g2p_wiki_100.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::ItalianRuleG2p g(dict);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
