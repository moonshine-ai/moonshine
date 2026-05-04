#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "portuguese.h"
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
      std::filesystem::temp_directory_path() / ("moonshine_pt_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

}  // namespace

TEST_CASE("portuguese: dialect flags") {
  using moonshine_tts::dialect_resolves_to_brazilian_portuguese_rules;
  using moonshine_tts::dialect_resolves_to_portugal_rules;
  CHECK(dialect_resolves_to_brazilian_portuguese_rules("pt-BR"));
  CHECK(dialect_resolves_to_brazilian_portuguese_rules("pt_br"));
  CHECK(dialect_resolves_to_portugal_rules("pt-PT"));
  CHECK(dialect_resolves_to_portugal_rules("pt_pt"));
  CHECK(dialect_resolves_to_portugal_rules("portugal"));
  CHECK_FALSE(dialect_resolves_to_portugal_rules("pt-br"));
}

TEST_CASE("portuguese: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "Rio\twrong\n"
      "rio\tright\n");
  moonshine_tts::PortugueseRuleG2p g(p, false);
  CHECK(g.word_to_ipa("rio") == "right");
  std::filesystem::remove(p);
}

TEST_CASE("portuguese: lexicon stress not shifted by vocoder") {
  static const std::string kPri{"\xCB\x88"};
  const auto p = make_temp_tsv("testlex\tt\xC9\xAA\xCB\x88st\n");
  moonshine_tts::PortugueseRuleG2p g(p, false);
  const std::string ipa = g.word_to_ipa("testlex");
  CHECK(ipa.find(kPri) != std::string::npos);
  std::filesystem::remove(p);
}

TEST_CASE("portuguese: casa matches reference IPA when data and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = r::moonshine_tts_bundled_data_dir_relative() / "pt_br" / "dict.tsv";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "pt_br" / "rule_g2p_casa.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::PortugueseRuleG2p g(dict, false);
  const std::string expected = r::load_ref_text_trimmed(golden);
  CHECK(g.text_to_ipa("casa") == expected);
}

TEST_CASE("portuguese: wiki-text first 100 lines pt_br match reference IPA when data and golden exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = r::moonshine_tts_bundled_data_dir_relative() / "pt_br" / "dict.tsv";
  const std::filesystem::path wiki = r::moonshine_tts_bundled_data_dir_relative() / "pt_br" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "pt_br" / "rule_g2p_wiki_100.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::PortugueseRuleG2p g(dict, false);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("pt_br wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}

TEST_CASE("portuguese: wiki-text first 100 lines pt_pt match reference IPA when data and golden exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = r::moonshine_tts_bundled_data_dir_relative() / "pt_pt" / "dict.tsv";
  const std::filesystem::path wiki = r::moonshine_tts_bundled_data_dir_relative() / "pt_pt" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "pt_pt" / "rule_g2p_wiki_100.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::PortugueseRuleG2p g(dict, true);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("pt_pt wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
