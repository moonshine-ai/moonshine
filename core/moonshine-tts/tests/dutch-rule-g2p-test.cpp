#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "dutch.h"
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
      std::filesystem::temp_directory_path() / ("moonshine_dutch_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

}  // namespace

TEST_CASE("dutch: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "Licht\twrong\n"
      "licht\tl\xC9\xAAxt\n");
  moonshine_tts::DutchRuleG2p g(p);
  CHECK(g.word_to_ipa("licht") == "l\xC9\xAAxt");
  std::filesystem::remove(p);
}

TEST_CASE("dutch: lexicon stress not shifted by vocoder (unlike German policy)") {
  static const std::string kPri{"\xCB\x88"};
  const auto p = make_temp_tsv("komma\tk\xC9\xAA\xCB\x88m\xC9\x91\n");  // komma\tkɪˈmɑ (example shape)
  moonshine_tts::DutchRuleG2p g(p);
  const std::string ipa = g.word_to_ipa("komma");
  CHECK(ipa.find(kPri) != std::string::npos);
  CHECK(ipa == "kɪˈmɑ");
  std::filesystem::remove(p);
}

TEST_CASE("dutch: rule IPA gets vocoder stress when enabled") {
  const auto p = make_temp_tsv("x\ty\n");
  moonshine_tts::DutchRuleG2p g(p);
  const std::string ipa = g.word_to_ipa("fiets");
  static const std::string kPri{"\xCB\x88"};
  CHECK(ipa.find(kPri) != std::string::npos);
  std::filesystem::remove(p);
}

TEST_CASE("dutch: normalize_ipa_stress_for_vocoder idempotent") {
  using moonshine_tts::DutchRuleG2p;
  // Same edge case as Python: coda-only tail after ˈ moves mark to end (fɪˈts -> fɪtsˈ).
  std::string a = DutchRuleG2p::normalize_ipa_stress_for_vocoder("f\xC9\xAA\xCB\x88ts");
  std::string b = DutchRuleG2p::normalize_ipa_stress_for_vocoder(a);
  CHECK(a == b);
  CHECK(a == "f\xC9\xAAts\xCB\x88");
}

TEST_CASE("dutch: dialect_resolves_to_dutch_rules") {
  using moonshine_tts::dialect_resolves_to_dutch_rules;
  CHECK(dialect_resolves_to_dutch_rules("nl"));
  CHECK(dialect_resolves_to_dutch_rules("nl-NL"));
  CHECK(dialect_resolves_to_dutch_rules("NL_nl"));
  CHECK(dialect_resolves_to_dutch_rules("dutch"));
  CHECK_FALSE(dialect_resolves_to_dutch_rules("de"));
}

TEST_CASE("dutch: optional real dict fiets matches Python when data present") {
  const std::filesystem::path dict = r::repo_root_from_tests_cpp(__FILE__) / "data" / "nl" / "dict.tsv";
  if (!std::filesystem::is_regular_file(dict)) {
    return;
  }
  moonshine_tts::DutchRuleG2p g(dict);
  CHECK(g.word_to_ipa("fiets") == "\xCB\x88" "fi" "\xCB\x90" "ts");
}

TEST_CASE("dutch: wiki-text first 100 lines match reference IPA when data and golden exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = repo / "data" / "nl" / "dict.tsv";
  const std::filesystem::path wiki = repo / "data" / "nl" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "nl" / "rule_g2p_wiki_100.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::DutchRuleG2p g(dict);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
