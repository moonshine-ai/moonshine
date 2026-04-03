#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "russian.h"
#ifdef MOONSHINE_TTS_WITH_G2P_CLASS
#include "moonshine-g2p.h"
#endif

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
      std::filesystem::temp_directory_path() / ("moonshine_russian_test_" + std::to_string(tick) + ".tsv");
  std::ofstream o(p);
  o << contents;
  return p;
}

}  // namespace

TEST_CASE("russian: dialect_resolves_to_russian_rules") {
  using moonshine_tts::dialect_resolves_to_russian_rules;
  CHECK(dialect_resolves_to_russian_rules("ru"));
  CHECK(dialect_resolves_to_russian_rules("ru-RU"));
  CHECK(dialect_resolves_to_russian_rules("RU_ru"));
  CHECK(dialect_resolves_to_russian_rules("russian"));
  CHECK_FALSE(dialect_resolves_to_russian_rules("de"));
}

TEST_CASE("russian: lowercase homograph overrides capitalized") {
  const auto p = make_temp_tsv(
      "\xD0\xA0\xD0\xBE\xD0\xBC\xD0\xB0\twrong\n"
      "\xD1\x80\xD0\xBE\xD0\xBC\xD0\xB0\tright\n");
  moonshine_tts::RussianRuleG2p g(p);
  CHECK(g.word_to_ipa("\xD1\x80\xD0\xBE\xD0\xBC\xD0\xB0") == "right");
  std::filesystem::remove(p);
}

#ifdef MOONSHINE_TTS_WITH_G2P_CLASS
TEST_CASE("russian: MoonshineG2P ru uses rule backend and matches RussianRuleG2p") {
  const auto p = make_temp_tsv(
      "\xD1\x82\xD0\xB5\xD1\x81\xD1\x82\t\xCB\x88t\xC9\x9Bst\n");  // тест + IPA
  moonshine_tts::MoonshineG2POptions opt;
  opt.files.set_path(moonshine_tts::kG2pRussianDictKey, p);
  moonshine_tts::MoonshineG2P g("ru", opt);
  CHECK(g.uses_russian_rules());
  CHECK_FALSE(g.uses_onnx());
  CHECK(g.dialect_id() == "ru-RU");
  moonshine_tts::RussianRuleG2p direct(p);
  const std::string w = "\xD1\x82\xD0\xB5\xD1\x81\xD1\x82";
  CHECK(g.text_to_ipa(w) == direct.text_to_ipa(w));
  std::filesystem::remove(p);
}
#endif

TEST_CASE("russian: litva matches reference IPA when data and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = r::moonshine_tts_bundled_data_dir_relative() / "ru" / "dict.tsv";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "ru" / "rule_g2p_litva_word.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::RussianRuleG2p g(dict);
  const std::string expected = r::load_ref_text_trimmed(golden);
  CHECK(g.word_to_ipa("\xD0\x9B\xD0\xB8\xD1\x82\xD0\xB2\xD0\xB0") == expected);
}

TEST_CASE("russian: Cyrillic preposition plus 1891 matches reference IPA when data and golden exist") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = r::moonshine_tts_bundled_data_dir_relative() / "ru" / "dict.tsv";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "ru" / "rule_g2p_v_1891_line.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(golden)) {
    return;
  }
  const std::string line = "\xD0\x92 1891";  // "В 1891"
  const std::string expected = r::load_ref_text_trimmed(golden);
  moonshine_tts::RussianRuleG2p g(dict);
  CHECK(g.text_to_ipa(line) == expected);
}

TEST_CASE("russian: wiki-text first 100 lines match reference IPA when data and golden exist") {
  constexpr std::size_t kWikiParityLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const std::filesystem::path dict = r::moonshine_tts_bundled_data_dir_relative() / "ru" / "dict.tsv";
  const std::filesystem::path wiki = r::moonshine_tts_bundled_data_dir_relative() / "ru" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "ru" / "rule_g2p_wiki_100.txt";
  if (!std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::RussianRuleG2p g(dict);
  const auto src = r::read_text_first_lines(wiki, kWikiParityLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g.text_to_ipa(src[i]) == py[i]);
  }
}
