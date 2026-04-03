#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "arabic.h"
#include "rule-g2p-test-support.h"

namespace r = moonshine_tts::rule_g2p_test;

TEST_CASE("arabic rule g2p: first 100 wiki lines match reference IPA when assets and golden exist") {
  constexpr std::size_t kWikiLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_tts::resolve_arabic_onnx_model_dir(repo / "models");
  const auto dict = moonshine_tts::resolve_arabic_dict_path(repo / "models");
  const auto wiki = repo / "data" / "ar" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "ar" / "rule_g2p_wiki_100.txt";
  if (!std::filesystem::is_regular_file(model / "model.onnx") ||
      !std::filesystem::is_regular_file(dict) || !std::filesystem::is_regular_file(wiki) ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::ArabicRuleG2p g2p(model, dict, false);
  const auto src = r::read_text_first_lines(wiki, kWikiLines);
  const std::vector<std::string> py = r::ref_lines_prefix(golden, src.size());
  REQUIRE(py.size() == src.size());
  for (std::size_t i = 0; i < src.size(); ++i) {
    INFO("wiki line " << (i + 1));
    CHECK(g2p.text_to_ipa(src[i]) == py[i]);
  }
}
