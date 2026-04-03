#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "chinese-tok-pos-onnx.h"
#include "rule-g2p-test-support.h"

namespace r = moonshine_tts::rule_g2p_test;

TEST_CASE("chinese tok pos: single sentence matches reference file") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_tts::default_chinese_tok_pos_model_dir(repo);
  const std::filesystem::path golden = r::tests_data_dir(repo) / "zh_hans" / "tok_pos_sample.txt";
  if (!std::filesystem::is_regular_file(model / "model.onnx") ||
      !std::filesystem::is_regular_file(golden)) {
    return;
  }
  const std::string expected = r::load_ref_text_trimmed(golden);
  moonshine_tts::ChineseTokPosOnnx pipe(model, false);
  const auto pairs = pipe.annotate("上海是一座城市。");
  CHECK(moonshine_tts::ChineseTokPosOnnx::format_annotated_line(pairs) == expected);
}

TEST_CASE("chinese tok pos: first 100 wiki lines match reference when assets and golden exist") {
  constexpr std::size_t kWikiLines = 100;
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto model = moonshine_tts::default_chinese_tok_pos_model_dir(repo);
  const auto wiki = repo / "data" / "zh_hans" / "wiki-text.txt";
  const std::filesystem::path golden = r::tests_data_dir(repo) / "zh_hans" / "tok_pos_wiki_filtered.txt";
  if (!std::filesystem::is_regular_file(model / "model.onnx") ||
      !std::filesystem::is_regular_file(wiki) || !std::filesystem::is_regular_file(golden)) {
    return;
  }
  moonshine_tts::ChineseTokPosOnnx pipe(model, false);
  const auto src = r::read_text_first_lines(wiki, kWikiLines);
  std::vector<std::string> tokenizable;
  for (const std::string& line : src) {
    try {
      (void)pipe.annotate(line);
      tokenizable.push_back(line);
    } catch (const std::exception&) {
      // Same WordPiece path as Python: skip lines with tricky mixed scripts / NFC edge cases.
    }
  }
  if (tokenizable.empty()) {
    return;
  }
  const std::vector<std::string> py = r::load_ref_lines(golden);
  REQUIRE(py.size() == tokenizable.size());
  for (std::size_t i = 0; i < tokenizable.size(); ++i) {
    INFO("wiki filtered index " << (i + 1));
    const auto pairs = pipe.annotate(tokenizable[i]);
    CHECK(moonshine_tts::ChineseTokPosOnnx::format_annotated_line(pairs) == py[i]);
  }
}
