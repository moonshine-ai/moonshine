#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p.h"
#include "rule-g2p-test-support.h"

#include <cstdlib>
#include <filesystem>

using namespace moonshine_tts;
namespace r = moonshine_tts::rule_g2p_test;

TEST_CASE("MoonshineG2P en_us rule-based when MOONSHINE_TTS_MODELS_ROOT is set") {
  const char* root = std::getenv("MOONSHINE_TTS_MODELS_ROOT");
  if (root == nullptr || root[0] == '\0') {
    MESSAGE("skip: export MOONSHINE_TTS_MODELS_ROOT to models/en_us parent");
    return;
  }
  MoonshineG2POptions opt;
  opt.model_root = std::filesystem::path(root).parent_path();
  const std::string dialect_dir = std::filesystem::path(root).filename().string();
  MoonshineG2P g2p(dialect_dir, opt);
  CHECK_FALSE(g2p.uses_onnx());
  CHECK(g2p.uses_english_rules());
  CHECK_FALSE(g2p.uses_spanish_rules());
  CHECK_FALSE(g2p.uses_german_rules());
  const std::string out = g2p.text_to_ipa("Hello world!");
  CHECK(out.size() > 0);
}

TEST_CASE("MoonshineG2P ja-JP when data/ja assets exist under repo") {
  const auto repo = r::repo_root_from_tests_cpp(__FILE__);
  const auto ja_onnx = repo / "data" / "ja" / "roberta_japanese_char_luw_upos_onnx" / "model.onnx";
  const auto ja_dict = repo / "data" / "ja" / "dict.tsv";
  if (!std::filesystem::is_regular_file(ja_onnx) || !std::filesystem::is_regular_file(ja_dict)) {
    return;
  }
  MoonshineG2POptions opt;
  opt.model_root = repo / "models";
  MoonshineG2P g2p("ja", opt);
  CHECK(g2p.uses_japanese_rules());
  CHECK_FALSE(g2p.uses_korean_rules());
  const std::string out = g2p.text_to_ipa("東京に行きます。");
  CHECK(out.find("toɯkjoɯ") != std::string::npos);
}
