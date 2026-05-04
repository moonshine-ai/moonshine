#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "json-config.h"

#include <filesystem>
#include <fstream>

using namespace moonshine_tts;

TEST_CASE("load_oov_tables from onnx-config.json") {
  const auto dir = std::filesystem::temp_directory_path() / "g2p_cfg_test";
  std::filesystem::create_directories(dir);
  const auto model = dir / "model.onnx";
  {
    std::ofstream touch(model);
    touch << "x";
  }
  {
    std::ofstream cfg(dir / "onnx-config.json");
    cfg << R"({
  "config_schema_version": 1,
  "model_kind": "oov",
  "char_vocab": {"<pad>": 0, "<unk>": 1, "a": 2},
  "phoneme_vocab": {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3, "x": 4},
  "train_config": {"max_seq_len": 8},
  "oov_index": {"max_phoneme_len": 16}
})";
  }
  const OovOnnxTables t = load_oov_tables(model);
  CHECK(t.max_seq_len == 8);
  CHECK(t.max_phoneme_len == 16);
  CHECK(t.bos == 2);
  std::filesystem::remove_all(dir);
}
