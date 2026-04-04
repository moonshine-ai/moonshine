#ifndef MOONSHINE_TTS_JSON_CONFIG_H
#define MOONSHINE_TTS_JSON_CONFIG_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.h>

namespace moonshine_tts {

struct OovOnnxTables {
  std::unordered_map<std::string, int64_t> char_stoi;
  std::unordered_map<std::string, int64_t> phoneme_stoi;
  std::vector<std::string> phoneme_itos;
  int max_seq_len = 0;
  int max_phoneme_len = 0;
  int64_t pad_id = 0;
  int64_t bos = 0;
  int64_t eos = 0;
  int64_t phon_pad = 0;
};

// Loads onnx-config.json from model.onnx directory.
OovOnnxTables load_oov_tables(const std::filesystem::path& model_onnx_path);

/// Parse merged ``onnx-config.json`` (already loaded).
OovOnnxTables load_oov_tables_from_json(const nlohmann::json& cfg, std::string_view source_label);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_JSON_CONFIG_H
