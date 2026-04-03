#ifndef MOONSHINE_TTS_JSON_CONFIG_H
#define MOONSHINE_TTS_JSON_CONFIG_H

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

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

struct HeteronymOnnxTables {
  std::unordered_map<std::string, int64_t> char_stoi;
  std::unordered_map<std::string, int64_t> phoneme_stoi;
  std::vector<std::string> phoneme_itos;
  std::unordered_map<std::string, std::vector<std::string>> ordered_candidates;
  std::string group_key;
  int max_seq_len = 0;
  int max_phoneme_len = 0;
  int levenshtein_extra = 4;
  int64_t pad_id = 0;
  int64_t bos = 0;
  int64_t eos = 0;
  int64_t phon_pad = 0;
};

// Loads onnx-config.json from model.onnx directory.
OovOnnxTables load_oov_tables(const std::filesystem::path& model_onnx_path);

HeteronymOnnxTables load_heteronym_tables(const std::filesystem::path& model_onnx_path);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_JSON_CONFIG_H
