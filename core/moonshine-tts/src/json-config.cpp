#include "json-config.h"

#include "constants.h"

#include <fstream>
#include <nlohmann/json.h>
#include <stdexcept>

namespace moonshine_tts {

namespace {

nlohmann::json read_json_file(const std::filesystem::path& p) {
  std::ifstream in(p);
  if (!in) {
    throw std::runtime_error("cannot open JSON: " + p.string());
  }
  nlohmann::json j;
  in >> j;
  return j;
}

void validate_header(const nlohmann::json& cfg, const std::string& expect_kind,
                     const std::filesystem::path& path_for_msg) {
  if (!cfg.contains("config_schema_version") ||
      cfg["config_schema_version"].get<int>() != kConfigOnnxSchemaVersion) {
    throw std::runtime_error("unsupported config_schema_version in " + path_for_msg.string());
  }
  if (!cfg.contains("model_kind") || cfg["model_kind"].get<std::string>() != expect_kind) {
    throw std::runtime_error("model_kind mismatch in " + path_for_msg.string());
  }
}

std::unordered_map<std::string, int64_t> json_object_to_string_int_map(const nlohmann::json& o) {
  std::unordered_map<std::string, int64_t> m;
  for (auto it = o.begin(); it != o.end(); ++it) {
    m[it.key()] = it.value().get<int64_t>();
  }
  return m;
}

std::vector<std::string> stoi_to_itos(const std::unordered_map<std::string, int64_t>& stoi) {
  int64_t mx = -1;
  for (const auto& [_, v] : stoi) {
    mx = std::max(mx, v);
  }
  std::vector<std::string> itos(static_cast<size_t>(mx + 1));
  for (const auto& [s, i] : stoi) {
    if (i >= 0 && static_cast<size_t>(i) < itos.size()) {
      itos[static_cast<size_t>(i)] = s;
    }
  }
  return itos;
}

}  // namespace

OovOnnxTables load_oov_tables(const std::filesystem::path& model_onnx_path) {
  const std::filesystem::path parent = model_onnx_path.parent_path();
  const std::filesystem::path merged = parent / std::string(kConfigOnnxFileName);
  nlohmann::json char_j;
  nlohmann::json phon_j;
  nlohmann::json train_cfg;
  nlohmann::json oov_meta;

  if (!std::filesystem::exists(merged)) {
    throw std::runtime_error("onnx-config.json not found at " + merged.generic_string());
  }
  const nlohmann::json cfg = read_json_file(merged);
  validate_header(cfg, "oov", merged);
  char_j = cfg["char_vocab"];
  phon_j = cfg["phoneme_vocab"];
  train_cfg = cfg["train_config"];
  oov_meta = cfg["oov_index"];

  OovOnnxTables t;
  t.char_stoi = json_object_to_string_int_map(char_j);
  t.phoneme_stoi = json_object_to_string_int_map(phon_j);
  t.phoneme_itos = stoi_to_itos(t.phoneme_stoi);
  t.max_seq_len = train_cfg.at("max_seq_len").get<int>();
  if (!oov_meta.contains("max_phoneme_len")) {
    throw std::runtime_error("oov_index missing max_phoneme_len");
  }
  t.max_phoneme_len = oov_meta.at("max_phoneme_len").get<int>();
  t.pad_id = t.char_stoi.at(std::string(kSpecialPad));
  t.bos = t.phoneme_stoi.at(std::string(kPhonBos));
  t.eos = t.phoneme_stoi.at(std::string(kPhonEos));
  t.phon_pad = t.phoneme_stoi.at(std::string(kPhonPad));
  return t;
}

HeteronymOnnxTables load_heteronym_tables(const std::filesystem::path& model_onnx_path) {
  const std::filesystem::path parent = model_onnx_path.parent_path();
  const std::filesystem::path merged = parent / std::string(kConfigOnnxFileName);
  nlohmann::json char_j;
  nlohmann::json phon_j;
  nlohmann::json train_cfg;
  nlohmann::json homograph;

  if (!std::filesystem::exists(merged)) {
    throw std::runtime_error("onnx-config.json not found at " + merged.generic_string());
  }
  const nlohmann::json cfg = read_json_file(merged);
  validate_header(cfg, "heteronym", merged);
  char_j = cfg["char_vocab"];
  phon_j = cfg["phoneme_vocab"];
  train_cfg = cfg["train_config"];
  homograph = cfg["homograph_index"];

  HeteronymOnnxTables t;
  t.char_stoi = json_object_to_string_int_map(char_j);
  t.phoneme_stoi = json_object_to_string_int_map(phon_j);
  t.phoneme_itos = stoi_to_itos(t.phoneme_stoi);
  const auto& oc = homograph.at("ordered_candidates");
  for (auto it = oc.begin(); it != oc.end(); ++it) {
    std::vector<std::string> alts;
    for (const auto& el : it.value()) {
      alts.push_back(el.get<std::string>());
    }
    t.ordered_candidates[it.key()] = std::move(alts);
  }
  const int mc = homograph.at("max_candidates").get<int>();
  const std::string gk = homograph.at("group_key").get<std::string>();
  if (train_cfg.at("max_candidates").get<int>() != mc) {
    throw std::runtime_error("train_config max_candidates != homograph_index");
  }
  if (train_cfg.at("group_key").get<std::string>() != gk) {
    throw std::runtime_error("train_config group_key != homograph_index");
  }
  t.group_key = gk;
  t.max_seq_len = train_cfg.at("max_seq_len").get<int>();
  if (train_cfg.contains("max_phoneme_len")) {
    t.max_phoneme_len = train_cfg.at("max_phoneme_len").get<int>();
  } else if (train_cfg.contains("max_ipa_len")) {
    t.max_phoneme_len = train_cfg.at("max_ipa_len").get<int>();
  } else {
    t.max_phoneme_len = 64;
  }
  if (train_cfg.contains("levenshtein_extra_phonemes")) {
    t.levenshtein_extra = train_cfg.at("levenshtein_extra_phonemes").get<int>();
  }
  t.pad_id = t.char_stoi.at(std::string(kSpecialPad));
  t.bos = t.phoneme_stoi.at(std::string(kPhonBos));
  t.eos = t.phoneme_stoi.at(std::string(kPhonEos));
  t.phon_pad = t.phoneme_stoi.at(std::string(kPhonPad));
  return t;
}

}  // namespace moonshine_tts
