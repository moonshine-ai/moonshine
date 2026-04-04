#ifndef MOONSHINE_TTS_ONNX_G2P_MODELS_H
#define MOONSHINE_TTS_ONNX_G2P_MODELS_H

#include "json-config.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

#include <nlohmann/json.h>

namespace moonshine_tts {

class OnnxOovG2p {
 public:
  OnnxOovG2p(Ort::Env& env, const std::filesystem::path& model_onnx, bool use_cuda);
  OnnxOovG2p(Ort::Env& env, const void* model_onnx_bytes, size_t model_onnx_size,
             const nlohmann::json& onnx_config, bool use_cuda);

  std::vector<std::string> predict_phonemes(const std::string& word);

 private:
  OovOnnxTables tab_;
  Ort::Session session_;
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
};

class OnnxHeteronymG2p {
 public:
  OnnxHeteronymG2p(Ort::Env& env, const std::filesystem::path& model_onnx, bool use_cuda);
  OnnxHeteronymG2p(Ort::Env& env, const void* model_onnx_bytes, size_t model_onnx_size,
                   const nlohmann::json& onnx_config, bool use_cuda);

  std::string disambiguate_ipa(const std::string& full_text,
                                             int span_s,
                                             int span_e,
                                             const std::string& lookup_key,
                                             const std::vector<std::string>& cmudict_alternatives);

 private:
  HeteronymOnnxTables tab_;
  Ort::Session session_;
  Ort::MemoryInfo mem_{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_ONNX_G2P_MODELS_H
