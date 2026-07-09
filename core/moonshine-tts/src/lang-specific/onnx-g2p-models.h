#ifndef MOONSHINE_TTS_ONNX_G2P_MODELS_H
#define MOONSHINE_TTS_ONNX_G2P_MODELS_H

#include <nlohmann/json.h>
#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "json-config.h"

namespace moonshine_tts {

class OnnxOovG2p {
 public:
  OnnxOovG2p(Ort::Env& env, const std::filesystem::path& model_onnx,
             const std::vector<std::string>& ort_providers,
             const std::string& coreml_cache_dir = {});
  OnnxOovG2p(Ort::Env& env, const void* model_onnx_bytes,
             size_t model_onnx_size, const nlohmann::json& onnx_config,
             const std::vector<std::string>& ort_providers,
             const std::string& coreml_cache_dir = {});

  std::vector<std::string> predict_phonemes(const std::string& word);

 private:
  OovOnnxTables tab_;
  Ort::Session session_;
  Ort::MemoryInfo mem_{
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_ONNX_G2P_MODELS_H
