#ifndef MOONSHINE_TTS_ARABIC_DIAC_ONNX_H
#define MOONSHINE_TTS_ARABIC_DIAC_ONNX_H

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

/// BERT token-classification tashkīl (Arabert-style), mirroring :class:`arabic_diac_onnx_infer.ArabicDiacOnnx`.
class ArabicDiacOnnx {
 public:
  explicit ArabicDiacOnnx(std::filesystem::path model_dir, bool use_cuda = false);
  ~ArabicDiacOnnx();

  ArabicDiacOnnx(const ArabicDiacOnnx&) = delete;
  ArabicDiacOnnx& operator=(const ArabicDiacOnnx&) = delete;
  ArabicDiacOnnx(ArabicDiacOnnx&&) noexcept = default;
  ArabicDiacOnnx& operator=(ArabicDiacOnnx&&) noexcept = default;

  /// Return NFC Arabic with predicted harakāt (partial model + same heuristics as Python post-process
  /// applied separately in :class:`ArabicRuleG2p`).
  std::string diacritize(std::string_view text_utf8) const;

  const std::filesystem::path& model_dir() const { return model_dir_; }

 private:
  Ort::Env env_;
  Ort::MemoryInfo mem_;
  std::filesystem::path model_dir_;
  std::vector<std::string> id2label_;
  int64_t pad_id_{0};
  int max_sequence_length_{512};
  std::unique_ptr<Ort::Session> session_;
  std::string logits_output_name_;
  std::unordered_map<std::string, std::string> label_to_diac_;
};

}  // namespace moonshine_tts

#endif
