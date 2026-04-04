#ifndef MOONSHINE_TTS_CHINESE_TOK_POS_ONNX_H
#define MOONSHINE_TTS_CHINESE_TOK_POS_ONNX_H

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moonshine_tts {

struct MoonshineG2POptions;

/// Simplified-Chinese surfaces + UD UPOS via ONNX (``KoichiYasuoka/chinese-roberta-base-upos`` BIO),
/// mirroring ``chinese_tok_pos_onnx.ChineseTokPosOnnx`` / WordPiece from
/// ``data/zh_hans/roberta_chinese_base_upos_onnx/``.
class ChineseTokPosOnnx {
 public:
  explicit ChineseTokPosOnnx(std::filesystem::path model_dir, bool use_cuda = false);
  /// Load tokenizer + ONNX from ``opt`` when non-null and assets exist, else from disk under
  /// ``model_dir_fallback``.
  ChineseTokPosOnnx(const MoonshineG2POptions* opt, std::string_view onnx_bundle_key,
                    std::filesystem::path model_dir_fallback, bool use_cuda = false);

  /// One ``(surface, UPOS)`` per BIO word (empty input → empty list).
  std::vector<std::pair<std::string, std::string>> annotate(std::string_view text_utf8);

  /// ``tok1/UPOS1 tok2/UPOS2 `` (trailing space if non-empty).
  static std::string format_annotated_line(const std::vector<std::pair<std::string, std::string>>& pairs);

  const std::filesystem::path& model_dir() const { return model_dir_; }

 private:
  Ort::Env env_;
  Ort::MemoryInfo mem_;
  std::filesystem::path model_dir_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<std::string> id2label_;
  int64_t pad_id_ = 1;
  int max_sequence_length_ = 512;
  std::string logits_output_name_;
  std::string cached_vocab_txt_;
  std::string cached_tokenizer_cfg_json_;
  std::vector<std::uint8_t> onnx_model_storage_;
};

/// ``<repo>/data/zh_hans/roberta_chinese_base_upos_onnx`` when *repo_root* is the repository root.
std::filesystem::path default_chinese_tok_pos_model_dir(const std::filesystem::path& g2p_data_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_CHINESE_TOK_POS_ONNX_H
