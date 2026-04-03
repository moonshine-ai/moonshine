#ifndef MOONSHINE_TTS_JAPANESE_TOK_POS_ONNX_H
#define MOONSHINE_TTS_JAPANESE_TOK_POS_ONNX_H

#include <onnxruntime_cxx_api.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace moonshine_tts {

/// Japanese LUW surfaces + UD UPOS via ONNX (``KoichiYasuoka/roberta-small-japanese-char-luw-upos``),
/// mirroring ``japanese_tok_pos.JapaneseTokPosOnnx`` / ``encode_for_morph_upos`` + WordPiece from
/// ``data/ja/roberta_japanese_char_luw_upos_onnx/``.
class JapaneseTokPosOnnx {
 public:
  explicit JapaneseTokPosOnnx(std::filesystem::path model_dir, bool use_cuda = false);

  /// One ``(surface, UPOS)`` per whitespace/punctuation word (empty input → empty list).
  std::vector<std::pair<std::string, std::string>> annotate(std::string_view text_utf8);

  /// Same string form as ``japanese_tok_pos`` CLI: ``tok1/UPOS1 tok2/UPOS2 `` (trailing space if non-empty).
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
};

/// ``<repo>/data/ja/roberta_japanese_char_luw_upos_onnx`` when *repo_root* is the repository root.
std::filesystem::path default_japanese_tok_pos_model_dir(const std::filesystem::path& g2p_data_root);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_JAPANESE_TOK_POS_ONNX_H
