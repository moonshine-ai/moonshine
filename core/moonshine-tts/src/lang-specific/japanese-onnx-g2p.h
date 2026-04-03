#ifndef MOONSHINE_TTS_JAPANESE_ONNX_G2P_H
#define MOONSHINE_TTS_JAPANESE_ONNX_G2P_H

#include "japanese-tok-pos-onnx.h"

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

/// ONNX LUW segmentation + ``data/ja/dict.tsv`` + kana IPA (mirrors ``japanese_onnx_g2p.py``).
class JapaneseOnnxG2p {
 public:
  explicit JapaneseOnnxG2p(std::filesystem::path model_dir,
                           std::filesystem::path dict_tsv,
                           bool use_cuda = false);

  std::string g2p_word(std::string word_utf8);
  std::string text_to_ipa(std::string text_utf8);

  const JapaneseTokPosOnnx& tok() const { return tok_; }

 private:
  JapaneseTokPosOnnx tok_;
  std::unordered_map<std::string, std::string> lex_;
  std::unordered_map<std::string, std::vector<std::string>> by_first_;
};

std::filesystem::path default_japanese_dict_path(const std::filesystem::path& g2p_data_root);

}  // namespace moonshine_tts

#endif
