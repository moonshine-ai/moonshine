#ifndef MOONSHINE_TTS_CHINESE_ONNX_G2P_H
#define MOONSHINE_TTS_CHINESE_ONNX_G2P_H

#include "chinese-tok-pos-onnx.h"
#include "chinese.h"
#include "rule-based-g2p.h"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;
struct MoonshineG2POptions;

/// ONNX BIO segmentation + UPOS + ``data/zh_hans/dict.tsv`` (mirrors ``chinese_rule_g2p.ChineseOnnxLexiconG2p``).
class ChineseOnnxG2p {
 public:
  explicit ChineseOnnxG2p(std::filesystem::path model_dir,
                          std::filesystem::path dict_tsv,
                          bool use_cuda = false);
  explicit ChineseOnnxG2p(std::filesystem::path model_dir, std::string dict_tsv_utf8, bool use_cuda = false);
  ChineseOnnxG2p(const MoonshineG2POptions& opt, std::filesystem::path model_dir,
                 std::filesystem::path dict_tsv, bool use_cuda = false);
  ChineseOnnxG2p(const MoonshineG2POptions& opt, std::filesystem::path model_dir,
                 std::string dict_tsv_utf8, bool use_cuda = false);

  std::string text_to_ipa(std::string text_utf8, std::vector<G2pWordLog>* per_word_log = nullptr);

  const ChineseTokPosOnnx& tok() const { return tok_; }

 private:
  ChineseTokPosOnnx tok_;
  ChineseRuleG2p lex_;
};

/// ``MoonshineG2P`` adapter: same dialect ids as ``ChineseRuleG2p`` but requires the RoBERTa UPOS ONNX bundle.
class ChineseOnnxRuleG2p : public RuleBasedG2p {
 public:
  explicit ChineseOnnxRuleG2p(std::filesystem::path onnx_model_dir,
                              std::filesystem::path dict_tsv,
                              bool use_cuda = false);
  explicit ChineseOnnxRuleG2p(std::filesystem::path onnx_model_dir, std::string dict_tsv_utf8,
                                bool use_cuda = false);
  ChineseOnnxRuleG2p(const MoonshineG2POptions& opt, std::filesystem::path onnx_model_dir,
                     std::filesystem::path dict_tsv, bool use_cuda = false);
  ChineseOnnxRuleG2p(const MoonshineG2POptions& opt, std::filesystem::path onnx_model_dir,
                     std::string dict_tsv_utf8, bool use_cuda = false);

  std::string text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::unique_ptr<ChineseOnnxG2p> impl_;
};

}  // namespace moonshine_tts

#endif
