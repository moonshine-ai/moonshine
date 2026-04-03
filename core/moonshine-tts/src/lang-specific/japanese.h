#ifndef MOONSHINE_TTS_LANG_SPECIFIC_JAPANESE_H
#define MOONSHINE_TTS_LANG_SPECIFIC_JAPANESE_H

#include "rule-based-g2p.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

class JapaneseOnnxG2p;

/// Japanese G2P via ONNX LUW segmentation + ``data/ja/dict.tsv`` (mirrors ``japanese_onnx_g2p.py``).
class JapaneseRuleG2p : public RuleBasedG2p {
 public:
  explicit JapaneseRuleG2p(std::filesystem::path onnx_model_dir,
                           std::filesystem::path dict_tsv,
                           bool use_cuda = false);
  ~JapaneseRuleG2p() override;

  JapaneseRuleG2p(JapaneseRuleG2p&&) noexcept;
  JapaneseRuleG2p& operator=(JapaneseRuleG2p&&) noexcept;
  JapaneseRuleG2p(const JapaneseRuleG2p&) = delete;
  JapaneseRuleG2p& operator=(const JapaneseRuleG2p&) = delete;

  static std::vector<std::string> dialect_ids();

  const std::string& dialect_id() const { return dialect_id_; }

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  std::string dialect_id_{"ja-JP"};
  std::unique_ptr<JapaneseOnnxG2p> impl_;
};

bool dialect_resolves_to_japanese_rules(std::string_view dialect_id);

/// ``<model-root>/../data/ja/dict.tsv`` or ``<model-root>/ja/dict.tsv``.
std::filesystem::path resolve_japanese_dict_path(const std::filesystem::path& model_root);

/// Directory containing ``model.onnx`` for char-LUW UPOS (default under ``data/ja/``).
std::filesystem::path resolve_japanese_onnx_model_dir(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif
