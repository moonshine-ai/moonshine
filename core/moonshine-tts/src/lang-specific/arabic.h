#ifndef MOONSHINE_TTS_LANG_SPECIFIC_ARABIC_H
#define MOONSHINE_TTS_LANG_SPECIFIC_ARABIC_H

#include "arabic-diac-onnx.h"
#include "rule-based-g2p.h"

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;
struct MoonshineG2POptions;

/// MSA Arabic G2P: ONNX partial tashkīl + lexicon + IPA rules (mirrors :mod:`arabic_rule_g2p`).
class ArabicRuleG2p : public RuleBasedG2p {
 public:
  ArabicRuleG2p(std::filesystem::path onnx_model_dir, std::filesystem::path dict_tsv, bool use_cuda = false);
  ArabicRuleG2p(std::filesystem::path onnx_model_dir, std::string dict_tsv_utf8, bool use_cuda = false);
  ArabicRuleG2p(const MoonshineG2POptions& opt, std::filesystem::path onnx_model_dir,
                std::filesystem::path dict_tsv, bool use_cuda = false);
  ArabicRuleG2p(const MoonshineG2POptions& opt, std::filesystem::path onnx_model_dir,
                std::string dict_tsv_utf8, bool use_cuda = false);

  ArabicRuleG2p(const ArabicRuleG2p&) = delete;
  ArabicRuleG2p& operator=(const ArabicRuleG2p&) = delete;
  ArabicRuleG2p(ArabicRuleG2p&&) noexcept = default;
  ArabicRuleG2p& operator=(ArabicRuleG2p&&) noexcept = default;

  static std::vector<std::string> dialect_ids();

  std::string text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log = nullptr) override;

  const std::string& dialect_id() const { return dialect_id_; }

 private:
  std::string dialect_id_{"ar-MSA"};
  std::unique_ptr<ArabicDiacOnnx> diac_;
  std::unordered_map<std::string, std::string> lex_;
  std::string g2p_word(std::string_view word_utf8);
};

bool dialect_resolves_to_arabic_rules(std::string_view dialect_id);

std::filesystem::path resolve_arabic_dict_path(const std::filesystem::path& model_root);
std::filesystem::path resolve_arabic_onnx_model_dir(const std::filesystem::path& model_root);

}  // namespace moonshine_tts

#endif
