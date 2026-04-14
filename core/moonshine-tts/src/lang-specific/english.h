#ifndef MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_H
#define MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_H

#include "rule-based-g2p.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

struct G2pWordLog;

/// OOV ONNX model bytes + merged ``onnx-config.json`` UTF-8 (for in-memory assets).
struct EnglishOnnxAuxMemory {
  std::vector<uint8_t> model_onnx;
  std::string onnx_config_json_utf8;
};

/// US English lexicon + OOV ONNX + hand OOV fallback (no heteronym ONNX).
class EnglishRuleG2p : public RuleBasedG2p {
 public:
  EnglishRuleG2p(std::filesystem::path dict_tsv, std::optional<std::filesystem::path> oov_onnx,
                 bool use_cuda = false,
                 std::optional<EnglishOnnxAuxMemory> oov_from_memory = std::nullopt,
                 bool prefer_british_heteronyms = false);
  /// Lexicon as UTF-8; OOV ONNX may be on-disk or in-memory.
  EnglishRuleG2p(std::string dict_tsv_utf8, std::optional<std::filesystem::path> oov_onnx,
                 bool use_cuda = false,
                 std::optional<EnglishOnnxAuxMemory> oov_from_memory = std::nullopt,
                 bool prefer_british_heteronyms = false);
  ~EnglishRuleG2p() override;

  EnglishRuleG2p(EnglishRuleG2p&&) noexcept;
  EnglishRuleG2p& operator=(EnglishRuleG2p&&) noexcept;

  EnglishRuleG2p(const EnglishRuleG2p&) = delete;
  EnglishRuleG2p& operator=(const EnglishRuleG2p&) = delete;

  static std::vector<std::string> dialect_ids();

  std::string text_to_ipa(std::string text,
                          std::vector<G2pWordLog>* per_word_log = nullptr) override;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  bool prefer_british_heteronyms_{false};
};

/// True for US English (``en_us``, ``en-us``, ``en``, ``english``) and British aliases
/// (``en_gb``, ``en-gb``, ``british``), case-insensitive after ``normalize_rule_based_dialect_cli_key``.
bool dialect_resolves_to_english_rules(std::string_view dialect_id);

/// True when the normalized dialect should use the British English G2P tag (``en_gb``) while sharing
/// the same CMUdict + OOV stack as US English. Does not include ``uk`` (reserved for Ukrainian ``uk``).
bool dialect_is_british_english_variant(std::string_view dialect_id);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_H
