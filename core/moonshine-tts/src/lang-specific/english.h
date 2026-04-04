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

/// Heteronym / OOV ONNX model bytes + merged ``onnx-config.json`` UTF-8 (for in-memory assets).
struct EnglishOnnxAuxMemory {
  std::vector<uint8_t> model_onnx;
  std::string onnx_config_json_utf8;
};

/// US English lexicon + homograph merge + optional ONNX heteronym/OOV + hand OOV fallback
/// (mirrors ``english_rule_g2p.EnglishLexiconRuleG2p`` + ``moonshine_onnx_g2p`` heteronym/OOV wiring).
class EnglishRuleG2p : public RuleBasedG2p {
 public:
  EnglishRuleG2p(std::filesystem::path dict_tsv,
                 std::filesystem::path homograph_json,
                 std::optional<std::filesystem::path> heteronym_onnx,
                 std::optional<std::filesystem::path> oov_onnx,
                 bool use_cuda = false,
                 std::optional<EnglishOnnxAuxMemory> heteronym_from_memory = std::nullopt,
                 std::optional<EnglishOnnxAuxMemory> oov_from_memory = std::nullopt);
  /// Lexicon and optional homograph index as UTF-8; ONNX may be on-disk or in-memory.
  EnglishRuleG2p(std::string dict_tsv_utf8, std::optional<std::string> homograph_index_json_utf8,
                 std::optional<std::filesystem::path> heteronym_onnx,
                 std::optional<std::filesystem::path> oov_onnx, bool use_cuda = false,
                 std::optional<EnglishOnnxAuxMemory> heteronym_from_memory = std::nullopt,
                 std::optional<EnglishOnnxAuxMemory> oov_from_memory = std::nullopt);
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
};

/// True for ``en_us``, ``en-us``, ``en``, ``english`` (case-insensitive).
bool dialect_resolves_to_english_rules(std::string_view dialect_id);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_LANG_SPECIFIC_ENGLISH_H
