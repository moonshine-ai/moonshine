#include "japanese.h"
#include "g2p-word-log.h"
#include "japanese-onnx-g2p.h"
#include "utf8-utils.h"

#include <cctype>
#include <optional>
#include <utility>

namespace moonshine_tts {

namespace {

std::filesystem::path absolute_model_root(const std::filesystem::path& model_root) {
  if (model_root.is_relative()) {
    std::error_code ec;
    std::filesystem::path p = std::filesystem::weakly_canonical(std::filesystem::absolute(model_root), ec);
    if (!ec) {
      return p;
    }
    return std::filesystem::absolute(model_root);
  }
  return model_root;
}

}  // namespace

JapaneseRuleG2p::JapaneseRuleG2p(std::filesystem::path onnx_model_dir,
                                  std::filesystem::path dict_tsv,
                                  bool use_cuda)
    : impl_(std::make_unique<JapaneseOnnxG2p>(std::move(onnx_model_dir), std::move(dict_tsv), use_cuda)) {}

JapaneseRuleG2p::~JapaneseRuleG2p() = default;

JapaneseRuleG2p::JapaneseRuleG2p(JapaneseRuleG2p&&) noexcept = default;
JapaneseRuleG2p& JapaneseRuleG2p::operator=(JapaneseRuleG2p&&) noexcept = default;

std::string JapaneseRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  (void)per_word_log;
  return impl_->text_to_ipa(std::move(text));
}

std::vector<std::string> JapaneseRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first({"ja", "ja-JP", "ja_JP", "japanese", "Japanese"});
}

bool dialect_resolves_to_japanese_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "ja" || s == "ja-jp" || s == "japanese";
}

std::filesystem::path resolve_japanese_dict_path(const std::filesystem::path& model_root) {
  const std::filesystem::path base = absolute_model_root(model_root);
  const std::filesystem::path p1 = base.parent_path() / "data" / "ja" / "dict.tsv";
  if (std::filesystem::is_regular_file(p1)) {
    return p1;
  }
  const std::filesystem::path p2 =
      base.parent_path().parent_path() / "data" / "ja" / "dict.tsv";
  if (std::filesystem::is_regular_file(p2)) {
    return p2;
  }
  return base / "ja" / "dict.tsv";
}

std::filesystem::path resolve_japanese_onnx_model_dir(const std::filesystem::path& model_root) {
  const std::filesystem::path base = absolute_model_root(model_root);
  const auto try_dir = [](const std::filesystem::path& dir) -> std::optional<std::filesystem::path> {
    const auto onnx = dir / "model.onnx";
    if (std::filesystem::is_regular_file(onnx)) {
      return dir;
    }
    return std::nullopt;
  };
  if (auto o = try_dir(base.parent_path() / "data" / "ja" / "roberta_japanese_char_luw_upos_onnx")) {
    return *o;
  }
  if (auto o = try_dir(base.parent_path().parent_path() / "data" / "ja" /
                       "roberta_japanese_char_luw_upos_onnx")) {
    return *o;
  }
  return base / "ja" / "roberta_japanese_char_luw_upos_onnx";
}

}  // namespace moonshine_tts
