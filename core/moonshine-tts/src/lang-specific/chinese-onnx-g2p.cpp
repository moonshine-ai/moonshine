#include "chinese-onnx-g2p.h"

#include "g2p-word-log.h"
#include "utf8-utils.h"

#include <utility>

extern "C" {
#include <utf8proc.h>
}

namespace moonshine_tts {
namespace {

std::string utf8_nfc_utf8proc(std::string_view s) {
  const std::string tmp(s);
  utf8proc_uint8_t* p =
      utf8proc_NFC(reinterpret_cast<const utf8proc_uint8_t*>(tmp.c_str()));
  if (p == nullptr) {
    return std::string(s);
  }
  std::string out(reinterpret_cast<char*>(p));
  std::free(p);
  return out;
}

}  // namespace

ChineseOnnxG2p::ChineseOnnxG2p(std::filesystem::path model_dir,
                               std::filesystem::path dict_tsv,
                               bool use_cuda)
    : tok_(std::move(model_dir), use_cuda), lex_(std::move(dict_tsv)) {}

std::string ChineseOnnxG2p::text_to_ipa(std::string text_utf8, std::vector<G2pWordLog>* per_word_log) {
  std::string raw = utf8_nfc_utf8proc(trim_ascii_ws_copy(text_utf8));
  if (raw.empty()) {
    return "";
  }
  const auto pairs = tok_.annotate(raw);
  std::string out;
  for (const auto& pr : pairs) {
    const std::string& w = pr.first;
    const std::string& pos = pr.second;
    std::string wipa = lex_.word_to_ipa_with_pos(w, pos);
    if (per_word_log != nullptr) {
      per_word_log->push_back(G2pWordLog{w, w, G2pWordPath::kRuleBasedG2p, wipa});
    }
    if (!wipa.empty()) {
      if (!out.empty() && out.back() != ' ') {
        out.push_back(' ');
      }
      out += wipa;
    }
  }
  return out;
}

ChineseOnnxRuleG2p::ChineseOnnxRuleG2p(std::filesystem::path onnx_model_dir,
                                       std::filesystem::path dict_tsv,
                                       bool use_cuda)
    : impl_(std::make_unique<ChineseOnnxG2p>(std::move(onnx_model_dir), std::move(dict_tsv), use_cuda)) {}

std::string ChineseOnnxRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  return impl_->text_to_ipa(std::move(text), per_word_log);
}

}  // namespace moonshine_tts
