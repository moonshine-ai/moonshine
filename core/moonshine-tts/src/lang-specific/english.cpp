#include "english.h"

#include "cmudict-tsv.h"
#include "g2p-word-log.h"
#include "english-hand-oov.h"
#include "english-numbers.h"
#include "onnx-g2p-models.h"
#include "text-normalize.h"
#include "utf8-utils.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <nlohmann/json.h>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace moonshine_tts {

struct EnglishRuleG2p::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "moonshine_tts_en"};
  std::unique_ptr<CmudictTsv> dict;
  std::unique_ptr<OnnxOovG2p> oov;
};

namespace {

void append_log(std::vector<G2pWordLog>* out, G2pWordLog entry) {
  if (out != nullptr) {
    out->push_back(std::move(entry));
  }
}

/// CMU-style heteronyms such as ``tomato`` include both US (stressed ``eɪ``) and UK (stressed ``ɑ``)
/// readings. Lexicographic IPA sort is not locale-aware; pick explicitly by dialect.
std::string pick_english_heteronym_ipa(std::vector<std::string> alts, bool prefer_british) {
  if (alts.empty()) {
    return {};
  }
  if (alts.size() == 1) {
    return alts[0];
  }
  std::sort(alts.begin(), alts.end());
  const auto has_stressed_ei = [](const std::string& s) {
    return s.find("ˈeɪ") != std::string::npos;
  };
  const auto has_stressed_open_back = [](const std::string& s) {
    return s.find("ˈɑ") != std::string::npos && s.find("ˈeɪ") == std::string::npos;
  };
  bool any_us = false;
  bool any_uk = false;
  for (const std::string& s : alts) {
    if (has_stressed_ei(s)) {
      any_us = true;
    }
    if (has_stressed_open_back(s)) {
      any_uk = true;
    }
  }
  if (any_us && any_uk) {
    for (const std::string& s : alts) {
      if (prefer_british) {
        if (has_stressed_open_back(s)) {
          return s;
        }
      } else if (has_stressed_ei(s)) {
        return s;
      }
    }
  }
  return alts[0];
}

}  // namespace

EnglishRuleG2p::EnglishRuleG2p(std::filesystem::path dict_tsv,
                               std::optional<std::filesystem::path> oov_onnx, bool use_cuda,
                               std::optional<EnglishOnnxAuxMemory> oov_from_memory,
                               bool prefer_british_heteronyms)
    : impl_(std::make_unique<Impl>()), prefer_british_heteronyms_(prefer_british_heteronyms) {
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    throw std::runtime_error("English G2P: dictionary not found at " + dict_tsv.generic_string());
  }
  impl_->dict = std::make_unique<CmudictTsv>(dict_tsv);
  if (oov_from_memory && !oov_from_memory->model_onnx.empty() &&
      !oov_from_memory->onnx_config_json_utf8.empty()) {
    impl_->oov = std::make_unique<OnnxOovG2p>(impl_->env, oov_from_memory->model_onnx.data(),
                                              oov_from_memory->model_onnx.size(),
                                              nlohmann::json::parse(oov_from_memory->onnx_config_json_utf8),
                                              use_cuda);
  } else if (oov_onnx && std::filesystem::is_regular_file(*oov_onnx)) {
    impl_->oov = std::make_unique<OnnxOovG2p>(impl_->env, *oov_onnx, use_cuda);
  }
}

EnglishRuleG2p::EnglishRuleG2p(std::string dict_tsv_utf8, std::optional<std::filesystem::path> oov_onnx,
                               bool use_cuda, std::optional<EnglishOnnxAuxMemory> oov_from_memory,
                               bool prefer_british_heteronyms)
    : impl_(std::make_unique<Impl>()), prefer_british_heteronyms_(prefer_british_heteronyms) {
  if (dict_tsv_utf8.empty()) {
    throw std::runtime_error("English G2P: empty dictionary buffer");
  }
  impl_->dict = std::make_unique<CmudictTsv>(std::string_view(dict_tsv_utf8));
  if (oov_from_memory && !oov_from_memory->model_onnx.empty() &&
      !oov_from_memory->onnx_config_json_utf8.empty()) {
    impl_->oov = std::make_unique<OnnxOovG2p>(impl_->env, oov_from_memory->model_onnx.data(),
                                              oov_from_memory->model_onnx.size(),
                                              nlohmann::json::parse(oov_from_memory->onnx_config_json_utf8),
                                              use_cuda);
  } else if (oov_onnx && std::filesystem::is_regular_file(*oov_onnx)) {
    impl_->oov = std::make_unique<OnnxOovG2p>(impl_->env, *oov_onnx, use_cuda);
  }
}

EnglishRuleG2p::~EnglishRuleG2p() = default;
EnglishRuleG2p::EnglishRuleG2p(EnglishRuleG2p&&) noexcept = default;
EnglishRuleG2p& EnglishRuleG2p::operator=(EnglishRuleG2p&&) noexcept = default;

std::vector<std::string> EnglishRuleG2p::dialect_ids() {
  return dedupe_dialect_ids_preserve_first(
      {"en_us", "en-US", "en-us", "english", "en", "en_gb", "en-GB", "en-gb", "british"});
}

std::string EnglishRuleG2p::text_to_ipa(std::string text, std::vector<G2pWordLog>* per_word_log) {
  std::vector<std::string> parts;
  int pos = 0;
  for (const auto& token : split_text_to_words(text)) {
    std::optional<std::pair<int, int>> se = utf8_find_token_codepoints(text, token, pos);
    if (!se) {
      se = utf8_find_token_codepoints(text, token, 0);
    }
    if (!se) {
      append_log(per_word_log,
                 G2pWordLog{token, "", G2pWordPath::kTokenNotLocatedInText, ""});
      continue;
    }
    const int start = se->first;
    const int end = se->second;
    (void)start;
    (void)end;
    pos = end;

    const std::string key_lookup = normalize_word_for_lookup(token);
    if (key_lookup.empty()) {
      append_log(per_word_log,
                 G2pWordLog{token, "", G2pWordPath::kSkippedEmptyKey, ""});
      continue;
    }
    const std::string gkey = normalize_grapheme_key(key_lookup);

    if (auto num_ipa = english_number_token_ipa(key_lookup)) {
      append_log(per_word_log,
                 G2pWordLog{token, gkey, G2pWordPath::kEnglishNumber, *num_ipa});
      parts.push_back(std::move(*num_ipa));
      continue;
    }

    const std::vector<std::string>* alts_ptr = impl_->dict->lookup(gkey);
    if (!alts_ptr || alts_ptr->empty()) {
      if (impl_->oov) {
        const std::vector<std::string> phones = impl_->oov->predict_phonemes(gkey);
        if (!phones.empty()) {
          std::string chunk;
          for (const auto& p : phones) {
            chunk += p;
          }
          append_log(per_word_log,
                     G2pWordLog{token, gkey, G2pWordPath::kOovModel, chunk});
          parts.push_back(std::move(chunk));
        } else {
          const std::string hand = english_hand_oov_rules_ipa(gkey);
          append_log(per_word_log,
                     G2pWordLog{token, gkey, G2pWordPath::kOovHandRules, hand});
          parts.push_back(hand);
        }
      } else {
        const std::string hand = english_hand_oov_rules_ipa(gkey);
        append_log(per_word_log,
                   G2pWordLog{token, gkey, G2pWordPath::kOovHandRules, hand});
        parts.push_back(hand);
      }
      continue;
    }

    std::vector<std::string> alts = *alts_ptr;
    if (alts.size() == 1) {
      append_log(per_word_log,
                 G2pWordLog{token, gkey, G2pWordPath::kDictUnambiguous, alts[0]});
      parts.push_back(alts[0]);
    } else {
      const std::string chosen = pick_english_heteronym_ipa(std::move(alts), prefer_british_heteronyms_);
      append_log(per_word_log,
                 G2pWordLog{token, gkey, G2pWordPath::kDictFirstAlternativeNoHeteronymModel, chosen});
      parts.push_back(chosen);
    }
  }

  std::ostringstream out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i > 0) {
      out << ' ';
    }
    out << parts[i];
  }
  return out.str();
}

bool dialect_is_british_english_variant(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  return s == "en-gb" || s == "british";
}

bool dialect_resolves_to_english_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  if (dialect_is_british_english_variant(s)) {
    return true;
  }
  return s == "en-us" || s == "english" || s == "en";
}

}  // namespace moonshine_tts
