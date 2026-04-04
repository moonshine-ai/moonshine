#include "english.h"

#include "builtin-cpp-data-root.h"
#include "cmudict-tsv.h"
#include "g2p-word-log.h"
#include "english-hand-oov.h"
#include "english-numbers.h"
#include "onnx-g2p-models.h"
#include "text-normalize.h"
#include "utf8-utils.h"

#include <cctype>
#include <fstream>
#include <iterator>
#include <optional>
#include <unordered_map>
#include <nlohmann/json.h>
#include <onnxruntime_cxx_api.h>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace moonshine_tts {

struct EnglishRuleG2p::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "moonshine_tts_en"};
  std::unique_ptr<CmudictTsv> dict;
  std::unordered_map<std::string, std::vector<std::string>> homograph_order;
  std::unique_ptr<OnnxHeteronymG2p> het;
  std::unique_ptr<OnnxOovG2p> oov;
};

namespace {

void append_log(std::vector<G2pWordLog>* out, G2pWordLog entry) {
  if (out != nullptr) {
    out->push_back(std::move(entry));
  }
}

void load_homograph_json_string(std::string_view utf8,
                                std::unordered_map<std::string, std::vector<std::string>>& out) {
  const nlohmann::json j = nlohmann::json::parse(std::string(utf8));
  const auto it = j.find("ordered_candidates");
  if (it == j.end() || !it->is_object()) {
    return;
  }
  for (const auto& [k, v] : it->items()) {
    if (!v.is_array()) {
      continue;
    }
    std::vector<std::string> ipas;
    for (const auto& x : v) {
      if (x.is_string()) {
        ipas.push_back(x.get<std::string>());
      }
    }
    std::string key = k;
    for (char& c : key) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    out[std::move(key)] = std::move(ipas);
  }
}

void load_homograph_json(const std::filesystem::path& path,
                         std::unordered_map<std::string, std::vector<std::string>>& out) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("English G2P: failed to open homograph index: " + path.generic_string());
  }
  const std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  load_homograph_json_string(content, out);
}

std::vector<std::string> merge_candidates(
    const std::string& key,
    const std::vector<std::string>& tsv_alts,
    const std::unordered_map<std::string, std::vector<std::string>>& homograph_order) {
  if (tsv_alts.size() <= 1) {
    return tsv_alts;
  }
  std::unordered_set<std::string> tset(tsv_alts.begin(), tsv_alts.end());
  const auto hit = homograph_order.find(key);
  if (hit == homograph_order.end()) {
    std::vector<std::string> s = tsv_alts;
    std::sort(s.begin(), s.end());
    return s;
  }
  std::vector<std::string> out;
  std::unordered_set<std::string> seen;
  for (const std::string& x : hit->second) {
    if (tset.count(x) != 0u && seen.count(x) == 0u) {
      out.push_back(x);
      seen.insert(x);
    }
  }
  std::vector<std::string> sorted_rest = tsv_alts;
  std::sort(sorted_rest.begin(), sorted_rest.end());
  for (const std::string& x : sorted_rest) {
    if (seen.count(x) == 0u) {
      out.push_back(x);
      seen.insert(x);
    }
  }
  return out.empty() ? sorted_rest : out;
}

}  // namespace

EnglishRuleG2p::EnglishRuleG2p(std::filesystem::path dict_tsv,
                             std::filesystem::path homograph_json,
                             std::optional<std::filesystem::path> heteronym_onnx,
                             std::optional<std::filesystem::path> oov_onnx,
                             bool use_cuda,
                             std::optional<EnglishOnnxAuxMemory> heteronym_from_memory,
                             std::optional<EnglishOnnxAuxMemory> oov_from_memory)
    : impl_(std::make_unique<Impl>()) {
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    throw std::runtime_error("English G2P: dictionary not found at " + dict_tsv.generic_string());
  }
  impl_->dict = std::make_unique<CmudictTsv>(dict_tsv);
  if (std::filesystem::is_regular_file(homograph_json)) {
    try {
      load_homograph_json(homograph_json, impl_->homograph_order);
    } catch (const std::exception&) {
      impl_->homograph_order.clear();
    }
  }
  if (heteronym_from_memory && !heteronym_from_memory->model_onnx.empty() &&
      !heteronym_from_memory->onnx_config_json_utf8.empty()) {
    impl_->het = std::make_unique<OnnxHeteronymG2p>(
        impl_->env, heteronym_from_memory->model_onnx.data(),
        heteronym_from_memory->model_onnx.size(),
        nlohmann::json::parse(heteronym_from_memory->onnx_config_json_utf8), use_cuda);
  } else if (heteronym_onnx && std::filesystem::is_regular_file(*heteronym_onnx)) {
    impl_->het = std::make_unique<OnnxHeteronymG2p>(impl_->env, *heteronym_onnx, use_cuda);
  }
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

EnglishRuleG2p::EnglishRuleG2p(std::string dict_tsv_utf8,
                               std::optional<std::string> homograph_index_json_utf8,
                               std::optional<std::filesystem::path> heteronym_onnx,
                               std::optional<std::filesystem::path> oov_onnx, bool use_cuda,
                               std::optional<EnglishOnnxAuxMemory> heteronym_from_memory,
                               std::optional<EnglishOnnxAuxMemory> oov_from_memory)
    : impl_(std::make_unique<Impl>()) {
  if (dict_tsv_utf8.empty()) {
    throw std::runtime_error("English G2P: empty dictionary buffer");
  }
  impl_->dict = std::make_unique<CmudictTsv>(std::string_view(dict_tsv_utf8));
  if (homograph_index_json_utf8 && !homograph_index_json_utf8->empty()) {
    try {
      load_homograph_json_string(*homograph_index_json_utf8, impl_->homograph_order);
    } catch (const std::exception&) {
      impl_->homograph_order.clear();
    }
  }
  if (heteronym_from_memory && !heteronym_from_memory->model_onnx.empty() &&
      !heteronym_from_memory->onnx_config_json_utf8.empty()) {
    impl_->het = std::make_unique<OnnxHeteronymG2p>(
        impl_->env, heteronym_from_memory->model_onnx.data(),
        heteronym_from_memory->model_onnx.size(),
        nlohmann::json::parse(heteronym_from_memory->onnx_config_json_utf8), use_cuda);
  } else if (heteronym_onnx && std::filesystem::is_regular_file(*heteronym_onnx)) {
    impl_->het = std::make_unique<OnnxHeteronymG2p>(impl_->env, *heteronym_onnx, use_cuda);
  }
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
      {"en_us", "en-US", "en-us", "english", "en"});
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

    std::vector<std::string> alts = merge_candidates(gkey, *alts_ptr, impl_->homograph_order);
    if (alts.size() == 1) {
      append_log(per_word_log,
                 G2pWordLog{token, gkey, G2pWordPath::kDictUnambiguous, alts[0]});
      parts.push_back(alts[0]);
    } else if (impl_->het) {
      const std::string ipa =
          impl_->het->disambiguate_ipa(text, start, end, gkey, alts);
      append_log(per_word_log,
                 G2pWordLog{token, gkey, G2pWordPath::kDictHeteronym, ipa});
      parts.push_back(ipa);
    } else {
      append_log(per_word_log,
                 G2pWordLog{token, gkey,
                            G2pWordPath::kDictFirstAlternativeNoHeteronymModel, alts[0]});
      parts.push_back(alts[0]);
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

bool dialect_resolves_to_english_rules(std::string_view dialect_id) {
  const std::string s = normalize_rule_based_dialect_cli_key(dialect_id);
  if (s.empty()) {
    return false;
  }
  return s == "en-us" || s == "english" || s == "en";
}

std::filesystem::path resolve_english_dict_path(const std::filesystem::path& model_root) {
  const std::filesystem::path under = model_root / "en_us" / "dict_filtered_heteronyms.tsv";
  if (std::filesystem::is_regular_file(under)) {
    return under;
  }
  const std::filesystem::path bundled =
      builtin_cpp_data_root() / "en_us" / "dict_filtered_heteronyms.tsv";
  if (std::filesystem::is_regular_file(bundled)) {
    return bundled;
  }
  return model_root.parent_path() / "models" / "en_us" / "dict_filtered_heteronyms.tsv";
}

}  // namespace moonshine_tts
