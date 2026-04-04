#include "moonshine-asset-catalog.h"

#include "moonshine-g2p-options.h"
#include "utf8-utils.h"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <unordered_map>
#include <unordered_set>

namespace moonshine_tts {

namespace {

std::string normalize_lang_key_cli(std::string_view raw) {
  std::string s = trim_ascii_ws_copy(raw);
  for (char& c : s) {
    if (c == ' ') {
      c = '_';
    } else if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return s;
}

std::string hyphen_to_underscore(std::string s) {
  for (char& c : s) {
    if (c == '-') {
      c = '_';
    }
  }
  return s;
}

std::vector<std::string> english_g2p_keys() {
  return {
      std::string(kG2pEnglishDictKey),
      std::string(kG2pEnglishG2pConfigKey),
      std::string(kG2pEnglishOovModelKey),
      std::string(kG2pEnglishOovOnnxConfigKey),
  };
}

std::vector<std::string> chinese_g2p_keys() {
  return {
      std::string(kG2pChineseDictKey),
      std::string(kG2pChineseOnnxDirKey),
      std::string(kG2pChineseOnnxMetaKey),
      std::string(kG2pChineseOnnxVocabKey),
      std::string(kG2pChineseOnnxTokenizerConfigKey),
      std::string(kG2pChineseOnnxModelKey),
  };
}

std::vector<std::string> japanese_g2p_keys() {
  return {
      std::string(kG2pJapaneseDictKey),
      std::string(kG2pJapaneseOnnxDirKey),
      std::string(kG2pJapaneseOnnxMetaKey),
      std::string(kG2pJapaneseOnnxVocabKey),
      std::string(kG2pJapaneseOnnxTokenizerConfigKey),
      std::string(kG2pJapaneseOnnxModelKey),
  };
}

std::vector<std::string> korean_g2p_keys() {
  // Rule-based Korean G2P uses the lexicon only (see ``try_korean``). Morph ONNX is test/tooling-only.
  return {std::string(kG2pKoreanDictKey)};
}

std::vector<std::string> arabic_g2p_keys() {
  return {
      std::string(kG2pArabicDictKey),
      std::string(kG2pArabicOnnxDirKey),
      std::string(kG2pArabicOnnxMetaKey),
      std::string(kG2pArabicOnnxVocabKey),
      std::string(kG2pArabicOnnxTokenizerConfigKey),
      std::string(kG2pArabicOnnxModelKey),
  };
}

const std::unordered_map<std::string, std::vector<std::string>>& g2p_dependency_map() {
  static const std::unordered_map<std::string, std::vector<std::string>> k = [] {
    const std::vector<std::string> kEmpty;
    const std::vector<std::string> kEnglish = english_g2p_keys();
    const std::vector<std::string> kDe = {std::string(kG2pGermanDictKey)};
    const std::vector<std::string> kFr = {std::string(kG2pFrenchDictKey), std::string(kG2pFrenchCsvDirKey)};
    const std::vector<std::string> kNl = {std::string(kG2pDutchDictKey)};
    const std::vector<std::string> kIt = {std::string(kG2pItalianDictKey)};
    const std::vector<std::string> kRu = {std::string(kG2pRussianDictKey)};
    const std::vector<std::string> kVi = {std::string(kG2pVietnameseDictKey)};
    const std::vector<std::string> kHi = {std::string(kG2pHindiDictKey)};
    const std::vector<std::string> kPtBr = {std::string(kG2pPtBrDictKey)};
    const std::vector<std::string> kPtPt = {std::string(kG2pPtPtDictKey)};
    const std::vector<std::string> kZh = chinese_g2p_keys();
    const std::vector<std::string> kJa = japanese_g2p_keys();
    const std::vector<std::string> kKo = korean_g2p_keys();
    const std::vector<std::string> kAr = arabic_g2p_keys();

    std::unordered_map<std::string, std::vector<std::string>> m;
    auto add = [&m](std::string_view key, const std::vector<std::string>& v) { m[std::string(key)] = v; };

    add("en_us", kEnglish);
    add("en-us", kEnglish);
    add("en", kEnglish);
    add("english", kEnglish);
    add("en_gb", kEnglish);
    add("en-gb", kEnglish);

    add("de", kDe);
    add("de-de", kDe);
    add("de_de", kDe);
    add("german", kDe);

    add("fr", kFr);
    add("fr-fr", kFr);
    add("fr_fr", kFr);
    add("french", kFr);

    add("nl", kNl);
    add("nl-nl", kNl);
    add("nl_nl", kNl);
    add("dutch", kNl);

    add("it", kIt);
    add("it-it", kIt);
    add("it_it", kIt);
    add("italian", kIt);

    add("ru", kRu);
    add("ru-ru", kRu);
    add("ru_ru", kRu);
    add("russian", kRu);

    add("zh", kZh);
    add("zh_hans", kZh);
    add("chinese", kZh);

    add("ja", kJa);
    add("jp", kJa);
    add("japanese", kJa);

    add("ko", kKo);
    add("ko_kr", kKo);
    add("korean", kKo);

    add("vi", kVi);
    add("vi-vn", kVi);
    add("vi_vn", kVi);
    add("vietnamese", kVi);

    add("ar_msa", kAr);
    add("ar", kAr);
    add("arabic", kAr);

    add("hi", kHi);
    add("hindi", kHi);

    add("pt_br", kPtBr);
    add("pt-br", kPtBr);
    add("pt", kPtBr);
    add("brazil", kPtBr);
    add("brazilian-portuguese", kPtBr);
    add("brazilianportuguese", kPtBr);
    add("portuguese-brazil", kPtBr);

    add("pt_pt", kPtPt);
    add("pt-pt", kPtPt);
    add("portugal", kPtPt);
    add("european-portuguese", kPtPt);
    add("europeanportuguese", kPtPt);

    add("tr", kEmpty);
    add("tr-tr", kEmpty);
    add("turkish", kEmpty);

    add("uk", kEmpty);
    add("uk-ua", kEmpty);
    add("uk_ua", kEmpty);
    add("ukrainian", kEmpty);

    add("es", kEmpty);
    add("es-mx", kEmpty);
    add("es_mx", kEmpty);
    add("es-es", kEmpty);
    add("es_es", kEmpty);
    add("es-ar", kEmpty);
    add("es_ar", kEmpty);
    add("spanish", kEmpty);

    return m;
  }();
  return k;
}

std::optional<std::vector<std::string>> lookup_g2p_dependency_keys(std::string_view raw) {
  const std::string lk = normalize_lang_key_cli(raw);
  const auto& m = g2p_dependency_map();
  if (const auto it = m.find(lk); it != m.end()) {
    return it->second;
  }
  const std::string alt = hyphen_to_underscore(normalize_rule_based_dialect_cli_key(lk));
  if (const auto it = m.find(alt); it != m.end()) {
    return it->second;
  }
  return std::nullopt;
}

}  // namespace

void moonshine_asset_catalog_populate_default_g2p_files(FileInformationMap& files) {
  for (const std::string& k : moonshine_asset_catalog_all_g2p_dependency_keys_union()) {
    files.set_path(k, std::filesystem::path(k));
  }
}

std::optional<std::vector<std::string>> moonshine_asset_catalog_g2p_dependency_keys(
    std::string_view lang_cli) {
  return lookup_g2p_dependency_keys(lang_cli);
}

std::vector<std::string> moonshine_asset_catalog_all_g2p_dependency_keys_union() {
  std::unordered_set<std::string> seen;
  const auto& m = g2p_dependency_map();
  for (const auto& pr : m) {
    for (const std::string& k : pr.second) {
      seen.insert(k);
    }
  }
  std::vector<std::string> out(seen.begin(), seen.end());
  std::sort(out.begin(), out.end());
  return out;
}

std::vector<std::string> moonshine_asset_catalog_all_registered_language_tags() {
  const auto& m = g2p_dependency_map();
  std::vector<std::string> tags;
  tags.reserve(m.size());
  for (const auto& pr : m) {
    tags.push_back(pr.first);
  }
  std::sort(tags.begin(), tags.end());
  return tags;
}

}  // namespace moonshine_tts
