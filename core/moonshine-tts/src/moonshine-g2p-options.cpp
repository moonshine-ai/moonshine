#include "moonshine-g2p-options.h"

#include "string-utils.h"

#include <unordered_set>

namespace moonshine_tts {

namespace {

std::optional<std::filesystem::path> optional_path_from_string(const std::string& value) {
  const std::string t = trim(value);
  if (t.empty()) {
    return std::nullopt;
  }
  return std::filesystem::path(t);
}

void set_canonical_file(FileInformationMap& files, std::string_view canonical_key,
                        const std::string& value) {
  const auto p = optional_path_from_string(value);
  if (!p) {
    files.erase_key(canonical_key);
  } else {
    files.set_path(canonical_key, *p);
  }
}

void set_override_file(FileInformationMap& files, std::string_view map_key,
                       const std::string& value) {
  const auto p = optional_path_from_string(value);
  if (!p) {
    files.erase_key(map_key);
  } else {
    files.set_path(map_key, *p);
  }
}

bool is_known_g2p_option(std::string_view key) {
  static const std::unordered_set<std::string> kKnown = {
      "g2p_root",
      "path_root",
      "model_root",
      "use_cuda",
      "spanish_with_stress",
      "spanish_narrow_obstruents",
      "german_dict_path",
      "german_with_stress",
      "german_vocoder_stress",
      "french_dict_path",
      "french_csv_dir",
      "french_with_stress",
      "french_liaison",
      "french_liaison_optional",
      "french_oov_rules",
      "french_expand_cardinal_digits",
      "dutch_dict_path",
      "dutch_with_stress",
      "dutch_vocoder_stress",
      "dutch_expand_cardinal_digits",
      "italian_dict_path",
      "italian_with_stress",
      "italian_vocoder_stress",
      "italian_expand_cardinal_digits",
      "russian_dict_path",
      "russian_with_stress",
      "russian_vocoder_stress",
      "chinese_dict_path",
      "chinese_onnx_model_dir",
      "korean_dict_path",
      "korean_expand_cardinal_digits",
      "vietnamese_dict_path",
      "japanese_dict_path",
      "japanese_onnx_model_dir",
      "arabic_onnx_model_dir",
      "arabic_dict_path",
      "portuguese_dict_path",
      "portuguese_with_stress",
      "portuguese_vocoder_stress",
      "portuguese_keep_syllable_dots",
      "portuguese_expand_cardinal_digits",
      "portuguese_apply_pt_pt_final_esh",
      "turkish_with_stress",
      "turkish_expand_cardinal_digits",
      "ukrainian_with_stress",
      "ukrainian_expand_cardinal_digits",
      "hindi_dict_path",
      "hindi_with_stress",
      "hindi_expand_cardinal_digits",
      "english_dict_path",
      "heteronym_onnx_override",
      "oov_onnx_override",
  };
  return kKnown.find(std::string(key)) != kKnown.end();
}

}  // namespace

MoonshineG2POptions::MoonshineG2POptions() {
  static const struct {
    std::string_view key;
    std::string_view path;
  } kDefaults[] = {
      {kG2pGermanDictKey, kG2pGermanDictKey},
      {kG2pFrenchDictKey, kG2pFrenchDictKey},
      {kG2pFrenchCsvDirKey, kG2pFrenchCsvDirKey},
      {kG2pDutchDictKey, kG2pDutchDictKey},
      {kG2pItalianDictKey, kG2pItalianDictKey},
      {kG2pRussianDictKey, kG2pRussianDictKey},
      {kG2pChineseDictKey, kG2pChineseDictKey},
      {kG2pChineseOnnxDirKey, kG2pChineseOnnxDirKey},
      {kG2pKoreanDictKey, kG2pKoreanDictKey},
      {kG2pVietnameseDictKey, kG2pVietnameseDictKey},
      {kG2pJapaneseDictKey, kG2pJapaneseDictKey},
      {kG2pJapaneseOnnxDirKey, kG2pJapaneseOnnxDirKey},
      {kG2pArabicOnnxDirKey, kG2pArabicOnnxDirKey},
      {kG2pArabicDictKey, kG2pArabicDictKey},
      {kG2pHindiDictKey, kG2pHindiDictKey},
      {kG2pEnglishDictKey, kG2pEnglishDictKey},
  };
  for (const auto& d : kDefaults) {
    files.set_path(d.key, std::filesystem::path(d.path));
  }
}

std::filesystem::path MoonshineG2POptions::relative_asset_path(std::string_view canonical_key) const {
  const std::string k(canonical_key);
  const auto it = files.entries.find(k);
  if (it == files.entries.end()) {
    return std::filesystem::path(canonical_key);
  }
  return it->second.path;
}

std::optional<std::filesystem::path> MoonshineG2POptions::optional_override_path(
    std::string_view map_key) const {
  const std::string k(map_key);
  const auto it = files.entries.find(k);
  if (it == files.entries.end()) {
    return std::nullopt;
  }
  if (it->second.path.empty()) {
    return std::nullopt;
  }
  return it->second.path;
}

void MoonshineG2POptions::parse_options(
    const std::vector<std::pair<std::string, std::string>>& options) {
  for (const auto& entry : options) {
    const std::string key = to_lowercase(entry.first);
    if (!is_known_g2p_option(key)) {
      throw std::runtime_error("Unknown G2P option: '" + entry.first + "'");
    }
  }

  for (const auto& entry : options) {
    const std::string& name = entry.first;
    const std::string& value = entry.second;
    const std::string key = to_lowercase(name);
    const char* v = value.c_str();

    if (key == "g2p_root" || key == "path_root" || key == "model_root") {
      g2p_root = std::filesystem::path(trim(value));
    } else if (key == "use_cuda") {
      use_cuda = bool_from_string(v);
    } else if (key == "spanish_with_stress") {
      spanish_with_stress = bool_from_string(v);
    } else if (key == "spanish_narrow_obstruents") {
      spanish_narrow_obstruents = bool_from_string(v);
    } else if (key == "german_dict_path") {
      set_canonical_file(files, kG2pGermanDictKey, value);
    } else if (key == "german_with_stress") {
      german_with_stress = bool_from_string(v);
    } else if (key == "german_vocoder_stress") {
      german_vocoder_stress = bool_from_string(v);
    } else if (key == "french_dict_path") {
      set_canonical_file(files, kG2pFrenchDictKey, value);
    } else if (key == "french_csv_dir") {
      set_canonical_file(files, kG2pFrenchCsvDirKey, value);
    } else if (key == "french_with_stress") {
      french_with_stress = bool_from_string(v);
    } else if (key == "french_liaison") {
      french_liaison = bool_from_string(v);
    } else if (key == "french_liaison_optional") {
      french_liaison_optional = bool_from_string(v);
    } else if (key == "french_oov_rules") {
      french_oov_rules = bool_from_string(v);
    } else if (key == "french_expand_cardinal_digits") {
      french_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "dutch_dict_path") {
      set_canonical_file(files, kG2pDutchDictKey, value);
    } else if (key == "dutch_with_stress") {
      dutch_with_stress = bool_from_string(v);
    } else if (key == "dutch_vocoder_stress") {
      dutch_vocoder_stress = bool_from_string(v);
    } else if (key == "dutch_expand_cardinal_digits") {
      dutch_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "italian_dict_path") {
      set_canonical_file(files, kG2pItalianDictKey, value);
    } else if (key == "italian_with_stress") {
      italian_with_stress = bool_from_string(v);
    } else if (key == "italian_vocoder_stress") {
      italian_vocoder_stress = bool_from_string(v);
    } else if (key == "italian_expand_cardinal_digits") {
      italian_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "russian_dict_path") {
      set_canonical_file(files, kG2pRussianDictKey, value);
    } else if (key == "russian_with_stress") {
      russian_with_stress = bool_from_string(v);
    } else if (key == "russian_vocoder_stress") {
      russian_vocoder_stress = bool_from_string(v);
    } else if (key == "chinese_dict_path") {
      set_canonical_file(files, kG2pChineseDictKey, value);
    } else if (key == "chinese_onnx_model_dir") {
      set_canonical_file(files, kG2pChineseOnnxDirKey, value);
    } else if (key == "korean_dict_path") {
      set_canonical_file(files, kG2pKoreanDictKey, value);
    } else if (key == "korean_expand_cardinal_digits") {
      korean_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "vietnamese_dict_path") {
      set_canonical_file(files, kG2pVietnameseDictKey, value);
    } else if (key == "japanese_dict_path") {
      set_canonical_file(files, kG2pJapaneseDictKey, value);
    } else if (key == "japanese_onnx_model_dir") {
      set_canonical_file(files, kG2pJapaneseOnnxDirKey, value);
    } else if (key == "arabic_onnx_model_dir") {
      set_canonical_file(files, kG2pArabicOnnxDirKey, value);
    } else if (key == "arabic_dict_path") {
      set_canonical_file(files, kG2pArabicDictKey, value);
    } else if (key == "portuguese_dict_path") {
      set_override_file(files, kG2pPortugueseDictOverrideKey, value);
    } else if (key == "portuguese_with_stress") {
      portuguese_with_stress = bool_from_string(v);
    } else if (key == "portuguese_vocoder_stress") {
      portuguese_vocoder_stress = bool_from_string(v);
    } else if (key == "portuguese_keep_syllable_dots") {
      portuguese_keep_syllable_dots = bool_from_string(v);
    } else if (key == "portuguese_expand_cardinal_digits") {
      portuguese_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "portuguese_apply_pt_pt_final_esh") {
      portuguese_apply_pt_pt_final_esh = bool_from_string(v);
    } else if (key == "turkish_with_stress") {
      turkish_with_stress = bool_from_string(v);
    } else if (key == "turkish_expand_cardinal_digits") {
      turkish_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "ukrainian_with_stress") {
      ukrainian_with_stress = bool_from_string(v);
    } else if (key == "ukrainian_expand_cardinal_digits") {
      ukrainian_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "hindi_dict_path") {
      set_canonical_file(files, kG2pHindiDictKey, value);
    } else if (key == "hindi_with_stress") {
      hindi_with_stress = bool_from_string(v);
    } else if (key == "hindi_expand_cardinal_digits") {
      hindi_expand_cardinal_digits = bool_from_string(v);
    } else if (key == "english_dict_path") {
      set_canonical_file(files, kG2pEnglishDictKey, value);
    } else if (key == "heteronym_onnx_override") {
      set_override_file(files, kG2pHeteronymOnnxOverrideKey, value);
    } else if (key == "oov_onnx_override") {
      set_override_file(files, kG2pOovOnnxOverrideKey, value);
    } else {
      throw std::logic_error("MoonshineG2POptions::parse_options: unhandled option '" + name + "'");
    }
  }
}

}  // namespace moonshine_tts
