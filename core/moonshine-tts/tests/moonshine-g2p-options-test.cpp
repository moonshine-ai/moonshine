#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-g2p-options.h"

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

using moonshine_tts::MoonshineG2POptions;
using moonshine_tts::kG2pEnglishDictKey;
using moonshine_tts::kG2pFrenchCsvDirKey;
using moonshine_tts::kG2pFrenchDictKey;
using moonshine_tts::kG2pGermanDictKey;
using moonshine_tts::kG2pOovOnnxOverrideKey;
using moonshine_tts::kG2pPortugueseDictOverrideKey;

TEST_CASE("MoonshineG2POptions default constructor seeds canonical file keys") {
  MoonshineG2POptions o;
  CHECK(o.files.contains(std::string(kG2pGermanDictKey)));
  CHECK(o.relative_asset_path(kG2pGermanDictKey) == std::filesystem::path{kG2pGermanDictKey});
  CHECK(o.relative_asset_path(kG2pFrenchDictKey) == std::filesystem::path{kG2pFrenchDictKey});
  CHECK(o.relative_asset_path(kG2pFrenchCsvDirKey) == std::filesystem::path{kG2pFrenchCsvDirKey});
  CHECK(o.relative_asset_path(kG2pEnglishDictKey) == std::filesystem::path{kG2pEnglishDictKey});
  CHECK_FALSE(o.optional_override_path(kG2pPortugueseDictOverrideKey).has_value());
  CHECK_FALSE(o.optional_override_path(kG2pOovOnnxOverrideKey).has_value());
}

TEST_CASE("MoonshineG2POptions relative_asset_path falls back when key absent") {
  MoonshineG2POptions o;
  o.files.erase_key(std::string(kG2pGermanDictKey));
  CHECK(o.relative_asset_path(kG2pGermanDictKey) == std::filesystem::path{kG2pGermanDictKey});
}

TEST_CASE("MoonshineG2POptions parse_options rejects unknown keys") {
  MoonshineG2POptions o;
  CHECK_THROWS_AS(o.parse_options({{"not_a_valid_g2p_option", "1"}}), std::runtime_error);
}

TEST_CASE("MoonshineG2POptions parse_options accepts every known option") {
  MoonshineG2POptions o;
  std::vector<std::pair<std::string, std::string>> all;
  all.emplace_back("g2p_root", "/tmp/g2p");
  all.emplace_back("path_root", "/tmp/p");
  all.emplace_back("model_root", "/tmp/m");
  all.emplace_back("use_cuda", "false");
  all.emplace_back("spanish_with_stress", "true");
  all.emplace_back("spanish_narrow_obstruents", "true");
  all.emplace_back("german_dict_path", "custom/de.tsv");
  all.emplace_back("german_with_stress", "true");
  all.emplace_back("german_vocoder_stress", "false");
  all.emplace_back("french_dict_path", "custom/fr.tsv");
  all.emplace_back("french_csv_dir", "custom/frdir");
  all.emplace_back("french_with_stress", "true");
  all.emplace_back("french_liaison", "true");
  all.emplace_back("french_liaison_optional", "false");
  all.emplace_back("french_oov_rules", "true");
  all.emplace_back("french_expand_cardinal_digits", "true");
  all.emplace_back("dutch_dict_path", "custom/nl.tsv");
  all.emplace_back("dutch_with_stress", "true");
  all.emplace_back("dutch_vocoder_stress", "true");
  all.emplace_back("dutch_expand_cardinal_digits", "false");
  all.emplace_back("italian_dict_path", "custom/it.tsv");
  all.emplace_back("italian_with_stress", "true");
  all.emplace_back("italian_vocoder_stress", "true");
  all.emplace_back("italian_expand_cardinal_digits", "true");
  all.emplace_back("russian_dict_path", "custom/ru.tsv");
  all.emplace_back("russian_with_stress", "true");
  all.emplace_back("russian_vocoder_stress", "false");
  all.emplace_back("chinese_dict_path", "custom/zh.tsv");
  all.emplace_back("chinese_onnx_model_dir", "custom/zh_onnx");
  all.emplace_back("korean_dict_path", "custom/ko.tsv");
  all.emplace_back("korean_expand_cardinal_digits", "true");
  all.emplace_back("vietnamese_dict_path", "custom/vi.tsv");
  all.emplace_back("japanese_dict_path", "custom/ja.tsv");
  all.emplace_back("japanese_onnx_model_dir", "custom/ja_onnx");
  all.emplace_back("arabic_onnx_model_dir", "custom/ar_onnx");
  all.emplace_back("arabic_dict_path", "custom/ar.tsv");
  all.emplace_back("portuguese_dict_path", "custom/pt.tsv");
  all.emplace_back("portuguese_with_stress", "true");
  all.emplace_back("portuguese_vocoder_stress", "true");
  all.emplace_back("portuguese_keep_syllable_dots", "false");
  all.emplace_back("portuguese_expand_cardinal_digits", "true");
  all.emplace_back("portuguese_apply_pt_pt_final_esh", "true");
  all.emplace_back("turkish_with_stress", "true");
  all.emplace_back("turkish_expand_cardinal_digits", "false");
  all.emplace_back("ukrainian_with_stress", "true");
  all.emplace_back("ukrainian_expand_cardinal_digits", "false");
  all.emplace_back("hindi_dict_path", "custom/hi.tsv");
  all.emplace_back("hindi_with_stress", "true");
  all.emplace_back("hindi_expand_cardinal_digits", "true");
  all.emplace_back("english_dict_path", "custom/en.tsv");
  all.emplace_back("oov_onnx_override", "custom/oov.onnx");

  CHECK_NOTHROW(o.parse_options(all));

  CHECK(o.g2p_root == std::filesystem::path{"/tmp/m"});
  CHECK(o.use_cuda == false);
  CHECK(o.german_vocoder_stress == false);
  CHECK(o.relative_asset_path(kG2pGermanDictKey) == std::filesystem::path{"custom/de.tsv"});
  CHECK(o.relative_asset_path(kG2pFrenchCsvDirKey) == std::filesystem::path{"custom/frdir"});
  CHECK(*o.optional_override_path(kG2pPortugueseDictOverrideKey) ==
        std::filesystem::path{"custom/pt.tsv"});
  CHECK(*o.optional_override_path(kG2pOovOnnxOverrideKey) ==
        std::filesystem::path{"custom/oov.onnx"});
}

TEST_CASE("MoonshineG2POptions parse_options empty path clears canonical entry") {
  MoonshineG2POptions o;
  o.parse_options({{"german_dict_path", " "}});
  CHECK_FALSE(o.files.contains(std::string(kG2pGermanDictKey)));
  CHECK(o.relative_asset_path(kG2pGermanDictKey) == std::filesystem::path{kG2pGermanDictKey});
}

TEST_CASE("MoonshineG2POptions option names are case-insensitive") {
  MoonshineG2POptions o;
  CHECK_NOTHROW(o.parse_options({{"GERMAN_WITH_STRESS", "false"}}));
  CHECK(o.german_with_stress == false);
}
