#ifndef MOONSHINE_TTS_MOONSHINE_G2P_OPTIONS_H
#define MOONSHINE_TTS_MOONSHINE_G2P_OPTIONS_H

#include "file-information.h"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moonshine_tts {

/// Canonical ``FileInformationMap`` keys for bundled relative assets (default path = key).
inline constexpr std::string_view kG2pGermanDictKey = "de/dict.tsv";
inline constexpr std::string_view kG2pFrenchDictKey = "fr/dict.tsv";
inline constexpr std::string_view kG2pFrenchCsvDirKey = "fr";
inline constexpr std::string_view kG2pDutchDictKey = "nl/dict.tsv";
inline constexpr std::string_view kG2pItalianDictKey = "it/dict.tsv";
inline constexpr std::string_view kG2pRussianDictKey = "ru/dict.tsv";
inline constexpr std::string_view kG2pChineseDictKey = "zh_hans/dict.tsv";
inline constexpr std::string_view kG2pChineseOnnxDirKey = "zh_hans/roberta_chinese_base_upos_onnx";
inline constexpr std::string_view kG2pKoreanDictKey = "ko/dict.tsv";
inline constexpr std::string_view kG2pVietnameseDictKey = "vi/dict.tsv";
inline constexpr std::string_view kG2pJapaneseDictKey = "ja/dict.tsv";
inline constexpr std::string_view kG2pJapaneseOnnxDirKey = "ja/roberta_japanese_char_luw_upos_onnx";
inline constexpr std::string_view kG2pArabicOnnxDirKey = "ar_msa/arabertv02_tashkeel_fadel_onnx";
inline constexpr std::string_view kG2pArabicDictKey = "ar_msa/dict.tsv";
inline constexpr std::string_view kG2pHindiDictKey = "hi/dict.tsv";
inline constexpr std::string_view kG2pEnglishDictKey = "en_us/dict_filtered_heteronyms.tsv";
inline constexpr std::string_view kG2pPtBrDictKey = "pt_br/dict.tsv";
inline constexpr std::string_view kG2pPtPtDictKey = "pt_pt/dict.tsv";
/// Map key for ``parse_options`` / CLI override (not a bundled default path).
inline constexpr std::string_view kG2pPortugueseDictOverrideKey = "portuguese_dict_path";
inline constexpr std::string_view kG2pHeteronymOnnxOverrideKey = "heteronym_onnx_override";
inline constexpr std::string_view kG2pOovOnnxOverrideKey = "oov_onnx_override";

/// Options for constructing ``MoonshineG2P`` (rule-engine paths and toggles; optional heteronym/OOV
/// ONNX overrides for English).
///
/// Lexicon and ONNX paths live in ``files`` under **canonical keys** equal to their default relative
/// paths (e.g. ``fr/dict.tsv``). Set ``g2p_root`` to the directory that contains those subpaths.
struct MoonshineG2POptions {
  MoonshineG2POptions();

  std::filesystem::path g2p_root{};
  bool use_cuda = false;
  bool spanish_with_stress = true;
  bool spanish_narrow_obstruents = true;
  bool german_with_stress = true;
  bool german_vocoder_stress = true;
  bool french_with_stress = true;
  bool french_liaison = true;
  bool french_liaison_optional = true;
  bool french_oov_rules = true;
  bool french_expand_cardinal_digits = true;
  bool dutch_with_stress = true;
  bool dutch_vocoder_stress = true;
  bool dutch_expand_cardinal_digits = true;
  bool italian_with_stress = true;
  bool italian_vocoder_stress = true;
  bool italian_expand_cardinal_digits = true;
  bool russian_with_stress = true;
  bool russian_vocoder_stress = true;
  bool korean_expand_cardinal_digits = true;
  bool portuguese_with_stress = true;
  bool portuguese_vocoder_stress = true;
  bool portuguese_keep_syllable_dots = false;
  bool portuguese_expand_cardinal_digits = true;
  bool portuguese_apply_pt_pt_final_esh = true;
  bool turkish_with_stress = true;
  bool turkish_expand_cardinal_digits = true;
  bool ukrainian_with_stress = true;
  bool ukrainian_expand_cardinal_digits = true;
  bool hindi_with_stress = true;
  bool hindi_expand_cardinal_digits = true;

  FileInformationMap files;

  /// Relative path (under ``g2p_root``) for a bundled asset. If ``files`` has no entry for
  /// ``canonical_key``, returns ``std::filesystem::path(canonical_key)``.
  std::filesystem::path relative_asset_path(std::string_view canonical_key) const;

  /// Non-empty path stored under ``map_key`` (used for overrides such as ``heteronym_onnx_override``).
  std::optional<std::filesystem::path> optional_override_path(
      std::string_view map_key) const;

  void parse_options(const std::vector<std::pair<std::string, std::string>>& options);
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_MOONSHINE_G2P_OPTIONS_H
