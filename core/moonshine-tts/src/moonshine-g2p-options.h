#ifndef MOONSHINE_TTS_MOONSHINE_G2P_OPTIONS_H
#define MOONSHINE_TTS_MOONSHINE_G2P_OPTIONS_H

#include "file-information.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moonshine_tts {

/// ``<bundle_dir_key>/<filename>`` for ``FileInformationMap`` / ``read_*_asset``.
inline std::string g2p_bundle_file_key(std::string_view bundle_dir_key, std::string_view filename) {
  std::string s;
  s.reserve(bundle_dir_key.size() + 1 + filename.size());
  s.append(bundle_dir_key);
  s.push_back('/');
  s.append(filename);
  return s;
}

/// Canonical ``FileInformationMap`` keys for bundled relative assets (default path = key).
inline constexpr std::string_view kG2pGermanDictKey = "de/dict.tsv";
inline constexpr std::string_view kG2pFrenchDictKey = "fr/dict.tsv";
/// Directory override for French POS CSVs (disk scan); CDN keys use the ``fr/*.csv`` paths below.
inline constexpr std::string_view kG2pFrenchCsvDirKey = "fr";
inline constexpr std::string_view kG2pFrenchPosAdjKey = "fr/adj.csv";
inline constexpr std::string_view kG2pFrenchPosAdvKey = "fr/adv.csv";
inline constexpr std::string_view kG2pFrenchPosConjKey = "fr/conj.csv";
inline constexpr std::string_view kG2pFrenchPosDetKey = "fr/det.csv";
inline constexpr std::string_view kG2pFrenchPosNounKey = "fr/noun.csv";
inline constexpr std::string_view kG2pFrenchPosPrepKey = "fr/prep.csv";
inline constexpr std::string_view kG2pFrenchPosPronKey = "fr/pron.csv";
inline constexpr std::string_view kG2pFrenchPosVerbKey = "fr/verb.csv";
inline constexpr std::string_view kG2pDutchDictKey = "nl/dict.tsv";
inline constexpr std::string_view kG2pItalianDictKey = "it/dict.tsv";
inline constexpr std::string_view kG2pRussianDictKey = "ru/dict.tsv";
inline constexpr std::string_view kG2pChineseDictKey = "zh_hans/dict.tsv";
inline constexpr std::string_view kG2pChineseOnnxDirKey = "zh_hans/roberta_chinese_base_upos_onnx";
inline constexpr std::string_view kG2pKoreanDictKey = "ko/dict.tsv";
inline constexpr std::string_view kG2pKoreanOnnxDirKey = "ko/roberta_korean_morph_upos_onnx";
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
/// Optional English assets (same ``FileInformationMap`` / memory pattern as lexicons).
inline constexpr std::string_view kG2pEnglishG2pConfigKey = "en_us/g2p-config.json";
inline constexpr std::string_view kG2pOovOnnxOverrideKey = "oov_onnx_override";
/// UTF-8 ``onnx-config.json`` when ``oov_onnx_override`` is used from memory.
inline constexpr std::string_view kG2pOovOnnxConfigOverrideKey = "oov_onnx_config";

/// Chinese tokenizer + ONNX bundle files (under ``kG2pChineseOnnxDirKey``).
inline constexpr std::string_view kG2pChineseOnnxMetaKey = "zh_hans/roberta_chinese_base_upos_onnx/meta.json";
inline constexpr std::string_view kG2pChineseOnnxVocabKey = "zh_hans/roberta_chinese_base_upos_onnx/vocab.txt";
inline constexpr std::string_view kG2pChineseOnnxTokenizerConfigKey =
    "zh_hans/roberta_chinese_base_upos_onnx/tokenizer_config.json";
inline constexpr std::string_view kG2pChineseOnnxModelKey = "zh_hans/roberta_chinese_base_upos_onnx/model.ort";

inline constexpr std::string_view kG2pJapaneseOnnxMetaKey = "ja/roberta_japanese_char_luw_upos_onnx/meta.json";
inline constexpr std::string_view kG2pJapaneseOnnxVocabKey = "ja/roberta_japanese_char_luw_upos_onnx/vocab.txt";
inline constexpr std::string_view kG2pJapaneseOnnxTokenizerConfigKey =
    "ja/roberta_japanese_char_luw_upos_onnx/tokenizer_config.json";
inline constexpr std::string_view kG2pJapaneseOnnxModelKey = "ja/roberta_japanese_char_luw_upos_onnx/model.ort";

inline constexpr std::string_view kG2pKoreanOnnxMetaKey = "ko/roberta_korean_morph_upos_onnx/meta.json";
inline constexpr std::string_view kG2pKoreanOnnxVocabKey = "ko/roberta_korean_morph_upos_onnx/vocab.txt";
inline constexpr std::string_view kG2pKoreanOnnxTokenizerConfigKey =
    "ko/roberta_korean_morph_upos_onnx/tokenizer_config.json";
inline constexpr std::string_view kG2pKoreanOnnxModelKey = "ko/roberta_korean_morph_upos_onnx/model.ort";

inline constexpr std::string_view kG2pArabicOnnxMetaKey = "ar_msa/arabertv02_tashkeel_fadel_onnx/meta.json";
inline constexpr std::string_view kG2pArabicOnnxVocabKey = "ar_msa/arabertv02_tashkeel_fadel_onnx/vocab.txt";
inline constexpr std::string_view kG2pArabicOnnxTokenizerConfigKey =
    "ar_msa/arabertv02_tashkeel_fadel_onnx/tokenizer_config.json";
inline constexpr std::string_view kG2pArabicOnnxModelKey = "ar_msa/arabertv02_tashkeel_fadel_onnx/model.ort";

/// English OOV ONNX next to ``en_us/g2p-config.json`` (``.onnx`` in-tree; ``.ort`` sibling allowed on disk).
inline constexpr std::string_view kG2pEnglishOovModelKey = "en_us/oov/model.onnx";
inline constexpr std::string_view kG2pEnglishOovOnnxConfigKey = "en_us/oov/onnx-config.json";

/// Options for constructing ``MoonshineG2P`` (rule-engine paths and toggles; optional English OOV ONNX
/// overrides).
///
/// Lexicon and ONNX paths live in ``files`` under **canonical keys** equal to their default relative
/// paths (e.g. ``fr/dict.tsv``). Set ``g2p_root`` to the directory that contains those subpaths, or leave
/// it empty to resolve those relative paths against the process current working directory only.
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

  /// Non-empty path stored under ``map_key`` (used for overrides such as ``oov_onnx_override``).
  std::optional<std::filesystem::path> optional_override_path(
      std::string_view map_key) const;

  /// True if the asset has a client buffer or exists on disk under ``g2p_root`` (canonical key or map key).
  bool asset_is_available(std::string_view canonical_key) const;

  /// Full file contents via ``FileInformation::load`` (memory buffer or disk). Does not mutate ``files``.
  std::vector<uint8_t> read_binary_asset(std::string_view canonical_key) const;

  /// ``read_binary_asset`` interpreted as UTF-8 bytes (no BOM handling).
  std::string read_utf8_asset(std::string_view canonical_key) const;

  void parse_options(const std::vector<std::pair<std::string, std::string>>& options);
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_MOONSHINE_G2P_OPTIONS_H
