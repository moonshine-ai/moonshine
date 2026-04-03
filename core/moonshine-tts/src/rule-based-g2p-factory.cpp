#include "rule-based-g2p-factory.h"

#include "builtin-cpp-data-root.h"
#include "g2p-path.h"
#include "moonshine-g2p-options.h"
#include "rule-based-g2p.h"
#include "dutch.h"
#include "english.h"
#include "french.h"
#include "german.h"
#include "chinese-onnx-g2p.h"
#include "chinese.h"
#include "korean.h"
#include "vietnamese.h"
#include "japanese.h"
#include "arabic.h"
#include "italian.h"
#include "portuguese.h"
#include "russian.h"
#include "spanish.h"
#include "turkish.h"
#include "ukrainian.h"
#include "hindi.h"
#include "utf8-utils.h"

#include <cctype>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.h>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace moonshine_tts {
namespace {

std::filesystem::path resolve_french_dict_path(const MoonshineG2POptions& opt) {
  return resolve_path_under_root(opt.g2p_root, opt.relative_asset_path(kG2pFrenchDictKey));
}

std::filesystem::path resolve_french_csv_dir(const MoonshineG2POptions& opt) {
  return resolve_path_under_root(opt.g2p_root, opt.relative_asset_path(kG2pFrenchCsvDirKey));
}

std::string normalize_spanish_dialect_cli_key(std::string_view raw) {
  std::string s = normalize_rule_based_dialect_cli_key(raw);
  if (s.size() >= 3 && s[0] == 'e' && s[1] == 's' && s[2] == '-') {
    size_t i = 3;
    while (i < s.size() && s[i] != '-') {
      if (std::isalpha(static_cast<unsigned char>(s[i])) != 0) {
        s[i] = static_cast<char>(std::toupper(static_cast<unsigned char>(s[i])));
      }
      ++i;
    }
    if (i < s.size() && s[i] == '-') {
      ++i;
      while (i < s.size()) {
        s[i] = static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
        ++i;
      }
    }
  }
  return s;
}

bool file_looks_like_git_lfs_pointer(const std::filesystem::path& p) {
  std::ifstream in(p);
  std::string line;
  if (!std::getline(in, line)) {
    return false;
  }
  static constexpr std::string_view kPrefix = "version https://git-lfs.github.com/spec/v1";
  return line.size() >= kPrefix.size() && line.compare(0, kPrefix.size(), kPrefix) == 0;
}

std::optional<RuleBasedG2pInstance> try_english(std::string_view trimmed,
                                                 const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_english_rules(trimmed)) {
    return std::nullopt;
  }
  std::filesystem::path dict_tsv = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pEnglishDictKey));
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    const std::filesystem::path bundled = resolve_path_under_root(
        builtin_cpp_data_root(), options.relative_asset_path(kG2pEnglishDictKey));
    if (std::filesystem::is_regular_file(bundled)) {
      dict_tsv = bundled;
    } else {
      dict_tsv = resolve_english_dict_path(options.g2p_root);
    }
  }
  if (!std::filesystem::is_regular_file(dict_tsv)) {
    throw std::runtime_error(
        "English G2P: lexicon not found at " + dict_tsv.generic_string() +
        " (set MoonshineG2POptions::files / english_dict_path)");
  }
  const std::filesystem::path data_root = dict_tsv.parent_path();
  std::optional<std::filesystem::path> het_onnx;
  std::optional<std::filesystem::path> oov_onnx;
  const std::filesystem::path g2p_cfg = data_root / "g2p-config.json";
  if (std::filesystem::is_regular_file(g2p_cfg)) {
    if (file_looks_like_git_lfs_pointer(g2p_cfg)) {
      throw std::runtime_error(
          "English G2P: " + g2p_cfg.generic_string() +
          " is a Git LFS pointer stub, not JSON. From the moonshine-tts directory run: git lfs pull");
    }
    std::ifstream cfg_in(g2p_cfg);
    const nlohmann::json j = nlohmann::json::parse(cfg_in);
    if (j.value("uses_heteronym_model", false)) {
      het_onnx = data_root / "heteronym" / "model.onnx";
    }
    if (j.value("uses_oov_model", false)) {
      oov_onnx = data_root / "oov" / "model.onnx";
    }
  }
  if (const auto het = options.optional_override_path(kG2pHeteronymOnnxOverrideKey)) {
    het_onnx = resolve_path_under_root(options.g2p_root, *het);
  }
  if (const auto oov = options.optional_override_path(kG2pOovOnnxOverrideKey)) {
    oov_onnx = resolve_path_under_root(options.g2p_root, *oov);
  }
  if (het_onnx && !std::filesystem::is_regular_file(*het_onnx)) {
    het_onnx.reset();
  }
  if (oov_onnx && !std::filesystem::is_regular_file(*oov_onnx)) {
    oov_onnx.reset();
  }
  const std::filesystem::path homograph_json = data_root / "heteronym" / "homograph_index.json";
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "en_us";
  out.kind = RuleBasedG2pKind::English;
  out.engine = std::make_unique<EnglishRuleG2p>(dict_tsv, homograph_json, het_onnx, oov_onnx,
                                                options.use_cuda);
  return out;
}

std::optional<RuleBasedG2pInstance> try_spanish(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  const std::string spanish_key = normalize_spanish_dialect_cli_key(trimmed);
  try {
    SpanishDialect d =
        spanish_dialect_from_cli_id(spanish_key, options.spanish_narrow_obstruents);
    RuleBasedG2pInstance out;
    out.canonical_dialect_id = std::move(d.id);
    out.kind = RuleBasedG2pKind::Spanish;
    out.engine = std::make_unique<SpanishRuleG2p>(std::move(d), options.spanish_with_stress);
    return out;
  } catch (const std::invalid_argument&) {
    return std::nullopt;
  }
}

std::optional<RuleBasedG2pInstance> try_german(std::string_view trimmed,
                                               const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_german_rules(trimmed)) {
    return std::nullopt;
  }
  std::filesystem::path gdict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pGermanDictKey));
  if (!std::filesystem::is_regular_file(gdict)) {
    const std::filesystem::path bundled =
        resolve_path_under_root(builtin_cpp_data_root(), std::filesystem::path{"de/dict.tsv"});
    if (std::filesystem::is_regular_file(bundled)) {
      gdict = bundled;
    }
  }
  if (!std::filesystem::is_regular_file(gdict)) {
    throw std::runtime_error(
        "German G2P: lexicon not found at " + gdict.generic_string() +
        " (set MoonshineG2POptions::files / german_dict_path)");
  }
  GermanRuleG2p g(gdict, GermanRuleG2p::Options{.with_stress = options.german_with_stress,
                                                 .vocoder_stress = options.german_vocoder_stress});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "de-DE";
  out.kind = RuleBasedG2pKind::German;
  out.engine = std::make_unique<GermanRuleG2p>(std::move(g));
  return out;
}

std::optional<RuleBasedG2pInstance> try_french(std::string_view trimmed,
                                               const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_french_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path fdict = resolve_french_dict_path(options);
  if (!std::filesystem::is_regular_file(fdict)) {
    throw std::runtime_error(
        "French G2P: lexicon not found at " + fdict.generic_string() +
        " (set MoonshineG2POptions::files / french_dict_path)");
  }
  const std::filesystem::path fcsv = resolve_french_csv_dir(options);
  FrenchRuleG2p::Options fo;
  fo.with_stress = options.french_with_stress;
  fo.liaison = options.french_liaison;
  fo.liaison_optional = options.french_liaison_optional;
  fo.oov_rules = options.french_oov_rules;
  fo.expand_cardinal_digits = options.french_expand_cardinal_digits;
  FrenchRuleG2p fr(fdict, fcsv, fo);
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "fr-FR";
  out.kind = RuleBasedG2pKind::French;
  out.engine = std::make_unique<FrenchRuleG2p>(std::move(fr));
  return out;
}

std::optional<RuleBasedG2pInstance> try_dutch(std::string_view trimmed,
                                              const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_dutch_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path ndict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pDutchDictKey));
  if (!std::filesystem::is_regular_file(ndict)) {
    throw std::runtime_error(
        "Dutch G2P: lexicon not found at " + ndict.generic_string() +
        " (set MoonshineG2POptions::files / dutch_dict_path)");
  }
  DutchRuleG2p dutch(ndict, DutchRuleG2p::Options{.with_stress = options.dutch_with_stress,
                                                  .vocoder_stress = options.dutch_vocoder_stress,
                                                  .expand_cardinal_digits =
                                                      options.dutch_expand_cardinal_digits});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "nl-NL";
  out.kind = RuleBasedG2pKind::Dutch;
  out.engine = std::make_unique<DutchRuleG2p>(std::move(dutch));
  return out;
}

std::optional<RuleBasedG2pInstance> try_italian(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_italian_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path idict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pItalianDictKey));
  if (!std::filesystem::is_regular_file(idict)) {
    throw std::runtime_error(
        "Italian G2P: lexicon not found at " + idict.generic_string() +
        " (set MoonshineG2POptions::files / italian_dict_path)");
  }
  ItalianRuleG2p it(idict, ItalianRuleG2p::Options{.with_stress = options.italian_with_stress,
                                                  .vocoder_stress = options.italian_vocoder_stress,
                                                  .expand_cardinal_digits =
                                                      options.italian_expand_cardinal_digits});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "it-IT";
  out.kind = RuleBasedG2pKind::Italian;
  out.engine = std::make_unique<ItalianRuleG2p>(std::move(it));
  return out;
}

std::optional<RuleBasedG2pInstance> try_russian(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_russian_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path rdict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pRussianDictKey));
  if (!std::filesystem::is_regular_file(rdict)) {
    throw std::runtime_error(
        "Russian G2P: lexicon not found at " + rdict.generic_string() +
        " (set MoonshineG2POptions::files / russian_dict_path)");
  }
  RussianRuleG2p ru(rdict, RussianRuleG2p::Options{.with_stress = options.russian_with_stress,
                                                     .vocoder_stress = options.russian_vocoder_stress});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "ru-RU";
  out.kind = RuleBasedG2pKind::Russian;
  out.engine = std::make_unique<RussianRuleG2p>(std::move(ru));
  return out;
}

std::optional<RuleBasedG2pInstance> try_chinese(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_chinese_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path mdir = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pChineseOnnxDirKey));
  const auto onnx = mdir / "model.onnx";
  if (!std::filesystem::is_regular_file(onnx)) {
    throw std::runtime_error(
        "Chinese G2P: ONNX bundle not found at " + onnx.generic_string() +
        " (set MoonshineG2POptions::files / chinese_onnx_model_dir or export "
        "KoichiYasuoka/chinese-roberta-base-upos to data/zh_hans/roberta_chinese_base_upos_onnx/)");
  }
  const std::filesystem::path cdict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pChineseDictKey));
  if (!std::filesystem::is_regular_file(cdict)) {
    throw std::runtime_error(
        "Chinese G2P: lexicon not found at " + cdict.generic_string() +
        " (set MoonshineG2POptions::files / chinese_dict_path)");
  }
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "zh-Hans";
  out.kind = RuleBasedG2pKind::Chinese;
  out.engine = std::make_unique<ChineseOnnxRuleG2p>(mdir, cdict, options.use_cuda);
  return out;
}

std::optional<RuleBasedG2pInstance> try_korean(std::string_view trimmed,
                                               const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_korean_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path kdict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pKoreanDictKey));
  if (!std::filesystem::is_regular_file(kdict)) {
    throw std::runtime_error(
        "Korean G2P: lexicon not found at " + kdict.generic_string() +
        " (set MoonshineG2POptions::files / korean_dict_path)");
  }
  KoreanRuleG2p::Options ko;
  ko.expand_cardinal_digits = options.korean_expand_cardinal_digits;
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "ko-KR";
  out.kind = RuleBasedG2pKind::Korean;
  out.engine = std::make_unique<KoreanRuleG2p>(kdict, ko);
  return out;
}

std::optional<RuleBasedG2pInstance> try_vietnamese(std::string_view trimmed,
                                                 const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_vietnamese_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path vdict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pVietnameseDictKey));
  if (!std::filesystem::is_regular_file(vdict)) {
    throw std::runtime_error(
        "Vietnamese G2P: lexicon not found at " + vdict.generic_string() +
        " (set MoonshineG2POptions::files / vietnamese_dict_path)");
  }
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "vi-VN";
  out.kind = RuleBasedG2pKind::Vietnamese;
  out.engine = std::make_unique<VietnameseRuleG2p>(vdict);
  return out;
}

std::optional<RuleBasedG2pInstance> try_japanese(std::string_view trimmed,
                                                 const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_japanese_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path mdir = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pJapaneseOnnxDirKey));
  const std::filesystem::path jdict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pJapaneseDictKey));
  const auto onnx = mdir / "model.onnx";
  if (!std::filesystem::is_regular_file(onnx)) {
    throw std::runtime_error(
        "Japanese G2P: ONNX bundle not found at " + onnx.generic_string() +
        " (set MoonshineG2POptions::files / japanese_onnx_model_dir or export the char-LUW model to "
        "data/ja/roberta_japanese_char_luw_upos_onnx/)");
  }
  if (!std::filesystem::is_regular_file(jdict)) {
    throw std::runtime_error(
        "Japanese G2P: lexicon not found at " + jdict.generic_string() +
        " (set MoonshineG2POptions::files / japanese_dict_path)");
  }
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "ja-JP";
  out.kind = RuleBasedG2pKind::Japanese;
  out.engine = std::make_unique<JapaneseRuleG2p>(mdir, jdict, options.use_cuda);
  return out;
}

std::optional<RuleBasedG2pInstance> try_arabic(std::string_view trimmed,
                                             const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_arabic_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path mdir = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pArabicOnnxDirKey));
  const auto onnx = mdir / "model.onnx";
  if (!std::filesystem::is_regular_file(onnx)) {
    throw std::runtime_error(
        "Arabic G2P: ONNX bundle not found at " + onnx.generic_string() +
        " (set MoonshineG2POptions::files / arabic_onnx_model_dir or run "
        "scripts/export_arabic_msa_diacritizer_onnx.py)");
  }
  const std::filesystem::path adict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pArabicDictKey));
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "ar-MSA";
  out.kind = RuleBasedG2pKind::Arabic;
  out.engine = std::make_unique<ArabicRuleG2p>(mdir, adict, options.use_cuda);
  return out;
}

std::optional<RuleBasedG2pInstance> try_turkish(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_turkish_rules(trimmed)) {
    return std::nullopt;
  }
  TurkishRuleG2p::Options to;
  to.with_stress = options.turkish_with_stress;
  to.expand_cardinal_digits = options.turkish_expand_cardinal_digits;
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "tr-TR";
  out.kind = RuleBasedG2pKind::Turkish;
  out.engine = std::make_unique<TurkishRuleG2p>(std::move(to));
  return out;
}

std::optional<RuleBasedG2pInstance> try_ukrainian(std::string_view trimmed,
                                                const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_ukrainian_rules(trimmed)) {
    return std::nullopt;
  }
  UkrainianRuleG2p::Options uo;
  uo.with_stress = options.ukrainian_with_stress;
  uo.expand_cardinal_digits = options.ukrainian_expand_cardinal_digits;
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "uk-UA";
  out.kind = RuleBasedG2pKind::Ukrainian;
  out.engine = std::make_unique<UkrainianRuleG2p>(std::move(uo));
  return out;
}

std::optional<RuleBasedG2pInstance> try_hindi(std::string_view trimmed,
                                            const MoonshineG2POptions& options) {
  if (!dialect_resolves_to_hindi_rules(trimmed)) {
    return std::nullopt;
  }
  const std::filesystem::path hdict = resolve_path_under_root(
      options.g2p_root, options.relative_asset_path(kG2pHindiDictKey));
  if (!std::filesystem::is_regular_file(hdict)) {
    throw std::runtime_error(
        "Hindi G2P: lexicon not found at " + hdict.generic_string() +
        " (set MoonshineG2POptions::files / hindi_dict_path)");
  }
  HindiRuleG2p::Options ho;
  ho.with_stress = options.hindi_with_stress;
  ho.expand_cardinal_digits = options.hindi_expand_cardinal_digits;
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = "hi-IN";
  out.kind = RuleBasedG2pKind::Hindi;
  out.engine = std::make_unique<HindiRuleG2p>(hdict, ho);
  return out;
}

std::optional<RuleBasedG2pInstance> try_portuguese(std::string_view trimmed,
                                                  const MoonshineG2POptions& options) {
  const bool want_pt_br = dialect_resolves_to_brazilian_portuguese_rules(trimmed);
  const bool want_pt_pt = dialect_resolves_to_portugal_rules(trimmed);
  if (!want_pt_br && !want_pt_pt) {
    return std::nullopt;
  }
  const bool is_portugal = want_pt_pt && !want_pt_br;
  std::filesystem::path rel_pt;
  if (const auto o = options.optional_override_path(kG2pPortugueseDictOverrideKey)) {
    rel_pt = *o;
  } else {
    rel_pt = std::filesystem::path(is_portugal ? kG2pPtPtDictKey : kG2pPtBrDictKey);
  }
  const std::filesystem::path pdict = resolve_path_under_root(options.g2p_root, rel_pt);
  if (!std::filesystem::is_regular_file(pdict)) {
    throw std::runtime_error(
        "Portuguese G2P: lexicon not found at " + pdict.generic_string() +
        " (set MoonshineG2POptions::files / portuguese_dict_path)");
  }
  PortugueseRuleG2p pt(
      pdict, is_portugal,
      PortugueseRuleG2p::Options{.with_stress = options.portuguese_with_stress,
                                 .vocoder_stress = options.portuguese_vocoder_stress,
                                 .keep_syllable_dots = options.portuguese_keep_syllable_dots,
                                 .apply_pt_pt_final_esh = options.portuguese_apply_pt_pt_final_esh,
                                 .expand_cardinal_digits = options.portuguese_expand_cardinal_digits});
  RuleBasedG2pInstance out;
  out.canonical_dialect_id = is_portugal ? "pt-PT" : "pt-BR";
  out.kind = RuleBasedG2pKind::Portuguese;
  out.engine = std::make_unique<PortugueseRuleG2p>(std::move(pt));
  return out;
}

using TryFn = std::optional<RuleBasedG2pInstance> (*)(std::string_view, const MoonshineG2POptions&);

const TryFn kTryChain[] = {
    try_english,
    try_spanish,
    try_german,
    try_french,
    try_dutch,
    try_italian,
    try_russian,
    try_chinese,
    try_korean,
    try_vietnamese,
    try_japanese,
    try_arabic,
    try_portuguese,
    try_turkish,
    try_ukrainian,
    try_hindi,
};

}  // namespace

std::optional<RuleBasedG2pInstance> create_rule_based_g2p(std::string_view dialect_id,
                                                        const MoonshineG2POptions& options) {
  const std::string norm = normalize_rule_based_dialect_cli_key(dialect_id);
  if (norm.empty()) {
    throw std::invalid_argument("empty dialect id");
  }
  for (TryFn fn : kTryChain) {
    if (auto o = fn(norm, options)) {
      return o;
    }
  }
  return std::nullopt;
}

std::vector<std::pair<RuleBasedG2pKind, std::vector<std::string>>> rule_based_g2p_dialect_catalog() {
  std::vector<std::pair<RuleBasedG2pKind, std::vector<std::string>>> out;
  out.emplace_back(RuleBasedG2pKind::English, EnglishRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Spanish, SpanishRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::German, GermanRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::French, FrenchRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Dutch, DutchRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Italian, ItalianRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Russian, RussianRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Chinese, ChineseRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Korean, KoreanRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Vietnamese, VietnameseRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Japanese, JapaneseRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Arabic, ArabicRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Portuguese, PortugueseRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Turkish, TurkishRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Ukrainian, UkrainianRuleG2p::dialect_ids());
  out.emplace_back(RuleBasedG2pKind::Hindi, HindiRuleG2p::dialect_ids());
  return out;
}

}  // namespace moonshine_tts
