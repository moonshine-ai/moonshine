// Unified G2P CLI: rule-based dialects (English, Spanish, German, …).
#include "builtin-cpp-data-root.h"
#include "g2p-word-log.h"
#include "spanish.h"
#include "moonshine-g2p.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using moonshine_tts::MoonshineG2P;
using moonshine_tts::MoonshineG2POptions;
using moonshine_tts::RuleBasedG2pKind;
using moonshine_tts::dialect_uses_rule_based_g2p;
using moonshine_tts::format_g2p_word_log_line;
using moonshine_tts::rule_based_g2p_dialect_catalog;
using moonshine_tts::spanish_dialect_cli_ids;

namespace {

void usage(const char *argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--language|--lang ID] [--model-root DIR]\n"
      << "       (default model root: builtin_cpp_data_root() / bundled data/; same as MoonshineTTS/PiperTTS)\n"
      << "       [--dict PATH] [--heteronym-onnx PATH] [--oov-onnx PATH]\n"
      << "       [--cuda] [--log-words|-v] [--debug-heteronym]\n"
      << "       [--no-stress] [--broad-phonemes] [--stdin]\n"
      << "       [--german-dict PATH] [--german-syllable-initial-stress]\n"
      << "       [--russian-dict PATH] [--russian-syllable-initial-stress]\n"
      << "       [--chinese-dict PATH] [--chinese-onnx-dir PATH] [--korean-dict PATH] "
         "[--no-korean-expand-digits]\n"
      << "       [--japanese-dict PATH] [--japanese-onnx-dir DIR]\n"
      << "       [--arabic-dict PATH] [--arabic-onnx-dir DIR]\n"
      << "       [--dutch-dict PATH] [--dutch-syllable-initial-stress] [--no-dutch-expand-digits]\n"
      << "       [--portuguese-dict PATH] [--portuguese-syllable-initial-stress] [--no-portuguese-expand-digits]\n"
      << "       [--no-turkish-expand-digits] [--no-ukrainian-expand-digits]\n"
      << "       [--hindi-dict PATH] [--no-hindi-expand-digits]\n"
      << "       [--french-dict PATH] [--french-csv-dir DIR]\n"
      << "       [--no-french-liaison] [--no-french-oov] [--no-french-expand-digits]\n"
      << "       [--no-french-optional-liaison]\n"
      << "       [--print-spanish-dialects] [TEXT...]\n"
      << "  Default dialect: en_us (rule-based CMUdict + optional heteronym/OOV ONNX under "
         "<model-root>/en_us/). Default phrase when no TEXT: \"Hello world!\".\n"
      << "  Spanish dialects use rule-based G2P (no ONNX). With a Spanish dialect and no TEXT, "
         "stdin is read unless you pass --stdin explicitly for empty input.\n"
      << "  German (de, de-DE, german): rule-based G2P with <model-root>/de/dict.tsv by default; "
         "override with --german-dict.\n"
      << "  French (fr, fr-FR, french): rule-based G2P; default lexicon <model-root>/../data/fr/dict.tsv "
         "or <model-root>/fr/dict.tsv; POS CSVs in the same directory tree.\n"
      << "  Dutch (nl, nl-NL, dutch): rule-based G2P; default lexicon <model-root>/../data/nl/dict.tsv "
         "or <model-root>/nl/dict.tsv; override with --dutch-dict.\n"
      << "  Russian (ru, ru-RU, russian): rule-based G2P; default <model-root>/../data/ru/dict.tsv "
         "or <model-root>/ru/dict.tsv; override with --russian-dict.\n"
      << "  Chinese (zh, zh-Hans, cmn, …): RoBERTa UPOS ONNX + lexicon; default "
         "<model-root>/../data/zh_hans/dict.tsv and "
         "<model-root>/../data/zh_hans/roberta_chinese_base_upos_onnx/; override with "
         "--chinese-dict / --chinese-onnx-dir.\n"
      << "  Korean (ko, ko-KR, korean): rule-based G2P; default <model-root>/../data/ko/dict.tsv "
         "or <model-root>/ko/dict.tsv; override with --korean-dict.\n"
      << "  Japanese (ja, ja-JP, japanese): ONNX LUW + lexicon; default <model-root>/../data/ja/dict.tsv "
         "and <model-root>/../data/ja/roberta_japanese_char_luw_upos_onnx/; override with "
         "--japanese-dict / --japanese-onnx-dir.\n"
      << "  Arabic (ar, ar-MSA, msa, arabic): ONNX tashkīl + rules; default <model-root>/../data/ar_msa/ "
         "arabertv02_tashkeel_fadel_onnx + dict.tsv; override with --arabic-onnx-dir / --arabic-dict.\n"
      << "  Portuguese (pt_br, pt-br, pt_pt, portugal, …): rule-based G2P; default "
         "<model-root>/../data/pt_br/dict.tsv or pt_pt/dict.tsv; override with --portuguese-dict.\n"
      << "  Turkish (tr, tr-TR, turkish): rule-based G2P (no lexicon); optional cardinal digit expansion.\n"
      << "  Ukrainian (uk, uk-UA, ukrainian): rule-based G2P (no lexicon); optional cardinal digit expansion.\n"
      << "  Hindi (hi, hi-IN, hindi): lexicon lookup + Devanagari rules; default <model-root>/../data/hi/dict.tsv "
         "(or grandparent …/data/hi), else <model-root>/hi/dict.tsv; bundled cpp/data/hi/dict.tsv; "
         "override with --hindi-dict.\n"
      << "  -d PATH / --dict PATH: English CMU TSV (en_us only; overrides default under "
         "<model-root>/en_us/).\n";
}

std::string read_all_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

const char *rule_based_kind_label(RuleBasedG2pKind k) {
  switch (k) {
  case RuleBasedG2pKind::English:
    return "English";
  case RuleBasedG2pKind::Spanish:
    return "Spanish";
  case RuleBasedG2pKind::German:
    return "German";
  case RuleBasedG2pKind::French:
    return "French";
  case RuleBasedG2pKind::Dutch:
    return "Dutch";
  case RuleBasedG2pKind::Italian:
    return "Italian";
  case RuleBasedG2pKind::Russian:
    return "Russian";
  case RuleBasedG2pKind::Chinese:
    return "Chinese";
  case RuleBasedG2pKind::Korean:
    return "Korean";
  case RuleBasedG2pKind::Vietnamese:
    return "Vietnamese";
  case RuleBasedG2pKind::Japanese:
    return "Japanese";
  case RuleBasedG2pKind::Arabic:
    return "Arabic";
  case RuleBasedG2pKind::Portuguese:
    return "Portuguese";
  case RuleBasedG2pKind::Turkish:
    return "Turkish";
  case RuleBasedG2pKind::Ukrainian:
    return "Ukrainian";
  case RuleBasedG2pKind::Hindi:
    return "Hindi";
  default:
    return "Unknown";
  }
}

void print_rule_based_dialect_catalog(std::ostream &os) {
  os << "\nSupported languages and dialect ids (pass with --language or --lang):\n\n";
  for (const auto &entry : rule_based_g2p_dialect_catalog()) {
    os << "  " << rule_based_kind_label(entry.first) << '\n';
    for (const std::string &id : entry.second) {
      os << "      " << id << '\n';
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::string dialect_str = "en_us";
  MoonshineG2POptions opt;
  bool model_root_from_cli = false;
  bool log_words = false;
  bool force_stdin = false;
  bool print_spanish_dialects = false;
  std::vector<std::string> text_parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--print-spanish-dialects") {
      print_spanish_dialects = true;
    } else if ((a == "--dict" || a == "-d") && i + 1 < argc) {
      opt.english_dict_path = argv[++i];
    } else if (a == "--heteronym-onnx" && i + 1 < argc) {
      opt.heteronym_onnx_override = argv[++i];
    } else if (a == "--oov-onnx" && i + 1 < argc) {
      opt.oov_onnx_override = argv[++i];
    } else if (a == "--cuda") {
      opt.use_cuda = true;
    } else if (a == "--log-words" || a == "-v") {
      log_words = true;
    } else if (a == "--debug-heteronym") {
      if (setenv("MOONSHINE_TTS_DEBUG_HET", "1", 1) != 0) {
        std::cerr << "error: setenv MOONSHINE_TTS_DEBUG_HET failed\n";
        return 1;
      }
    } else if ((a == "--language" || a == "--lang") && i + 1 < argc) {
      dialect_str = argv[++i];
    } else if (a == "--model-root" && i + 1 < argc) {
      opt.model_root = argv[++i];
      model_root_from_cli = true;
    } else if (a == "--german-dict" && i + 1 < argc) {
      opt.german_dict_path = argv[++i];
    } else if (a == "--chinese-dict" && i + 1 < argc) {
      opt.chinese_dict_path = argv[++i];
    } else if (a == "--chinese-onnx-dir" && i + 1 < argc) {
      opt.chinese_onnx_model_dir = argv[++i];
    } else if (a == "--korean-dict" && i + 1 < argc) {
      opt.korean_dict_path = argv[++i];
    } else if (a == "--japanese-dict" && i + 1 < argc) {
      opt.japanese_dict_path = argv[++i];
    } else if (a == "--japanese-onnx-dir" && i + 1 < argc) {
      opt.japanese_onnx_model_dir = argv[++i];
    } else if (a == "--arabic-dict" && i + 1 < argc) {
      opt.arabic_dict_path = argv[++i];
    } else if (a == "--arabic-onnx-dir" && i + 1 < argc) {
      opt.arabic_onnx_model_dir = argv[++i];
    } else if (a == "--no-korean-expand-digits") {
      opt.korean_expand_cardinal_digits = false;
    } else if (a == "--russian-dict" && i + 1 < argc) {
      opt.russian_dict_path = argv[++i];
    } else if (a == "--russian-syllable-initial-stress") {
      opt.russian_vocoder_stress = false;
    } else if (a == "--dutch-dict" && i + 1 < argc) {
      opt.dutch_dict_path = argv[++i];
    } else if (a == "--portuguese-dict" && i + 1 < argc) {
      opt.portuguese_dict_path = argv[++i];
    } else if (a == "--portuguese-syllable-initial-stress") {
      opt.portuguese_vocoder_stress = false;
    } else if (a == "--no-portuguese-expand-digits") {
      opt.portuguese_expand_cardinal_digits = false;
    } else if (a == "--no-turkish-expand-digits") {
      opt.turkish_expand_cardinal_digits = false;
    } else if (a == "--no-ukrainian-expand-digits") {
      opt.ukrainian_expand_cardinal_digits = false;
    } else if (a == "--hindi-dict" && i + 1 < argc) {
      opt.hindi_dict_path = argv[++i];
    } else if (a == "--no-hindi-expand-digits") {
      opt.hindi_expand_cardinal_digits = false;
    } else if (a == "--french-dict" && i + 1 < argc) {
      opt.french_dict_path = argv[++i];
    } else if (a == "--french-csv-dir" && i + 1 < argc) {
      opt.french_csv_dir = argv[++i];
    } else if (a == "--no-french-liaison") {
      opt.french_liaison = false;
    } else if (a == "--no-french-oov") {
      opt.french_oov_rules = false;
    } else if (a == "--no-french-expand-digits") {
      opt.french_expand_cardinal_digits = false;
    } else if (a == "--no-french-optional-liaison") {
      opt.french_liaison_optional = false;
    } else if (a == "--german-syllable-initial-stress") {
      opt.german_vocoder_stress = false;
    } else if (a == "--dutch-syllable-initial-stress") {
      opt.dutch_vocoder_stress = false;
    } else if (a == "--no-dutch-expand-digits") {
      opt.dutch_expand_cardinal_digits = false;
    } else if (a == "--no-stress") {
      opt.spanish_with_stress = false;
      opt.german_with_stress = false;
      opt.french_with_stress = false;
      opt.dutch_with_stress = false;
      opt.italian_with_stress = false;
      opt.russian_with_stress = false;
      opt.portuguese_with_stress = false;
      opt.turkish_with_stress = false;
      opt.ukrainian_with_stress = false;
      opt.hindi_with_stress = false;
    } else if (a == "--broad-phonemes") {
      opt.spanish_narrow_obstruents = false;
    } else if (a == "--stdin") {
      force_stdin = true;
    } else {
      text_parts.push_back(a);
    }
  }

  if (!model_root_from_cli) {
    opt.model_root = moonshine_tts::builtin_cpp_data_root();
  }

  if (print_spanish_dialects) {
    for (const std::string &id : spanish_dialect_cli_ids()) {
      std::cout << id << '\n';
    }
    if (text_parts.empty() && !force_stdin) {
      return 0;
    }
  }

  std::string phrase;
  if (!text_parts.empty()) {
    for (size_t t = 0; t < text_parts.size(); ++t) {
      if (t > 0) {
        phrase += ' ';
      }
      phrase += text_parts[t];
    }
  } else if (force_stdin || dialect_uses_rule_based_g2p(dialect_str, opt)) {
    phrase = read_all_stdin();
  } else {
    phrase = "Hello world!";
  }

  try {
    MoonshineG2P g2p(dialect_str, opt);
    std::vector<moonshine_tts::G2pWordLog> word_log;
    std::cout << g2p.text_to_ipa(phrase, log_words ? &word_log : nullptr) << '\n';
    if (log_words) {
      for (const auto &e : word_log) {
        std::cerr << format_g2p_word_log_line(e) << '\n';
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "error: " << e.what() << '\n';
    if (std::strstr(e.what(), "unsupported dialect") != nullptr) {
      print_rule_based_dialect_catalog(std::cerr);
    }
    return 1;
  }
  return 0;
}
