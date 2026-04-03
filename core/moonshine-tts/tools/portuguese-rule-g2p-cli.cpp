// Stand-alone Portuguese rule + lexicon G2P (no ONNX). Mirrors ``portuguese_rule_g2p.py`` CLI subset.
#include "portuguese.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--dict PATH] [--portugal] [--no-stress] [--syllable-initial-stress] [--keep-syllable-dots] "
               "[--no-expand-digits] [--stdin] [TEXT...]\n"
            << "  Default dict: data/pt_br/dict.tsv (use --portugal for data/pt_pt/dict.tsv).\n";
}

std::string read_all_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path dict_path = std::filesystem::path("data") / "pt_br" / "dict.tsv";
  bool is_portugal = false;
  moonshine_tts::PortugueseRuleG2p::Options opt;
  bool force_stdin = false;
  std::vector<std::string> parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--dict" && i + 1 < argc) {
      dict_path = argv[++i];
    } else if (a == "--portugal") {
      is_portugal = true;
      dict_path = std::filesystem::path("data") / "pt_pt" / "dict.tsv";
    } else if (a == "--no-stress") {
      opt.with_stress = false;
    } else if (a == "--syllable-initial-stress") {
      opt.vocoder_stress = false;
    } else if (a == "--keep-syllable-dots") {
      opt.keep_syllable_dots = true;
    } else if (a == "--no-expand-digits") {
      opt.expand_cardinal_digits = false;
    } else if (a == "--stdin") {
      force_stdin = true;
    } else {
      parts.push_back(a);
    }
  }

  std::string text;
  if (!parts.empty()) {
    for (size_t t = 0; t < parts.size(); ++t) {
      if (t > 0) {
        text += ' ';
      }
      text += parts[t];
    }
  } else if (force_stdin || parts.empty()) {
    text = read_all_stdin();
  }

  try {
    moonshine_tts::PortugueseRuleG2p g2p(dict_path, is_portugal, opt);
    std::cout << g2p.text_to_ipa(text) << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
