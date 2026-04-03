// Vietnamese rule + lexicon G2P. Mirrors ``vietnamese_rule_g2p.py`` CLI subset.
#include "vietnamese.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " [--dict PATH] [--stdin] [TEXT...]\n"
            << "  Default dict: data/vi/dict.tsv (relative to current directory).\n";
}

std::string read_all_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path dict_path = std::filesystem::path("data") / "vi" / "dict.tsv";
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
    moonshine_tts::VietnameseRuleG2p g2p(dict_path);
    std::cout << g2p.text_to_ipa(std::move(text)) << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
