// French rule + lexicon G2P (no ONNX). Batch mode: one input line -> one IPA line on stdout.
// For fast parity checks vs Python without spawning moonshine-tts-g2p per phrase.
#include "french.h"

#include <cctype>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--dict PATH] [--csv-dir DIR]\n"
      << "       [--no-stress] [--no-liaison] [--no-optional-liaison]\n"
      << "       [--no-oov] [--no-expand-digits] [--whole-stdin]\n"
      << "       [TEXT...]\n"
      << "  With no TEXT: read stdin line-by-line; each non-empty line is phonemized; "
         "empty lines print an empty line.\n"
      << "  With TEXT...: join arguments with spaces, phonemize once, print one line.\n"
      << "  --whole-stdin: read entire stdin as one string (newlines kept) and phonemize once.\n"
      << "  Default dict: data/fr/dict.tsv ; default CSV dir: data/fr (relative to cwd).\n";
}

std::string trim_sv(std::string s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())) != 0) {
    s.erase(s.begin());
  }
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())) != 0) {
    s.pop_back();
  }
  return s;
}

std::string read_all_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path dict_path = std::filesystem::path("data") / "fr" / "dict.tsv";
  std::filesystem::path csv_dir = std::filesystem::path("data") / "fr";
  moonshine_tts::FrenchRuleG2p::Options opt;
  bool whole_stdin = false;
  std::vector<std::string> parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--dict" && i + 1 < argc) {
      dict_path = argv[++i];
    } else if (a == "--csv-dir" && i + 1 < argc) {
      csv_dir = argv[++i];
    } else if (a == "--no-stress") {
      opt.with_stress = false;
    } else if (a == "--no-liaison") {
      opt.liaison = false;
    } else if (a == "--no-optional-liaison") {
      opt.liaison_optional = false;
    } else if (a == "--no-oov") {
      opt.oov_rules = false;
    } else if (a == "--no-expand-digits") {
      opt.expand_cardinal_digits = false;
    } else if (a == "--whole-stdin") {
      whole_stdin = true;
    } else {
      parts.push_back(a);
    }
  }

  try {
    moonshine_tts::FrenchRuleG2p g2p(dict_path, csv_dir, opt);

    if (whole_stdin) {
      const std::string text = read_all_stdin();
      std::cout << g2p.text_to_ipa(text) << '\n';
      return 0;
    }

    if (!parts.empty()) {
      std::string text;
      for (size_t t = 0; t < parts.size(); ++t) {
        if (t > 0) {
          text.push_back(' ');
        }
        text += parts[t];
      }
      std::cout << g2p.text_to_ipa(text) << '\n';
      return 0;
    }

    std::string line;
    while (std::getline(std::cin, line)) {
      if (!line.empty() && line.back() == '\r') {
        line.pop_back();
      }
      line = trim_sv(std::move(line));
      if (line.empty()) {
        std::cout << '\n';
        continue;
      }
      std::cout << g2p.text_to_ipa(line) << '\n';
    }
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
