// Dutch rule + lexicon G2P (no ONNX). Batch mode: one input line -> one IPA line on stdout.
// For parity checks vs Python without spawning one process per phrase.
#include "dutch.h"

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
      << " [--dict PATH] [--no-stress] [--syllable-initial-stress] [--no-expand-digits]\n"
      << "       [--whole-stdin] [TEXT...]\n"
      << "  With no TEXT: read stdin line-by-line; each non-empty line is phonemized; "
         "empty lines print an empty line.\n"
      << "  With TEXT...: join arguments with spaces, phonemize once, print one line.\n"
      << "  --whole-stdin: read entire stdin as one string (newlines kept) and phonemize once.\n"
      << "  Default dict: data/nl/dict.tsv (relative to cwd).\n";
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
  std::filesystem::path dict_path = std::filesystem::path("data") / "nl" / "dict.tsv";
  moonshine_tts::DutchRuleG2p::Options opt;
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
    } else if (a == "--no-stress") {
      opt.with_stress = false;
    } else if (a == "--syllable-initial-stress") {
      opt.vocoder_stress = false;
    } else if (a == "--no-expand-digits") {
      opt.expand_cardinal_digits = false;
    } else if (a == "--whole-stdin") {
      whole_stdin = true;
    } else {
      parts.push_back(a);
    }
  }

  try {
    moonshine_tts::DutchRuleG2p g2p(dict_path, opt);

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
