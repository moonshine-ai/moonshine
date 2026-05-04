// MSA Arabic ONNX + rule G2P (mirrors ``arabic_rule_g2p.py`` CLI subset).
#include "arabic.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " [--onnx-dir DIR] [--dict PATH] [--cuda] [--stdin] [TEXT...]\n"
            << "  Defaults: data/ar_msa/arabertv02_tashkeel_fadel_onnx, data/ar_msa/dict.tsv\n";
}

std::string read_all_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path onnx_dir = std::filesystem::path("data") / "ar_msa" / "arabertv02_tashkeel_fadel_onnx";
  std::filesystem::path dict_path = std::filesystem::path("data") / "ar_msa" / "dict.tsv";
  bool use_cuda = false;
  bool force_stdin = false;
  std::vector<std::string> parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--onnx-dir" && i + 1 < argc) {
      onnx_dir = argv[++i];
    } else if (a == "--dict" && i + 1 < argc) {
      dict_path = argv[++i];
    } else if (a == "--cuda") {
      use_cuda = true;
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
    moonshine_tts::ArabicRuleG2p g2p(onnx_dir, dict_path, use_cuda);
    std::cout << g2p.text_to_ipa(std::move(text)) << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
