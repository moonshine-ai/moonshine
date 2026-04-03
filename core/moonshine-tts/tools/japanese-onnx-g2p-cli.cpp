#include "japanese-onnx-g2p.h"

#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0 << " [--model-dir PATH] [--dict PATH] [--stdin] [TEXT...]\n";
}

std::string read_all_stdin() {
  std::ostringstream oss;
  oss << std::cin.rdbuf();
  return oss.str();
}

}  // namespace

int main(int argc, char** argv) {
  std::filesystem::path model_dir = std::filesystem::path("data") / "ja" / "roberta_japanese_char_luw_upos_onnx";
  std::filesystem::path dict_path = std::filesystem::path("data") / "ja" / "dict.tsv";
  bool force_stdin = false;
  std::vector<std::string> parts;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--model-dir" && i + 1 < argc) {
      model_dir = argv[++i];
    } else if (a == "--dict" && i + 1 < argc) {
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
    moonshine_tts::JapaneseOnnxG2p g2p(std::move(model_dir), std::move(dict_path), false);
    std::cout << g2p.text_to_ipa(text) << '\n';
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
