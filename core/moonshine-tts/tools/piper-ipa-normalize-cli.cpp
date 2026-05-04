// Prints Piper-ready IPA (NFC + replacements + optional inventory coercion) for parity tests.
// Usage: piper-ipa-normalize-cli --lang en_us --phoneme-json PATH [--no-coerce] -- IPA_UTF8

#include "ipa-postprocess.h"

#include <fstream>
#include <iostream>
#include <nlohmann/json.h>
#include <string>
#include <string_view>
#include <unordered_set>

namespace {

void print_usage() {
  std::cerr << "Usage: piper-ipa-normalize-cli --lang KEY --phoneme-json PATH [--no-coerce] -- IPA\n";
}

bool load_keys(const std::string& path, std::unordered_set<std::string>& keys) {
  std::ifstream f(path);
  if (!f) {
    return false;
  }
  nlohmann::json j;
  f >> j;
  if (!j.contains("phoneme_id_map") || !j["phoneme_id_map"].is_object()) {
    return false;
  }
  keys.clear();
  for (auto it = j["phoneme_id_map"].begin(); it != j["phoneme_id_map"].end(); ++it) {
    keys.insert(it.key());
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string lang = "en_us";
  std::string json_path;
  bool coerce = true;
  int pos = 1;
  while (pos < argc) {
    std::string_view a = argv[pos];
    if (a == "--") {
      ++pos;
      break;
    }
    if (a == "--lang" && pos + 1 < argc) {
      lang = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (a == "--phoneme-json" && pos + 1 < argc) {
      json_path = argv[pos + 1];
      pos += 2;
      continue;
    }
    if (a == "--no-coerce") {
      coerce = false;
      ++pos;
      continue;
    }
    if (a == "-h" || a == "--help") {
      print_usage();
      return 0;
    }
    std::cerr << "Unknown argument: " << a << '\n';
    print_usage();
    return 2;
  }
  if (json_path.empty()) {
    print_usage();
    return 2;
  }
  std::unordered_set<std::string> keys;
  if (!load_keys(json_path, keys)) {
    std::cerr << "Failed to load phoneme_id_map from " << json_path << '\n';
    return 1;
  }
  std::string ipa;
  if (pos < argc) {
    ipa = argv[pos];
  }
  for (int i = pos + 1; i < argc; ++i) {
    ipa.push_back(' ');
    ipa += argv[i];
  }
  const std::string out = moonshine_tts::ipa_to_piper_ready(ipa, lang, keys, coerce);
  std::cout << out;
  return 0;
}
