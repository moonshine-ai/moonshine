#include "cmudict-tsv.h"

#include "text-normalize.h"

#include <fstream>
#include <istream>
#include <set>
#include <sstream>

namespace moonshine_tts {

namespace {

void parse_cmudict_tsv_lines(std::istream& in, std::unordered_map<std::string, std::vector<std::string>>& ipa_by_word) {
  std::unordered_map<std::string, std::set<std::string>> raw;
  std::string line;
  while (std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const auto tab = line.find('\t');
    if (tab == std::string::npos) {
      continue;
    }
    std::string word_token = line.substr(0, tab);
    std::string ipa = line.substr(tab + 1);
    while (!word_token.empty() && (word_token.back() == ' ' || word_token.back() == '\t')) {
      word_token.pop_back();
    }
    while (!ipa.empty() && (ipa.back() == ' ' || ipa.back() == '\t')) {
      ipa.pop_back();
    }
    size_t ws = 0;
    while (ws < word_token.size() && word_token[ws] == ' ') {
      ++ws;
    }
    word_token = word_token.substr(ws);
    if (word_token.empty() || ipa.empty()) {
      continue;
    }
    const std::string key = normalize_grapheme_key(word_token);
    raw[key].insert(std::move(ipa));
  }
  ipa_by_word.clear();
  for (auto& [k, v] : raw) {
    ipa_by_word[k] = {v.begin(), v.end()};
  }
}

}  // namespace

CmudictTsv::CmudictTsv(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open dictionary: " + path.string());
  }
  parse_cmudict_tsv_lines(in, ipa_by_word_);
}

CmudictTsv::CmudictTsv(std::string_view utf8_contents) {
  const std::string buf(utf8_contents);
  std::istringstream in(buf);
  parse_cmudict_tsv_lines(in, ipa_by_word_);
}

const std::vector<std::string>* CmudictTsv::lookup(std::string_view key) const {
  const std::string k(key.begin(), key.end());
  const auto it = ipa_by_word_.find(k);
  if (it == ipa_by_word_.end()) {
    return nullptr;
  }
  return &it->second;
}

}  // namespace moonshine_tts
