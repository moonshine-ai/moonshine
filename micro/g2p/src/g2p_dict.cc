#include "g2p_dict.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>

#include "g2p_dict_data.h"

namespace g2p {

namespace {

// Decode `count` packed phone ids starting at body offset `start` into IPA.
std::string DecodeIpa(uint32_t start, unsigned count) {
  std::string ipa;
  for (unsigned k = 0; k < count; ++k) {
    const unsigned id = kG2pBody[start + k];
    if (static_cast<int>(id) < kG2pNumPhones) ipa += kG2pPhones[id];
  }
  return ipa;
}

// The restart (first) key of a block: its entry always has sharedPrefixLen ==
// 0.
std::string RestartKey(int block) {
  const uint32_t off = kG2pBlockOffsets[block];
  const unsigned slen = kG2pBody[off + 1];
  return std::string(reinterpret_cast<const char*>(kG2pBody + off + 2), slen);
}

}  // namespace

std::string NormalizeWordKey(std::string_view word) {
  std::string key;
  key.reserve(word.size());
  for (unsigned char uc : word) {
    const char c = static_cast<char>(std::tolower(uc));
    // Keep letters and internal apostrophes so contractions ("what's", "don't")
    // match their dictionary entries instead of a reduced apostrophe-free form.
    if ((c >= 'a' && c <= 'z') || c == '\'') key.push_back(c);
  }
  // Drop apostrophes used as surrounding quotes ('word').
  size_t b = 0, e = key.size();
  while (b < e && key[b] == '\'') ++b;
  while (e > b && key[e - 1] == '\'') --e;
  return key.substr(b, e - b);
}

bool DictLookup(std::string_view word, std::string* ipa) {
  if (kG2pNumEntries == 0) return false;
  const std::string key = NormalizeWordKey(word);
  if (key.empty()) return false;

  // Find the block that could contain `key`: the largest block whose restart
  // key is <= key.
  int lo = 0, hi = kG2pNumBlocks - 1, cand = -1;
  while (lo <= hi) {
    const int mid = (lo + hi) / 2;
    const int cmp = key.compare(RestartKey(mid));
    if (cmp == 0) {
      cand = mid;
      break;
    }
    if (cmp > 0) {
      cand = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  if (cand < 0) return false;

  // Linear scan within the candidate block, reconstructing front-coded keys.
  uint32_t off = kG2pBlockOffsets[cand];
  const int count =
      (cand < kG2pNumBlocks - 1)
          ? kG2pBlockSize
          : (kG2pNumEntries - (kG2pNumBlocks - 1) * kG2pBlockSize);
  std::string cur;
  for (int e = 0; e < count; ++e) {
    const unsigned shared = kG2pBody[off++];
    const unsigned slen = kG2pBody[off++];
    cur.resize(shared);
    cur.append(reinterpret_cast<const char*>(kG2pBody + off), slen);
    off += slen;
    const unsigned ilen = kG2pBody[off++];
    const uint32_t istart = off;
    off += ilen;
    const int cmp = key.compare(cur);
    if (cmp == 0) {
      *ipa = DecodeIpa(istart, ilen);
      return true;
    }
    if (cmp < 0) return false;  // keys are sorted; we've passed it
  }
  return false;
}

// ---------------------------------------------------------------------------
// // Lexicon (runtime overrides)
// ---------------------------------------------------------------------------
// //

void Lexicon::Add(std::string_view word, std::string_view ipa) {
  const std::string key = NormalizeWordKey(word);
  if (key.empty() || ipa.empty()) return;
  entries_.emplace_back(key, std::string(ipa));
  sorted_ = false;
}

void Lexicon::EnsureSorted() const {
  if (sorted_) return;
  std::stable_sort(
      entries_.begin(), entries_.end(),
      [](const auto& a, const auto& b) { return a.first < b.first; });
  // De-duplicate, keeping the last occurrence of each key (later wins).
  std::vector<std::pair<std::string, std::string>> out;
  out.reserve(entries_.size());
  for (auto& kv : entries_) {
    if (!out.empty() && out.back().first == kv.first) {
      out.back().second = kv.second;
    } else {
      out.push_back(kv);
    }
  }
  entries_ = std::move(out);
  sorted_ = true;
}

bool Lexicon::Lookup(std::string_view word, std::string* ipa) const {
  if (entries_.empty()) return false;
  EnsureSorted();
  const std::string key = NormalizeWordKey(word);
  if (key.empty()) return false;
  const auto it = std::lower_bound(
      entries_.begin(), entries_.end(), key,
      [](const auto& a, const std::string& k) { return a.first < k; });
  if (it == entries_.end() || it->first != key) return false;
  *ipa = it->second;
  return true;
}

bool Lexicon::LoadFromFile(const std::string& path) {
  std::ifstream in(path);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) {
      line.pop_back();
    }
    if (line.empty() || line[0] == '#') continue;
    const auto tab = line.find('\t');
    if (tab == std::string::npos) continue;
    std::string word = line.substr(0, tab);
    std::string ipa = line.substr(tab + 1);
    // Trim surrounding spaces from the IPA field.
    while (!ipa.empty() && (ipa.front() == ' ' || ipa.front() == '\t'))
      ipa.erase(0, 1);
    while (!ipa.empty() && (ipa.back() == ' ' || ipa.back() == '\t'))
      ipa.pop_back();
    Add(word, ipa);
  }
  return true;
}

}  // namespace g2p
