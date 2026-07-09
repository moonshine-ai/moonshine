// Baked common-word pronunciation dictionary + runtime override table.
//
// The baked dictionary is a read-only, flash-resident blob (see the generated
// g2p_dict_data.h, produced by tools/build_g2p_dict.py). It stores only the
// frequent English words whose CMUdict pronunciation disagrees with the
// rule-based letter-to-sound (g2p_rules.h), so the rules cover everything else.
// Lookups binary-search the blob directly in flash: no SRAM copy, only a little
// stack scratch.
//
// The override table (Lexicon) is a small, user-supplied word->IPA map loaded
// at runtime (e.g. proper nouns). It is the only G2P structure that lives in
// SRAM and is consulted before the baked dictionary.

#ifndef TTS_G2P_DICT_H_
#define TTS_G2P_DICT_H_

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace tts {

// Look up `word` (case-insensitive; only a-z letters are significant) in the
// baked flash dictionary. On a hit, writes the IPA to *ipa and returns true.
bool DictLookup(std::string_view word, std::string* ipa);

// Runtime, user-supplied pronunciation overrides. Sorted vector + binary
// search; checked before the baked dictionary.
class Lexicon {
 public:
  // Parse a "word<TAB>IPA" TSV: one entry per line, '#'-comments and blank
  // lines ignored, later duplicates win. Returns false if the file can't be
  // opened; malformed lines are skipped.
  bool LoadFromFile(const std::string& path);

  // Add/replace a single entry (word is lowercased, a-z only).
  void Add(std::string_view word, std::string_view ipa);

  // On a hit, writes IPA to *ipa and returns true.
  bool Lookup(std::string_view word, std::string* ipa) const;

  size_t size() const { return entries_.size(); }
  bool empty() const { return entries_.empty(); }

 private:
  void EnsureSorted() const;
  // Mutable so const Lookup can lazily sort after Add().
  mutable std::vector<std::pair<std::string, std::string>> entries_;
  mutable bool sorted_ = true;
};

// Normalize a raw word to a dictionary key: lowercase, keep only a-z letters.
std::string NormalizeWordKey(std::string_view word);

}  // namespace tts

#endif  // TTS_G2P_DICT_H_
