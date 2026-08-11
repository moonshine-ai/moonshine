#ifndef BIN_TOKENIZER_H
#define BIN_TOKENIZER_H

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(ANDROID)
#include <android/asset_manager.h>
#endif

// How a vocabulary spells text as token ids. Which one is correct is a property
// of the vocabulary rather than a preference, so it is chosen where the
// tokenizer is built, where the model it came from is known, instead of at each
// call site.
enum class BinTokenizerEncoding {
  // Take the longest entry matching at each position, ties to the lowest id.
  kLongestMatch,
  // Byte-pair encoding, taking a token's id as its merge rank: start from one
  // piece per character and repeatedly join the adjacent pair whose combined
  // spelling has the lowest id. Correct for a vocabulary learned by merges,
  // where the subwords a word is spelled with follow the order the merges were
  // learned rather than which entry happens to be longest.
  //
  // The two rules agree on short words and part ways on longer ones. Over six
  // thousand rare English words against the streaming vocabulary, three in four
  // needed more than one subword and half of those came out differently under
  // the two rules: "luminous" is "lum" "ino" "us" by longest match where the
  // decoder emits "l" "umin" "ous".
  //
  // Needs the 256 single-byte fallback entries present as one contiguous block,
  // which is what a character no merge can spell falls back to. Vocabularies
  // without that block are encoded as kLongestMatch instead, with a warning.
  kBpe,
};

struct BinTokenizer {
  std::vector<std::vector<uint8_t>> tokens_to_bytes;
  const char *space_string;

  BinTokenizer(
      const char *tokenizer_path, const char *space_string = "▁",
      BinTokenizerEncoding encoding = BinTokenizerEncoding::kLongestMatch);
  BinTokenizer(
      const uint8_t *tokenizer_data, size_t tokenizer_data_size,
      const char *space_string = "▁",
      BinTokenizerEncoding encoding = BinTokenizerEncoding::kLongestMatch);
#if defined(ANDROID)
  BinTokenizer(
      const char *tokenizer_path, AAssetManager *assetManager,
      const char *space_string = "▁",
      BinTokenizerEncoding encoding = BinTokenizerEncoding::kLongestMatch);
#endif
  template <typename T>
  std::vector<T> text_to_tokens(const std::string &text);
  template <typename T>
  std::string tokens_to_text(const std::vector<T> &tokens,
                             bool skipSpecials = true);

  template <typename T>
  T text_to_special_token(const std::string &text);

  // Which encoding this tokenizer ended up using. Not always what the caller
  // asked for: kBpe falls back to kLongestMatch on a vocabulary with no byte
  // fallback block. Exposed for tests.
  BinTokenizerEncoding encoding_for_test() const { return this->encoding; }

 private:
  // Token ids grouped by the byte they start with, in ascending id order, so
  // that encoding only compares against the entries that could possibly match
  // instead of walking the whole vocabulary for every subword it emits.
  void build_first_byte_index();
  std::vector<std::vector<int32_t>> tokens_by_first_byte;

  // Finds the byte fallback block and indexes the pieces that merging can
  // produce. Only called for kBpe, since it costs a hash entry per piece.
  void build_merge_index();

  template <typename T>
  std::vector<T> text_to_tokens_longest_match(const std::string &text);
  template <typename T>
  std::vector<T> text_to_tokens_bpe(const std::string &text);

  BinTokenizerEncoding encoding = BinTokenizerEncoding::kLongestMatch;
  // Id of the entry holding byte 0x00; the other 255 follow in order.
  int32_t byte_fallback_base = -1;
  // Spelling to lowest id, over the entries after the byte fallback block.
  // Everything at or before that block is either a control token or a raw byte,
  // and no merge produces either.
  std::unordered_map<std::string, int32_t> merge_ids;
};

#endif