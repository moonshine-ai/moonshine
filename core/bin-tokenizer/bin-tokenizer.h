#ifndef BIN_TOKENIZER_H
#define BIN_TOKENIZER_H

#include <array>
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
// call site. A tokenizer.bin with a header (see below) records it itself.
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
  // SentencePiece Unigram: every entry carries a log probability and a word is
  // spelled by the Viterbi path through its pieces that maximizes their sum.
  // Correct for a vocabulary trained as a unigram language model, which is
  // what SentencePiece builds by default and what Phonon's text goes through;
  // over ordinary English, longest match disagrees with it on one word in
  // five ("source" comes out "so" "ur" "ce" where the model learned "source").
  //
  // Needs the scores, so it is only available from a tokenizer.bin with a
  // header. Text is pre-tokenized the SentencePiece way: a space marker is
  // prepended, spaces become the marker, and each marker starts a word that
  // is encoded on its own. A character no piece covers becomes its UTF-8
  // bytes when the vocabulary holds "<0xNN>" entries for them, else the
  // unknown id.
  kUnigram,
};

// Reads and encodes with a tokenizer.bin.
//
// The file is a run of entries, one per token id in order, each a length
// prefix and the token's bytes: one byte for lengths under 128, else two with
// the high bit of the first set (length = second * 128 + first - 128). That is
// the whole of the original format, and files in it have no header.
//
// A file may instead begin with a header, which carries what the flat form
// cannot: which encoding the vocabulary needs, its unknown id, and for a
// Unigram vocabulary a score per entry. It starts with 0xFF 0xFF, which the
// flat form would read as a 32767-byte first token, something no vocabulary
// has, followed by "BTOK", so the two forms cannot be confused:
//
//   0xFF 0xFF 'B' 'T' 'O' 'K'   magic
//   u8  version                 1
//   u8  encoding                BinTokenizerEncoding as an integer
//   u32 count                   little-endian number of entries
//   u32 unk_id                  little-endian, 0xFFFFFFFF when there is none
//   entries                     exactly `count`, as in the flat form
//   f32 scores[count]           little-endian, Unigram only
//
// scripts/convert_tokenizer.py writes both forms.
struct BinTokenizer {
  std::vector<std::vector<uint8_t>> tokens_to_bytes;
  const char *space_string;

  // `encoding` is what a flat (headerless) file is encoded with. A file with a
  // header names its own encoding, and that wins.
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
  // asked for: a header overrides it, and kBpe falls back to kLongestMatch on
  // a vocabulary with no byte fallback block. Exposed for tests.
  BinTokenizerEncoding encoding_for_test() const { return this->encoding; }

 private:
  // Reads either form of the file from memory and builds the indexes.
  void parse(const uint8_t *data, size_t size, const char *what);
  void parse_entries(const uint8_t *data, size_t size, size_t &offset,
                     size_t count_or_zero, const char *what);
  void finish_setup();

  // Token ids grouped by the byte they start with, in ascending id order, so
  // that encoding only compares against the entries that could possibly match
  // instead of walking the whole vocabulary for every subword it emits.
  void build_first_byte_index();
  std::vector<std::vector<int32_t>> tokens_by_first_byte;

  // Finds the byte fallback block and indexes the pieces that merging can
  // produce. Only called for kBpe, since it costs a hash entry per piece.
  void build_merge_index();

  // Indexes every spelling and the "<0xNN>" byte pieces. Only for kUnigram.
  void build_unigram_index();

  template <typename T>
  std::vector<T> text_to_tokens_longest_match(const std::string &text);
  template <typename T>
  std::vector<T> text_to_tokens_bpe(const std::string &text);
  template <typename T>
  std::vector<T> text_to_tokens_unigram(const std::string &text);
  template <typename T>
  void encode_unigram_word(const std::string &word, std::vector<T> &out);

  BinTokenizerEncoding encoding = BinTokenizerEncoding::kLongestMatch;
  // Id of the entry holding byte 0x00; the other 255 follow in order.
  int32_t byte_fallback_base = -1;
  // Spelling to lowest id, over the entries after the byte fallback block.
  // Everything at or before that block is either a control token or a raw byte,
  // and no merge produces either.
  std::unordered_map<std::string, int32_t> merge_ids;

  // Unigram state: one log probability per entry, the unknown id from the
  // header, the spelling index, and the id of each "<0xNN>" piece or -1.
  std::vector<float> scores;
  int32_t unk_id = -1;
  std::unordered_map<std::string, int32_t> piece_ids;
  size_t max_piece_bytes = 1;
  float min_score = 0.0f;
  std::array<int32_t, 256> byte_piece_ids{};
};

#endif
