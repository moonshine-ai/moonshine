#include "bin-tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "debug-utils.h"
#include "string-utils.h"

namespace {

// Bytes in the UTF-8 sequence a lead byte introduces. A byte that cannot start
// one counts as a sequence of its own, so malformed input is split into single
// bytes and handled by the byte fallback rather than rejected.
size_t utf8_sequence_length(uint8_t lead) {
  if ((lead & 0x80) == 0x00) {
    return 1;
  }
  if ((lead & 0xE0) == 0xC0) {
    return 2;
  }
  if ((lead & 0xF0) == 0xE0) {
    return 3;
  }
  if ((lead & 0xF8) == 0xF0) {
    return 4;
  }
  return 1;
}

// The header a tokenizer.bin may start with; see bin-tokenizer.h.
constexpr uint8_t kHeaderMagic[] = {0xFF, 0xFF, 'B', 'T', 'O', 'K'};
constexpr size_t kHeaderSize = sizeof(kHeaderMagic) + 1 + 1 + 4 + 4;
constexpr uint8_t kHeaderVersion = 1;
constexpr uint32_t kNoUnkId = 0xFFFFFFFFu;

// The penalty below the lowest piece score that the ``tokenizers`` crate
// gives an unknown character, so the lattice only uses one where no piece
// fits.
constexpr float kUnigramUnkPenalty = 10.0f;

uint32_t read_u32_le(const uint8_t *p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}

float read_f32_le(const uint8_t *p) {
  const uint32_t bits = read_u32_le(p);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

std::vector<uint8_t> read_whole_file(const char *path) {
  FILE *file = std::fopen(path, "rb");
  if (!file) {
    std::string message =
        "Failed to open tokenizer file at " + std::string(path);
    std::perror(message.c_str());
    throw std::runtime_error(message);
  }
  std::vector<uint8_t> data;
  uint8_t buffer[65536];
  while (true) {
    const size_t got = std::fread(buffer, 1, sizeof(buffer), file);
    data.insert(data.end(), buffer, buffer + got);
    if (got < sizeof(buffer)) {
      break;
    }
  }
  std::fclose(file);
  return data;
}

}  // namespace

BinTokenizer::BinTokenizer(const char *tokenizer_path, const char *space_string,
                           BinTokenizerEncoding encoding) {
  this->space_string = space_string;
  this->encoding = encoding;
  const std::vector<uint8_t> data = read_whole_file(tokenizer_path);
  this->parse(data.data(), data.size(), tokenizer_path);
}

BinTokenizer::BinTokenizer(const uint8_t *tokenizer_data,
                           size_t tokenizer_data_size, const char *space_string,
                           BinTokenizerEncoding encoding) {
  this->space_string = space_string;
  this->encoding = encoding;
  if (!tokenizer_data || tokenizer_data_size == 0) {
    std::string message = "Tokenizer data is nullptr or empty";
    throw std::runtime_error(message);
  }
  this->parse(tokenizer_data, tokenizer_data_size, "tokenizer data");
}

#if defined(ANDROID)
BinTokenizer::BinTokenizer(const char *tokenizer_path,
                           AAssetManager *assetManager,
                           const char *space_string,
                           BinTokenizerEncoding encoding) {
  this->space_string = space_string;
  this->encoding = encoding;
  AAsset *asset =
      AAssetManager_open(assetManager, tokenizer_path, AASSET_MODE_STREAMING);
  if (asset == nullptr) {
    fprintf(stderr, "Failed to open asset %s at %s:%d\n", tokenizer_path,
            __FILE__, __LINE__);
    throw std::runtime_error("Failed to open tokenizer file at " +
                             std::string(tokenizer_path));
  }
  std::vector<uint8_t> data;
  uint8_t buffer[65536];
  while (true) {
    const int got = AAsset_read(asset, buffer, sizeof(buffer));
    if (got <= 0) {
      break;
    }
    data.insert(data.end(), buffer, buffer + got);
  }
  AAsset_close(asset);
  this->parse(data.data(), data.size(), tokenizer_path);
}
#endif

void BinTokenizer::parse(const uint8_t *data, size_t size, const char *what) {
  const bool has_header =
      size >= kHeaderSize &&
      std::memcmp(data, kHeaderMagic, sizeof(kHeaderMagic)) == 0;
  size_t offset = 0;
  size_t count = 0;
  if (has_header) {
    offset = sizeof(kHeaderMagic);
    const uint8_t version = data[offset++];
    if (version != kHeaderVersion) {
      throw std::runtime_error("Unsupported tokenizer format version " +
                               std::to_string(version) + " in " +
                               std::string(what));
    }
    const uint8_t encoding_byte = data[offset++];
    switch (encoding_byte) {
      case 0:
        this->encoding = BinTokenizerEncoding::kLongestMatch;
        break;
      case 1:
        this->encoding = BinTokenizerEncoding::kBpe;
        break;
      case 2:
        this->encoding = BinTokenizerEncoding::kUnigram;
        break;
      default:
        throw std::runtime_error("Unknown tokenizer encoding " +
                                 std::to_string(encoding_byte) + " in " +
                                 std::string(what));
    }
    count = read_u32_le(data + offset);
    offset += 4;
    const uint32_t unk = read_u32_le(data + offset);
    offset += 4;
    this->unk_id = unk == kNoUnkId ? -1 : static_cast<int32_t>(unk);
    if (count == 0) {
      throw std::runtime_error("No tokens found in " + std::string(what));
    }
  }
  this->parse_entries(data, size, offset, count, what);
  if (has_header && this->encoding == BinTokenizerEncoding::kUnigram) {
    const size_t needed = this->tokens_to_bytes.size() * 4;
    if (size - offset < needed) {
      throw std::runtime_error("Truncated tokenizer data: missing scores in " +
                               std::string(what));
    }
    this->scores.resize(this->tokens_to_bytes.size());
    for (size_t i = 0; i < this->scores.size(); i++) {
      this->scores[i] = read_f32_le(data + offset);
      offset += 4;
    }
  }
  if (this->tokens_to_bytes.empty()) {
    throw std::runtime_error("No tokens found in " + std::string(what));
  }
  this->finish_setup();
}

// Reads entries from `offset`: exactly `count_or_zero` of them when that is
// set, else until the data runs out (the flat form).
void BinTokenizer::parse_entries(const uint8_t *data, size_t size,
                                 size_t &offset, size_t count_or_zero,
                                 const char *what) {
  while (offset < size &&
         (count_or_zero == 0 || this->tokens_to_bytes.size() < count_or_zero)) {
    uint8_t first_byte = data[offset];
    offset++;
    if (first_byte == 0) {
      tokens_to_bytes.push_back(std::vector<uint8_t>());
      continue;
    }
    size_t byte_count;
    if (first_byte < 128) {
      byte_count = first_byte;
    } else {
      if (offset >= size) {
        throw std::runtime_error(
            "Truncated tokenizer data: missing length byte at offset " +
            std::to_string(offset) + " in " + std::string(what));
      }
      uint8_t second_byte = data[offset];
      byte_count = (second_byte * 128) + first_byte - 128;
      offset++;
    }
    // Guard against a truncated blob so the memcpy below cannot read past the
    // end of the input (offset is always <= size here, so the subtraction
    // cannot underflow).
    if (byte_count > size - offset) {
      throw std::runtime_error("Truncated tokenizer data: token of " +
                               std::to_string(byte_count) +
                               " bytes at offset " + std::to_string(offset) +
                               " exceeds input size " + std::to_string(size) +
                               " in " + std::string(what));
    }
    std::vector<uint8_t> bytes(byte_count);
    // A two-byte length can legitimately encode a zero-length token; skip the
    // copy in that case, since bytes.data() is null for an empty vector and
    // memcpy() is declared nonnull (passing null is UB even when count is 0).
    if (byte_count > 0) {
      std::memcpy(bytes.data(), data + offset, byte_count);
    }
    offset += byte_count;
    tokens_to_bytes.push_back(bytes);
  }
  if (count_or_zero != 0 && this->tokens_to_bytes.size() != count_or_zero) {
    throw std::runtime_error("Truncated tokenizer data: header promises " +
                             std::to_string(count_or_zero) + " tokens, found " +
                             std::to_string(this->tokens_to_bytes.size()) +
                             " in " + std::string(what));
  }
}

void BinTokenizer::finish_setup() {
  this->build_first_byte_index();
  if (this->encoding == BinTokenizerEncoding::kBpe) {
    this->build_merge_index();
  } else if (this->encoding == BinTokenizerEncoding::kUnigram) {
    this->build_unigram_index();
  }
}

template <typename T>
T BinTokenizer::text_to_special_token(const std::string &text) {
  // A special token is one exact vocabulary entry rather than something to be
  // spelled out of pieces, so look it up directly. Encoding it would be wrong
  // under kBpe, where control tokens sit outside the merge index on purpose and
  // would come back as a string of raw bytes.
  const std::vector<uint8_t> wanted(text.begin(), text.end());
  if (!wanted.empty()) {
    for (const int32_t i : this->tokens_by_first_byte.at(wanted.front())) {
      if (this->tokens_to_bytes[i] == wanted) {
        return static_cast<T>(i);
      }
    }
  }
  std::vector<T> tokens = text_to_tokens<T>(text);
  if (tokens.size() != 1) {
    std::string errorMessage =
        "Expected 1 token, got " + std::to_string(tokens.size()) + " tokens (";
    for (T token : tokens) {
      errorMessage += std::to_string(token) + ", ";
    }
    errorMessage += ") for text " + text;
    fprintf(stderr, "%s\n", errorMessage.c_str());
    throw std::runtime_error(errorMessage);
  }
  return tokens[0];
}

template int32_t BinTokenizer::text_to_special_token<int32_t>(
    const std::string &text);
template int64_t BinTokenizer::text_to_special_token<int64_t>(
    const std::string &text);

void BinTokenizer::build_first_byte_index() {
  this->tokens_by_first_byte.assign(256, std::vector<int32_t>());
  for (size_t i = 0; i < this->tokens_to_bytes.size(); i++) {
    const std::vector<uint8_t> &bytes = this->tokens_to_bytes[i];
    // An empty entry has no first byte to file it under, and it could never win
    // the longest match in text_to_tokens anyway.
    if (bytes.empty()) {
      continue;
    }
    this->tokens_by_first_byte.at(bytes.front())
        .push_back(static_cast<int32_t>(i));
  }
}

void BinTokenizer::build_merge_index() {
  // Find the byte fallback block: 256 consecutive entries holding the single
  // bytes 0x00 to 0xFF in order. Searched for rather than assumed at a fixed id
  // so that a vocabulary laying its control tokens out differently still works.
  this->byte_fallback_base = -1;
  const size_t count = this->tokens_to_bytes.size();
  for (size_t start = 0; start + 256 <= count; start++) {
    bool complete = true;
    for (size_t offset = 0; offset < 256; offset++) {
      const std::vector<uint8_t> &entry = this->tokens_to_bytes[start + offset];
      if (entry.size() != 1 || entry[0] != offset) {
        complete = false;
        break;
      }
    }
    if (complete) {
      this->byte_fallback_base = static_cast<int32_t>(start);
      break;
    }
  }
  if (this->byte_fallback_base < 0) {
    LOG("No byte fallback block in this vocabulary, so it cannot be byte-pair "
        "encoded; falling back to longest-match encoding.\n");
    this->encoding = BinTokenizerEncoding::kLongestMatch;
    return;
  }

  // Merging can only ever produce an entry that comes after the byte fallback
  // block: everything at or before it is a raw byte or a control token, and no
  // pair of adjacent pieces joins into one of those. Ids ascend, so the first
  // spelling wins and duplicates later in the vocabulary are ignored.
  const size_t first_piece =
      static_cast<size_t>(this->byte_fallback_base) + 256;
  this->merge_ids.reserve(count > first_piece ? count - first_piece : 0);
  for (size_t i = first_piece; i < count; i++) {
    const std::vector<uint8_t> &bytes = this->tokens_to_bytes[i];
    if (bytes.empty()) {
      continue;
    }
    this->merge_ids.emplace(std::string(bytes.begin(), bytes.end()),
                            static_cast<int32_t>(i));
  }
}

void BinTokenizer::build_unigram_index() {
  if (this->scores.size() != this->tokens_to_bytes.size()) {
    throw std::runtime_error(
        "Unigram encoding needs a score per token, which only a tokenizer.bin "
        "with a header carries");
  }
  this->byte_piece_ids.fill(-1);
  this->piece_ids.reserve(this->tokens_to_bytes.size());
  this->max_piece_bytes = 1;
  this->min_score = std::numeric_limits<float>::max();
  const auto hex_value = [](uint8_t c) -> int {
    if (c >= '0' && c <= '9') {
      return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
      return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
      return c - 'A' + 10;
    }
    return -1;
  };
  for (size_t i = 0; i < this->tokens_to_bytes.size(); i++) {
    const std::vector<uint8_t> &bytes = this->tokens_to_bytes[i];
    if (bytes.empty()) {
      continue;
    }
    // "<0xNN>" is the byte fallback piece SentencePiece writes for byte NN.
    // It is a spelling of the unknown, not of the text "<0xNN>", so it goes in
    // the byte table rather than the piece index.
    if (bytes.size() == 6 && bytes[0] == '<' && bytes[1] == '0' &&
        bytes[2] == 'x' && bytes[5] == '>' && hex_value(bytes[3]) >= 0 &&
        hex_value(bytes[4]) >= 0) {
      this->byte_piece_ids[static_cast<size_t>(hex_value(bytes[3]) * 16 +
                                               hex_value(bytes[4]))] =
          static_cast<int32_t>(i);
      continue;
    }
    this->piece_ids.emplace(std::string(bytes.begin(), bytes.end()),
                            static_cast<int32_t>(i));
    this->max_piece_bytes = std::max(this->max_piece_bytes, bytes.size());
    this->min_score = std::min(this->min_score, this->scores[i]);
  }
  if (this->min_score == std::numeric_limits<float>::max()) {
    this->min_score = 0.0f;
  }
}

template <typename T>
std::vector<T> BinTokenizer::text_to_tokens(const std::string &text) {
  if (this->encoding == BinTokenizerEncoding::kBpe) {
    return this->text_to_tokens_bpe<T>(text);
  }
  if (this->encoding == BinTokenizerEncoding::kUnigram) {
    return this->text_to_tokens_unigram<T>(text);
  }
  return this->text_to_tokens_longest_match<T>(text);
}

// Greedy longest-match encoding. Only the entries starting with the byte we are
// looking at can match, so we compare against that bucket rather than the whole
// vocabulary: key-term biasing encodes thousands of terms when a transcriber
// loads, which made the full scan the dominant cost of setting up a large list.
// Ties still go to the lowest token id, since buckets hold ascending ids and
// only a strictly longer match displaces the incumbent.
template <typename T>
std::vector<T> BinTokenizer::text_to_tokens_longest_match(
    const std::string &text) {
  std::vector<T> result;
  std::string replaced_spaces_text = replace_all(text, " ", space_string);
  std::vector<uint8_t> remaining_bytes(replaced_spaces_text.begin(),
                                       replaced_spaces_text.end());

  while (!remaining_bytes.empty()) {
    size_t longest_match_size = 0;
    T longest_match_token = -1;
    for (const int32_t i :
         this->tokens_by_first_byte.at(remaining_bytes.front())) {
      const std::vector<uint8_t> &bytes = this->tokens_to_bytes[i];
      if (remaining_bytes.size() < bytes.size()) {
        continue;
      }
      if (bytes.size() > longest_match_size &&
          std::equal(bytes.begin(), bytes.end(), remaining_bytes.begin())) {
        longest_match_size = bytes.size();
        longest_match_token = (T)i;
      }
    }
    if (longest_match_token == -1) {
      std::string errorMessage =
          "No match found for remaining bytes " +
          std::string(remaining_bytes.begin(), remaining_bytes.end()) + " (";
      for (uint8_t byte : remaining_bytes) {
        char hex_byte[5] = {0};
        snprintf(hex_byte, sizeof(hex_byte), "0x%02X", byte);
        errorMessage += std::string(hex_byte) + ", ";
      }
      errorMessage += ")";
      fprintf(stderr, "%s\n", errorMessage.c_str());
      throw std::runtime_error(errorMessage);
    }
    result.push_back(longest_match_token);
    remaining_bytes.erase(remaining_bytes.begin(),
                          remaining_bytes.begin() + longest_match_size);
  }

  return result;
}

// Byte-pair encoding, with a token's id standing in for its merge rank. Start
// from one piece per character and keep joining the adjacent pair whose
// combined spelling has the lowest id, which replays the merges in the order
// they were learned. Nothing is scanned per candidate here the way
// longest-match scans a first-byte bucket, because the question at each step is
// not "what is the longest entry starting here" but "is this exact pair a
// piece", which is one hash lookup.
//
// Quadratic in the number of pieces, which is fine for what this encodes: key
// terms and single words, a handful of characters each. Whole sentences would
// want a heap of candidate merges instead.
template <typename T>
std::vector<T> BinTokenizer::text_to_tokens_bpe(const std::string &text) {
  const std::string replaced_spaces_text = replace_all(text, " ", space_string);

  std::vector<std::string> pieces;
  for (size_t offset = 0; offset < replaced_spaces_text.size();) {
    const size_t length = std::min(utf8_sequence_length(static_cast<uint8_t>(
                                       replaced_spaces_text[offset])),
                                   replaced_spaces_text.size() - offset);
    pieces.push_back(replaced_spaces_text.substr(offset, length));
    offset += length;
  }

  std::string candidate;
  while (pieces.size() > 1) {
    int32_t best_id = -1;
    size_t best_position = 0;
    for (size_t position = 0; position + 1 < pieces.size(); position++) {
      candidate.assign(pieces[position]);
      candidate.append(pieces[position + 1]);
      const auto found = this->merge_ids.find(candidate);
      if (found != this->merge_ids.end() &&
          (best_id < 0 || found->second < best_id)) {
        best_id = found->second;
        best_position = position;
      }
    }
    if (best_id < 0) {
      break;
    }
    pieces[best_position].append(pieces[best_position + 1]);
    pieces.erase(pieces.begin() + best_position + 1);
  }

  std::vector<T> result;
  for (const std::string &piece : pieces) {
    const auto found = this->merge_ids.find(piece);
    if (found != this->merge_ids.end()) {
      result.push_back(static_cast<T>(found->second));
      continue;
    }
    // A character the merges cannot spell, so it goes out as its raw bytes.
    // Unlike longest-match encoding this never fails: every byte has an entry.
    for (const char byte : piece) {
      result.push_back(static_cast<T>(this->byte_fallback_base +
                                      static_cast<uint8_t>(byte)));
    }
  }
  return result;
}

// Unigram encoding, the SentencePiece way: the space marker is prepended,
// spaces become the marker, and the text is cut before each marker so every
// word is encoded on its own. Matches the ``tokenizers`` crate's Metaspace
// pre-tokenizer with `prepend_scheme: always` and `split: true`.
template <typename T>
std::vector<T> BinTokenizer::text_to_tokens_unigram(const std::string &text) {
  const std::string marker(this->space_string);
  std::string replaced = replace_all(text, " ", marker);
  // Prepended whether or not the text starts with a space, so "  x" gets
  // three markers, as the reference does.
  if (!replaced.empty()) {
    replaced.insert(0, marker);
  }
  std::vector<T> result;
  if (marker.empty()) {
    this->encode_unigram_word<T>(replaced, result);
    return result;
  }
  size_t start = 0;
  while (start < replaced.size()) {
    size_t next = replaced.find(marker, start + marker.size());
    if (next == std::string::npos) {
      next = replaced.size();
    }
    this->encode_unigram_word<T>(replaced.substr(start, next - start), result);
    start = next;
  }
  return result;
}

// Viterbi over the lattice of pieces covering `word`, one node per character
// boundary. Same tie-break as the ``tokenizers`` crate's
// `Unigram::encode_optimized`: candidates at a position are tried shortest
// first and only a strictly better score displaces one, so a tie keeps the
// shorter piece. A character with no single-character piece gets an unknown
// node at a penalty, which the byte fallback then spells out.
template <typename T>
void BinTokenizer::encode_unigram_word(const std::string &word,
                                       std::vector<T> &out) {
  // Byte offsets of every character start, plus one past the end.
  std::vector<size_t> bounds;
  bounds.reserve(word.size() + 1);
  for (size_t i = 0; i < word.size();) {
    bounds.push_back(i);
    i += std::min(utf8_sequence_length(static_cast<uint8_t>(word[i])),
                  word.size() - i);
  }
  bounds.push_back(word.size());
  const size_t n = bounds.size() - 1;
  if (n == 0) {
    return;
  }

  const float unreached = -std::numeric_limits<float>::infinity();
  const float unk_score = this->min_score - kUnigramUnkPenalty;
  std::vector<float> best(n + 1, unreached);
  std::vector<size_t> prev(n + 1, 0);
  std::vector<int32_t> ids(n + 1, this->unk_id);
  best[0] = 0.0f;
  std::string key;
  for (size_t i = 0; i < n; i++) {
    if (best[i] == unreached) {
      continue;
    }
    bool has_single = false;
    for (size_t j = i + 1; j <= n; j++) {
      const size_t length = bounds[j] - bounds[i];
      if (length > this->max_piece_bytes) {
        break;
      }
      key.assign(word, bounds[i], length);
      const auto found = this->piece_ids.find(key);
      if (found == this->piece_ids.end()) {
        continue;
      }
      if (j == i + 1) {
        has_single = true;
      }
      const float score = best[i] + this->scores[found->second];
      if (best[j] == unreached || score > best[j]) {
        best[j] = score;
        prev[j] = i;
        ids[j] = found->second;
      }
    }
    if (!has_single) {
      const float score = best[i] + unk_score;
      if (best[i + 1] == unreached || score > best[i + 1]) {
        best[i + 1] = score;
        prev[i + 1] = i;
        ids[i + 1] = this->unk_id;
      }
    }
  }

  // Walk the best path back from the end, then emit it forwards.
  std::vector<size_t> ends;
  for (size_t j = n; j > 0; j = prev[j]) {
    ends.push_back(j);
  }
  std::reverse(ends.begin(), ends.end());
  for (const size_t j : ends) {
    const int32_t id = ids[j];
    if (id != this->unk_id) {
      out.push_back(static_cast<T>(id));
      continue;
    }
    // An unknown span becomes its UTF-8 bytes when the vocabulary can spell
    // every one of them, otherwise the unknown id itself.
    const size_t a = bounds[prev[j]];
    const size_t b = bounds[j];
    bool all_bytes_known = true;
    for (size_t k = a; k < b; k++) {
      if (this->byte_piece_ids[static_cast<uint8_t>(word[k])] < 0) {
        all_bytes_known = false;
        break;
      }
    }
    if (!all_bytes_known) {
      if (this->unk_id < 0) {
        throw std::runtime_error("No piece or unknown id for text " + word);
      }
      out.push_back(static_cast<T>(this->unk_id));
      continue;
    }
    for (size_t k = a; k < b; k++) {
      out.push_back(
          static_cast<T>(this->byte_piece_ids[static_cast<uint8_t>(word[k])]));
    }
  }
}

// Instantiated after both encoders, so each is defined before the dispatcher
// that calls it is instantiated.
template std::vector<int64_t> BinTokenizer::text_to_tokens<int64_t>(
    const std::string &text);
template std::vector<int32_t> BinTokenizer::text_to_tokens<int32_t>(
    const std::string &text);

template <typename T>
std::string BinTokenizer::tokens_to_text(const std::vector<T> &tokens,
                                         bool skipSpecials) {
  std::vector<uint8_t> result_bytes;
  for (const auto &token : tokens) {
    std::vector<uint8_t> bytes = tokens_to_bytes.at(token);
    if (bytes.size() == 0) {
      throw std::runtime_error("Invalid token " + std::to_string(token));
    }
    if (skipSpecials && bytes.size() > 2 && bytes[0] == '<' &&
        bytes[bytes.size() - 1] == '>') {
      // This is a special token, not text, so skip it.
      continue;
    }
    result_bytes.insert(result_bytes.end(), bytes.begin(), bytes.end());
  }
  std::string result(result_bytes.begin(), result_bytes.end());
  result = replace_all(result, space_string, " ");
  result = trim(result);
  return result;
}

template std::string BinTokenizer::tokens_to_text<int32_t>(
    const std::vector<int32_t> &, bool);
template std::string BinTokenizer::tokens_to_text<int64_t>(
    const std::vector<int64_t> &, bool);
