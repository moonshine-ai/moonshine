#include "bin-tokenizer.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "debug-utils.h"
#include "file-utils.h"
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

}  // namespace

BinTokenizer::BinTokenizer(const char *tokenizer_path, const char *space_string,
                           BinTokenizerEncoding encoding) {
  this->space_string = space_string;
  this->encoding = encoding;
  FILE *file = std::fopen(tokenizer_path, "rb");
  if (!file) {
    std::string message =
        "Failed to open tokenizer file at " + std::string(tokenizer_path);
    std::perror(message.c_str());
    throw std::runtime_error(message);
  }
  while (true) {
    uint8_t first_byte;
    if (std::fread(&first_byte, 1, 1, file) != 1) {
      break;
    }
    if (first_byte == 0) {
      tokens_to_bytes.push_back(std::vector<uint8_t>());
      continue;
    }
    size_t byte_count;
    if (first_byte < 128) {
      byte_count = first_byte;
    } else {
      uint8_t second_byte;
      fread_exact(&second_byte, 1, 1, file, "tokenizer length byte");
      byte_count = (second_byte * 128) + first_byte - 128;
    }
    std::vector<uint8_t> bytes(byte_count);
    fread_exact(bytes.data(), 1, byte_count, file, "tokenizer token bytes");
    tokens_to_bytes.push_back(bytes);
  }
  std::fclose(file);
  if (tokens_to_bytes.size() == 0) {
    throw std::runtime_error("No tokens found in tokenizer file '" +
                             std::string(tokenizer_path) + "'");
  }
  this->build_first_byte_index();
  if (this->encoding == BinTokenizerEncoding::kBpe) {
    this->build_merge_index();
  }
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
  size_t offset = 0;
  while (offset < tokenizer_data_size) {
    uint8_t first_byte = tokenizer_data[offset];
    offset++;
    if (first_byte == 0) {
      tokens_to_bytes.push_back(std::vector<uint8_t>());
      continue;
    }
    size_t byte_count;
    if (first_byte < 128) {
      byte_count = first_byte;
    } else {
      if (offset >= tokenizer_data_size) {
        throw std::runtime_error(
            "Truncated tokenizer data: missing length byte at offset " +
            std::to_string(offset));
      }
      uint8_t second_byte = tokenizer_data[offset];
      byte_count = (second_byte * 128) + first_byte - 128;
      offset++;
    }
    // Guard against a truncated blob so the memcpy below cannot read past the
    // end of the input (offset is always <= tokenizer_data_size here, so the
    // subtraction cannot underflow).
    if (byte_count > tokenizer_data_size - offset) {
      throw std::runtime_error(
          "Truncated tokenizer data: token of " + std::to_string(byte_count) +
          " bytes at offset " + std::to_string(offset) +
          " exceeds input size " + std::to_string(tokenizer_data_size));
    }
    std::vector<uint8_t> bytes(byte_count);
    // A two-byte length can legitimately encode a zero-length token; skip the
    // copy in that case, since bytes.data() is null for an empty vector and
    // memcpy() is declared nonnull (passing null is UB even when count is 0).
    if (byte_count > 0) {
      std::memcpy(bytes.data(), tokenizer_data + offset, byte_count);
    }
    offset += byte_count;
    tokens_to_bytes.push_back(bytes);
  }
  if (tokens_to_bytes.size() == 0) {
    throw std::runtime_error(
        "No tokens found in tokenizer input data of size " +
        std::to_string(tokenizer_data_size));
  }
  this->build_first_byte_index();
  if (this->encoding == BinTokenizerEncoding::kBpe) {
    this->build_merge_index();
  }
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
  while (true) {
    uint8_t first_byte;
    if (AAsset_read(asset, &first_byte, 1) != 1) {
      break;
    }
    if (first_byte == 0) {
      tokens_to_bytes.push_back(std::vector<uint8_t>());
      continue;
    }
    size_t byte_count;
    if (first_byte < 128) {
      byte_count = first_byte;
    } else {
      uint8_t second_byte;
      AAsset_read(asset, &second_byte, 1);
      byte_count = (second_byte * 128) + first_byte - 128;
    }
    std::vector<uint8_t> bytes(byte_count);
    AAsset_read(asset, bytes.data(), byte_count);
    tokens_to_bytes.push_back(bytes);
  }
  AAsset_close(asset);
  if (tokens_to_bytes.size() == 0) {
    throw std::runtime_error("No data found in tokenizer file at " +
                             std::string(tokenizer_path));
  }
  this->build_first_byte_index();
  if (this->encoding == BinTokenizerEncoding::kBpe) {
    this->build_merge_index();
  }
}
#endif

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

template <typename T>
std::vector<T> BinTokenizer::text_to_tokens(const std::string &text) {
  if (this->encoding == BinTokenizerEncoding::kBpe) {
    return this->text_to_tokens_bpe<T>(text);
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