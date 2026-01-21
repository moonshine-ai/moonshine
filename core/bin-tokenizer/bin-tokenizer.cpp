#include "bin-tokenizer.h"

#include <cstdint>
#include <cstdlib>
#include <stdexcept>

#include "debug-utils.h"
#include "string-utils.h"

BinTokenizer::BinTokenizer(const char *tokenizer_path,
                           const char *space_string) {
  this->space_string = space_string;
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
      std::fread(&second_byte, 1, 1, file);
      byte_count = (second_byte * 128) + first_byte - 128;
    }
    std::vector<uint8_t> bytes(byte_count);
    std::fread(bytes.data(), 1, byte_count, file);
    tokens_to_bytes.push_back(bytes);
  }
  std::fclose(file);
  if (tokens_to_bytes.size() == 0) {
    throw std::runtime_error("No tokens found in tokenizer file '" +
                             std::string(tokenizer_path) + "'");
  }
}

BinTokenizer::BinTokenizer(const uint8_t *tokenizer_data,
                           size_t tokenizer_data_size,
                           const char *space_string) {
  this->space_string = space_string;
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
      uint8_t second_byte = tokenizer_data[offset];
      byte_count = (second_byte * 128) + first_byte - 128;
      offset++;
    }
    std::vector<uint8_t> bytes(byte_count);
    std::memcpy(bytes.data(), tokenizer_data + offset, byte_count);
    offset += byte_count;
    tokens_to_bytes.push_back(bytes);
  }
  if (tokens_to_bytes.size() == 0) {
    throw std::runtime_error(
        "No tokens found in tokenizer input data of size " +
        std::to_string(tokenizer_data_size));
  }
}

#if defined(ANDROID)
BinTokenizer::BinTokenizer(const char *tokenizer_path,
                           AAssetManager *assetManager,
                           const char *space_string) {
  this->space_string = space_string;
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
}
#endif

template <typename T>
T BinTokenizer::text_to_special_token(const std::string &text) {
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

// Uses a naive algorithm to encode text into tokens.
// This is not the most efficient way to do it, but it's functional and
// unlikely to be a performance bottleneck. If it becomes one, we can use
// all sorts of fun data structures to make it faster.
template <typename T>
std::vector<T> BinTokenizer::text_to_tokens(const std::string &text) {
  std::vector<T> result;
  std::string replaced_spaces_text = replace_all(text, " ", space_string);
  std::vector<uint8_t> remaining_bytes(replaced_spaces_text.begin(),
                                       replaced_spaces_text.end());

  while (!remaining_bytes.empty()) {
    std::vector<uint8_t> longest_match;
    T longest_match_token = -1;
    for (size_t i = 0; i < tokens_to_bytes.size(); i++) {
      std::vector<uint8_t> bytes = tokens_to_bytes[i];
      if (remaining_bytes.size() < bytes.size()) {
        continue;
      }
      if (std::equal(remaining_bytes.begin(),
                     remaining_bytes.begin() + bytes.size(), bytes.begin())) {
        if (bytes.size() > longest_match.size()) {
          longest_match = bytes;
          longest_match_token = (T)i;
        }
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
                          remaining_bytes.begin() + longest_match.size());
  }

  return result;
}

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