#ifndef BIN_TOKENIZER_H
#define BIN_TOKENIZER_H

#include <cstdint>
#include <string>
#include <vector>

#if defined(ANDROID)
#include <android/asset_manager.h>
#endif

struct BinTokenizer {
  std::vector<std::vector<uint8_t>> tokens_to_bytes;
  const char *space_string;

  BinTokenizer(const char *tokenizer_path, const char *space_string = "▁");
  BinTokenizer(const uint8_t *tokenizer_data, size_t tokenizer_data_size,
               const char *space_string = "▁");
#if defined(ANDROID)
  BinTokenizer(const char *tokenizer_path, AAssetManager *assetManager,
               const char *space_string = "▁");
#endif
  template <typename T>
  std::vector<T> text_to_tokens(const std::string &text);
  template <typename T>
  std::string tokens_to_text(const std::vector<T> &tokens,
                             bool skipSpecials = true);

  template <typename T>
  T text_to_special_token(const std::string &text);
};

#endif