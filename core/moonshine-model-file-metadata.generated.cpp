// GENERATED FILE - DO NOT EDIT BY HAND.
//
// Regenerate with:
//   python3 scripts/generate-model-file-metadata.py
//
// This maps every downloadable model file's full CDN URL to its expected size
// (bytes) and CRC32C checksum (base64, as reported by Google Cloud Storage).
// It is the single source of truth for per-file integrity metadata that
// moonshine-model-catalog.cpp joins into the download manifest.

#include "moonshine-model-file-metadata.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace moonshine {
namespace {

struct Entry {
  std::string_view url;
  int64_t size;
  std::string_view checksum;
  std::string_view checksum_type;
};

// Sorted by `url` (ascending) so lookups can binary-search.
constexpr std::array<Entry, 64> kEntries = {{
    {"https://download.moonshine.ai/model/base-ar/quantized/base-ar/decoder_model_merged.ort", 109424552, "u17PkA==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ar/quantized/base-ar/encoder_model.ort", 31326824, "QgxL3Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ar/quantized/base-ar/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/decoder_model_merged.ort", 109424400, "71VpRQ==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/decoder_with_attention.ort", 109003328, "+5Kaeg==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/encoder_model.ort", 31326816, "+MATVw==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-es/quantized/base-es/decoder_model_merged.ort", 43612200, "9NUuYQ==", "crc32c"},
    {"https://download.moonshine.ai/model/base-es/quantized/base-es/encoder_model.ort", 20964320, "2ngM7g==", "crc32c"},
    {"https://download.moonshine.ai/model/base-es/quantized/base-es/tokenizer.bin", 241639, "EBC+Hw==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ja/quantized/base-ja/decoder_model_merged.ort", 109424424, "FLBs0A==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ja/quantized/base-ja/encoder_model.ort", 31326816, "LErsPw==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ja/quantized/base-ja/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-uk/quantized/base-uk/decoder_model_merged.ort", 109424424, "mAf6aA==", "crc32c"},
    {"https://download.moonshine.ai/model/base-uk/quantized/base-uk/encoder_model.ort", 31326816, "V6nZMg==", "crc32c"},
    {"https://download.moonshine.ai/model/base-uk/quantized/base-uk/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-vi/quantized/base-vi/decoder_model_merged.ort", 109424520, "0Q2nLA==", "crc32c"},
    {"https://download.moonshine.ai/model/base-vi/quantized/base-vi/encoder_model.ort", 31326816, "v/aGsg==", "crc32c"},
    {"https://download.moonshine.ai/model/base-vi/quantized/base-vi/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-zh/quantized/base-zh/decoder_model_merged.ort", 109424520, "CbdJ2A==", "crc32c"},
    {"https://download.moonshine.ai/model/base-zh/quantized/base-zh/encoder_model.ort", 31326816, "MHjyrQ==", "crc32c"},
    {"https://download.moonshine.ai/model/base-zh/quantized/base-zh/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/model.ort", 1235247424, "ZKybVA==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/model_fp16.ort", 831143896, "g/tW1w==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/model_q4.ort", 197683216, "7LR/2g==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/model_q4f16.ort", 176777928, "lp47kA==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/model_quantized.ort", 309797160, "aQOaDQ==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/tokenizer.bin", 2578500, "uuSrLw==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/adapter.ort", 3647712, "T4ZPsw==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/cross_kv.ort", 11544952, "iy7jtw==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/decoder_kv.ort", 146216448, "jDzNQA==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/decoder_kv_with_attention.ort", 146138304, "UjHy9Q==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/encoder.ort", 94202872, "vGp6QA==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/frontend.ort", 47467256, "qm1RfA==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/streaming_config.json", 513, "+1bE9Q==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/quantized/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/adapter.ort", 2867424, "1HGn5A==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/cross_kv.ort", 5298736, "irqnmQ==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/decoder_kv.ort", 81435904, "jHbK+Q==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/decoder_kv_with_attention.ort", 81380336, "c5YAXw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/encoder.ort", 43853224, "pFunOg==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/frontend.ort", 30984200, "jmdzUw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/streaming_config.json", 512, "dPbFiw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/quantized/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/spelling-en/spelling_cnn.ort", 1664920, "fKPVyA==", "crc32c"},
    {"https://download.moonshine.ai/model/spelling-en/spelling_cnn_meta.json", 622, "/G7lyg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/decoder_model_merged.ort", 30412256, "LngmQA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/decoder_with_attention.ort", 30092072, "LiFkSQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/encoder_model.ort", 13281600, "jmn6JA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja/decoder_model_merged.ort", 58327272, "yiH1qg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja/encoder_model.ort", 13238184, "UJX6fA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko/decoder_model_merged.ort", 58327336, "sjwMCA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko/encoder_model.ort", 13238176, "kczhwQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/adapter.ort", 1319440, "Rz/GmQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/cross_kv.ort", 1264384, "y57xZw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/decoder_kv.ort", 32403688, "saXmVw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/decoder_kv_with_attention.ort", 32370152, "eqRTlw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/encoder.ort", 7569200, "njDK5A==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/frontend.ort", 8324600, "/61fjA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/streaming_config.json", 509, "HGL0Ug==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized/tokenizer.bin", 249974, "B7s10Q==", "crc32c"},
}};

}  // namespace

ModelFileMetadata find_model_file_metadata(const std::string& url) {
  const std::string_view key(url);
  const auto* begin = kEntries.data();
  const auto* end = begin + kEntries.size();
  const auto* it = std::lower_bound(
      begin, end, key,
      [](const Entry& entry, std::string_view value) {
        return entry.url < value;
      });
  if (it != end && it->url == key) {
    return ModelFileMetadata{it->size, std::string(it->checksum),
                             std::string(it->checksum_type)};
  }
  return ModelFileMetadata{-1, "", ""};
}

}  // namespace moonshine
