// GENERATED FILE - DO NOT EDIT BY HAND.
//
// Regenerate with:
//   python3 scripts/generate-model-file-metadata.py
//
// This maps every downloadable model file's full CDN URL to its expected size
// (bytes) and CRC32C checksum (base64).
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
constexpr std::array<Entry, 146> kEntries = {{
    {"https://download.moonshine.ai/model/base-ar/quantized/base-ar/"
     "decoder_model_merged.ort",
     109424552, "u17PkA==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ar/quantized/base-ar/"
     "encoder_model.ort",
     31326824, "QgxL3Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ar/quantized/base-ar/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/"
     "decoder_model_merged.ort",
     109424400, "71VpRQ==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/"
     "decoder_with_attention.ort",
     109003328, "+5Kaeg==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/"
     "encoder_model.ort",
     31326816, "+MATVw==", "crc32c"},
    {"https://download.moonshine.ai/model/base-en/quantized/base-en/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-es/quantized/base-es/"
     "decoder_model_merged.ort",
     43612200, "9NUuYQ==", "crc32c"},
    {"https://download.moonshine.ai/model/base-es/quantized/base-es/"
     "encoder_model.ort",
     20964320, "2ngM7g==", "crc32c"},
    {"https://download.moonshine.ai/model/base-es/quantized/base-es/"
     "tokenizer.bin",
     241639, "EBC+Hw==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ja/quantized/base-ja/"
     "decoder_model_merged.ort",
     109424424, "FLBs0A==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ja/quantized/base-ja/"
     "encoder_model.ort",
     31326816, "LErsPw==", "crc32c"},
    {"https://download.moonshine.ai/model/base-ja/quantized/base-ja/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-uk/quantized/base-uk/"
     "decoder_model_merged.ort",
     109424424, "mAf6aA==", "crc32c"},
    {"https://download.moonshine.ai/model/base-uk/quantized/base-uk/"
     "encoder_model.ort",
     31326816, "V6nZMg==", "crc32c"},
    {"https://download.moonshine.ai/model/base-uk/quantized/base-uk/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-vi/quantized/base-vi/"
     "decoder_model_merged.ort",
     109424520, "0Q2nLA==", "crc32c"},
    {"https://download.moonshine.ai/model/base-vi/quantized/base-vi/"
     "encoder_model.ort",
     31326816, "v/aGsg==", "crc32c"},
    {"https://download.moonshine.ai/model/base-vi/quantized/base-vi/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/base-zh/quantized/base-zh/"
     "decoder_model_merged.ort",
     109424520, "CbdJ2A==", "crc32c"},
    {"https://download.moonshine.ai/model/base-zh/quantized/base-zh/"
     "encoder_model.ort",
     31326816, "MHjyrQ==", "crc32c"},
    {"https://download.moonshine.ai/model/base-zh/quantized/base-zh/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/diarization-community1/embedding.ort",
     6975920, "r5LVbQ==", "crc32c"},
    {"https://download.moonshine.ai/model/diarization-community1/"
     "segmentation.ort",
     1594080, "hcqhEA==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/model_q4.ort",
     197683216, "7LR/2g==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/"
     "model_quantized.ort",
     309797160, "aQOaDQ==", "crc32c"},
    {"https://download.moonshine.ai/model/embeddinggemma-300m/tokenizer.bin",
     2578500, "uuSrLw==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/adapter.ort",
     3651296, "Ds3Hsg==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/cross_kv.ort",
     11643776, "liqlOg==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/decoder_kv.ort",
     146972408, "GPnzeQ==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/decoder_kv_with_attention.ort",
     146813344, "2SQOQA==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/encoder.ort",
     94705376, "yWvDwA==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/frontend.model.ort",
     28720, "k1kKsQ==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/frontend.weights.ort",
     11889560, "w5QjIA==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/streaming_config.json",
     513, "+1bE9Q==", "crc32c"},
    {"https://download.moonshine.ai/model/medium-streaming-en/"
     "quantized_26_08_21/tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/adapter.ort",
     2869296, "rx5eAg==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/cross_kv.ort",
     5358752, "B9ac2w==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/decoder_kv.ort",
     61314512, "IRMZCQ==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/encoder.ort",
     44358376, "AI1jZg==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/frontend.model.ort",
     26776, "nGFUEw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/frontend.weights.ort",
     7769280, "kUSrbA==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/streaming_config.json",
     512, "Y/oEHw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-de/"
     "quantized_26_08_24/tokenizer.bin",
     103319, "x219Qw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/adapter.ort",
     2870368, "XlWjTg==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/cross_kv.ort",
     5356536, "5ySK8g==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/decoder_kv.ort",
     81878600, "Rn/VUA==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/decoder_kv_with_attention.ort",
     81766608, "IzJXVA==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/encoder.ort",
     44148576, "41D2jQ==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/frontend.model.ort",
     26944, "WW2/2g==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/frontend.weights.ort",
     7769464, "9pROuQ==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/streaming_config.json",
     512, "dPbFiw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-en/"
     "quantized_26_08_21/tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/adapter.ort",
     2869296, "ZCOnJw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/cross_kv.ort",
     5358752, "B6G3LA==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/decoder_kv.ort",
     61314512, "Q6TnrQ==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/encoder.ort",
     44358376, "EYLK1A==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/frontend.model.ort",
     26776, "bvdZXg==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/frontend.weights.ort",
     7769280, "ITuX4A==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/streaming_config.json",
     512, "Y/oEHw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-es/"
     "quantized_26_08_24/tokenizer.bin",
     102888, "/7v8NQ==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/adapter.ort",
     2869296, "GzlnQA==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/cross_kv.ort",
     5358752, "0btjiw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/decoder_kv.ort",
     61314512, "qXEgyA==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/encoder.ort",
     44358376, "fltsVQ==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/frontend.model.ort",
     31208, "RhG3gA==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/frontend.weights.ort",
     7769288, "VjqzBw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/streaming_config.json",
     512, "Y/oEHw==", "crc32c"},
    {"https://download.moonshine.ai/model/small-streaming-ja/"
     "quantized_26_08_23/tokenizer.bin",
     101836, "M94Ogg==", "crc32c"},
    {"https://download.moonshine.ai/model/spelling-en/spelling_cnn.ort",
     1664920, "fKPVyA==", "crc32c"},
    {"https://download.moonshine.ai/model/spelling-en/spelling_cnn_meta.json",
     622, "/G7lyg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/"
     "decoder_model_merged.ort",
     30412256, "LngmQA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/"
     "decoder_with_attention.ort",
     30092072, "LiFkSQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/"
     "encoder_model.ort",
     13281600, "jmn6JA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-en/quantized/tiny-en/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja/"
     "decoder_model_merged.ort",
     58327272, "yiH1qg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja/"
     "encoder_model.ort",
     13238184, "UJX6fA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ja/quantized/tiny-ja/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko/"
     "decoder_model_merged.ort",
     58327336, "sjwMCA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko/"
     "encoder_model.ort",
     13238176, "kczhwQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-ko/quantized/tiny-ko/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "adapter.ort",
     1318472, "Ba/M3A==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "cross_kv.ort",
     1288120, "BXUe/w==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "decoder_kv.ort",
     19717336, "s5bwZA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "encoder.ort",
     7772792, "6tp7Mg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "frontend.model.ort",
     23176, "lLmrsA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "frontend.weights.ort",
     2093280, "exI8jg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "streaming_config.json",
     509, "wH/VeA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ar/quantized_26_08_24/"
     "tokenizer.bin",
     135726, "y80toQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "adapter.ort",
     1318472, "Q5RVow==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "cross_kv.ort",
     1288120, "Y8sggQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "decoder_kv.ort",
     19717336, "0Btj7Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "encoder.ort",
     7772792, "UMxvQg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "frontend.model.ort",
     23176, "jBdx4Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "frontend.weights.ort",
     2093280, "Ljirxw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "streaming_config.json",
     509, "wH/VeA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-de/quantized_26_08_24/"
     "tokenizer.bin",
     103319, "x219Qw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "adapter.ort",
     1319664, "kwQ+Bw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "cross_kv.ort",
     1287544, "76wzFQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "decoder_kv.ort",
     32583720, "KJjeNw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "decoder_kv_with_attention.ort",
     32515016, "zFeWyQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "encoder.ort",
     7675440, "UjAIpQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "frontend.model.ort",
     23344, "aM/+wQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "frontend.weights.ort",
     2093464, "WYZ7EQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "streaming_config.json",
     509, "HGL0Ug==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-en/quantized_26_08_21/"
     "tokenizer.bin",
     249974, "B7s10Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "adapter.ort",
     1318472, "+JOBGg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "cross_kv.ort",
     1288120, "lvRCDw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "decoder_kv.ort",
     19717336, "KEm/xA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "encoder.ort",
     7772792, "AAXCVA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "frontend.model.ort",
     23176, "v7nfyQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "frontend.weights.ort",
     2093280, "LYbiaA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "streaming_config.json",
     509, "wH/VeA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-es/quantized_26_08_24/"
     "tokenizer.bin",
     102888, "/7v8NQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "adapter.ort",
     1318472, "obzOjg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "cross_kv.ort",
     1288120, "gbYCsw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "decoder_kv.ort",
     19717336, "Mc/YqQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "encoder.ort",
     7772792, "ZS0TPg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "frontend.model.ort",
     27608, "BTpcfg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "frontend.weights.ort",
     2093288, "T4l/3Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "streaming_config.json",
     509, "wH/VeA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-ja/quantized_26_08_23/"
     "tokenizer.bin",
     101836, "M94Ogg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "adapter.ort",
     1318472, "Zau+Ig==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "cross_kv.ort",
     1288120, "iwurdw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "decoder_kv.ort",
     19717336, "jsOHgA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "encoder.ort",
     7772792, "nx0MFw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "frontend.model.ort",
     27608, "s9Ro7g==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "frontend.weights.ort",
     2093288, "UEAV9Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "streaming_config.json",
     509, "wH/VeA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-tl/quantized_26_08_24/"
     "tokenizer.bin",
     91356, "4uONfQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "adapter.ort",
     1318472, "cdV7Qw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "cross_kv.ort",
     1288120, "aeNa0A==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "decoder_kv.ort",
     19717336, "792dSQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "encoder.ort",
     7772792, "5vA9hw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "frontend.model.ort",
     23176, "T4ZpJw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "frontend.weights.ort",
     2093280, "QeZo8Q==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "streaming_config.json",
     509, "wH/VeA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-vi/quantized_26_08_24/"
     "tokenizer.bin",
     95323, "7zvjaA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "adapter.ort",
     1318472, "hr7j6g==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "cross_kv.ort",
     1288120, "fiGyHw==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "decoder_kv.ort",
     19717336, "rr9O2A==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "encoder.ort",
     7772792, "BHPY3w==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "frontend.model.ort",
     27608, "4gF+xg==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "frontend.weights.ort",
     2090728, "Kv1kYQ==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "streaming_config.json",
     509, "wH/VeA==", "crc32c"},
    {"https://download.moonshine.ai/model/tiny-streaming-zh/quantized_26_08_24/"
     "tokenizer.bin",
     74587, "nQJxPQ==", "crc32c"},
}};

}  // namespace

ModelFileMetadata find_model_file_metadata(const std::string& url) {
  const std::string_view key(url);
  const auto* begin = kEntries.data();
  const auto* end = begin + kEntries.size();
  const auto* it = std::lower_bound(
      begin, end, key, [](const Entry& entry, std::string_view value) {
        return entry.url < value;
      });
  if (it != end && it->url == key) {
    return ModelFileMetadata{it->size, std::string(it->checksum),
                             std::string(it->checksum_type)};
  }
  return ModelFileMetadata{-1, "", ""};
}

}  // namespace moonshine
