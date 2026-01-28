#ifndef MOONSHINE_MODEL_H
#define MOONSHINE_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include <mutex>

#include "bin-tokenizer.h"
#include "moonshine-c-api.h"
#include "moonshine-ort-allocator.h"
#include "onnxruntime_c_api.h"

struct MoonshineModel {
  const OrtApi *ort_api;
  OrtEnv *ort_env;
  OrtSessionOptions *ort_session_options;
  OrtMemoryInfo *ort_memory_info;
  // Separate allocators for session and string allocations to avoid memory
  // fragmentation.
  MoonshineOrtAllocator *ort_session_allocator;
  MoonshineOrtAllocator *ort_string_allocator;
  OrtSession *encoder_session;
  OrtSession *decoder_session;
  BinTokenizer *tokenizer;
  std::mutex processing_mutex;

  int32_t num_layers;
  int32_t num_kv_heads;
  int32_t head_dim;
  int32_t past_element_count;

  const char *encoder_mmapped_data = nullptr;
  size_t encoder_mmapped_data_size = 0;

  const char *decoder_mmapped_data = nullptr;
  size_t decoder_mmapped_data_size = 0;

  std::vector<float> stream_audio_data;
  bool stream_active = false;

  float max_tokens_per_second = 6.5f;

  std::vector<transcript_line_t> lines;

  std::string last_result;

  bool log_ort_run = false;

  MoonshineModel(bool log_ort_run = false, float max_tokens_per_second = 6.5f);
  ~MoonshineModel();

  int load(const char *encoder_model_path, const char *decoder_model_path,
           const char *tokenizer_path, int32_t model_type);

  int load_from_memory(const uint8_t *encoder_model_data,
                       size_t encoder_model_data_size,
                       const uint8_t *decoder_model_data,
                       size_t decoder_model_data_size,
                       const uint8_t *tokenizer_data,
                       size_t tokenizer_data_size, int32_t model_type);

#if defined(ANDROID)
  int load_from_assets(const char *encoder_model_path,
                       const char *decoder_model_path,
                       const char *tokenizer_path, int32_t model_type,
                       AAssetManager *assetManager);
#endif

  int transcribe(const float *input_audio_data, size_t input_audio_data_size,
                 char **out_text);

  int transcribe_wav(const char *wav_path, char **out_text);
};

#endif