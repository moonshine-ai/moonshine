#ifndef SPEAKER_EMBEDDING_MODEL_H
#define SPEAKER_EMBEDDING_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include <mutex>
#include <vector>

#include "onnxruntime_c_api.h"

struct SpeakerEmbeddingModel {
  static constexpr size_t ideal_input_size = 80000;
  static constexpr size_t embedding_size = 512;
  static constexpr int32_t input_sample_rate = 16000;

  const OrtApi *ort_api;
  OrtEnv *ort_env;
  OrtSessionOptions *ort_session_options;
  OrtMemoryInfo *ort_memory_info;
  OrtSession *embedding_session;
  bool log_ort_run;
  std::mutex processing_mutex;

  const char *embedding_mmapped_data = nullptr;
  size_t embedding_mmapped_data_size = 0;

  SpeakerEmbeddingModel(bool log_ort_run = false);
  ~SpeakerEmbeddingModel();

  int load(const char *speaker_embedding_model_path);
  int load_from_memory(const uint8_t *speaker_embedding_model_data,
                       size_t speaker_embedding_model_data_size);

  int calculate_embedding(const float *input_audio_data,
                          size_t input_audio_data_size,
                          std::vector<float> *out_embedding);
};

#endif