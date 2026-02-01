#include "speaker-embedding-model.h"

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "debug-utils.h"
#include "ort-utils.h"
#include "moonshine-tensor-view.h"

SpeakerEmbeddingModel::SpeakerEmbeddingModel(bool log_ort_run)
    : embedding_session(nullptr), log_ort_run(log_ort_run) {
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  LOG_ORT_ERROR(ort_api, ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                            "SpeakerEmbeddingModel", &ort_env));
  LOG_ORT_ERROR(ort_api,
                ort_api->CreateCpuMemoryInfo(
                    OrtDeviceAllocator, OrtMemTypeDefault, &ort_memory_info));

  LOG_ORT_ERROR(ort_api, ort_api->CreateSessionOptions(&ort_session_options));
  LOG_ORT_ERROR(ort_api, ort_api->SetSessionGraphOptimizationLevel(
                             ort_session_options, ORT_ENABLE_EXTENDED));
  LOG_ORT_ERROR(ort_api,
                ort_api->AddSessionConfigEntry(
                    ort_session_options, "session.load_model_format", "ORT"));
  LOG_ORT_ERROR(ort_api, ort_api->AddSessionConfigEntry(
                             ort_session_options,
                             "session.use_ort_model_bytes_directly", "1"));
  LOG_ORT_ERROR(ort_api,
                ort_api->AddSessionConfigEntry(
                    ort_session_options, "session.disable_prepacking", "1"));
  LOG_ORT_ERROR(ort_api, ort_api->DisableCpuMemArena(ort_session_options));
}

SpeakerEmbeddingModel::~SpeakerEmbeddingModel() {
  ort_api->ReleaseEnv(ort_env);
  ort_api->ReleaseMemoryInfo(ort_memory_info);
  ort_api->ReleaseSessionOptions(ort_session_options);
  ort_api->ReleaseSession(embedding_session);
#ifndef _WIN32
  if (embedding_mmapped_data) {
    munmap(const_cast<char *>(embedding_mmapped_data),
           embedding_mmapped_data_size);
  }
#endif
}

int SpeakerEmbeddingModel::load(const char *embedding_model_path) {
  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, embedding_model_path,
      &embedding_session, &embedding_mmapped_data, &embedding_mmapped_data_size));
  RETURN_ON_NULL(embedding_session);
  return 0;
}

int SpeakerEmbeddingModel::load_from_memory(const uint8_t *embedding_model_data,
                                            size_t embedding_model_data_size) {
  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, embedding_model_data,
      embedding_model_data_size, &embedding_session));
  RETURN_ON_NULL(embedding_session);
  return 0;
}

int SpeakerEmbeddingModel::calculate_embedding(
    const float *input_audio_data, size_t input_audio_data_size,
    std::vector<float> *out_embedding) {
  RETURN_ON_NULL(out_embedding);
  std::vector<float> padded_input_audio_data;
  // If the input audio is too short, extend it by repeating the audio data.
  if (input_audio_data_size < ideal_input_size) {
    padded_input_audio_data.resize(ideal_input_size);
    for (size_t offset = 0; offset < (ideal_input_size - input_audio_data_size); offset += input_audio_data_size) {
      std::copy(input_audio_data, input_audio_data + input_audio_data_size,
                padded_input_audio_data.data() + offset);
    }
    input_audio_data = padded_input_audio_data.data();
    input_audio_data_size = ideal_input_size;
  }
  const std::vector<int64_t> embedding_input_shape = {
      1, static_cast<int64_t>(input_audio_data_size)};
  MoonshineTensorView *embedding_input_tensor = new MoonshineTensorView(
      embedding_input_shape, ort_get_input_type(ort_api, embedding_session, 0),
      const_cast<float *>(input_audio_data), "waveform");
  OrtValue *embedding_input =
      embedding_input_tensor->create_ort_value(ort_api, ort_memory_info);

  const char *embedding_input_name = "waveform";
  const char *embedding_output_name = "embeddings";

  OrtValue *embedding_output = nullptr;
  // TIMER_START(speaker_embedding_run);
  RETURN_ON_ORT_ERROR(
      ort_api,
      ORT_RUN(ort_api, embedding_session, &embedding_input_name, &embedding_input, 1,
              &embedding_output_name, 1, &embedding_output));
  // TIMER_END(speaker_embedding_run);

  ort_api->ReleaseValue(embedding_input);
  delete embedding_input_tensor;
  embedding_input_tensor = nullptr;

  MoonshineTensorView *embedding_tensor =
      new MoonshineTensorView(ort_api, embedding_output, "embedding_tensor");
  ort_api->ReleaseValue(embedding_output);
  out_embedding->clear();
  out_embedding->insert(
      out_embedding->end(), embedding_tensor->data<float>(),
      embedding_tensor->data<float>() + embedding_tensor->element_count());
  delete embedding_tensor;
  return 0;
}
