#include "silero-vad.h"

#include "ort-utils.h"
#include "silero-vad-model-data.h"

void SileroVad::init_onnx_env() {
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  LOG_ORT_ERROR(ort_api, ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                            "SileroVAD", &env));
  LOG_ORT_ERROR(ort_api, ort_api->CreateSessionOptions(&session_options));
  LOG_ORT_ERROR(ort_api, ort_api->SetIntraOpNumThreads(session_options, 1));
  LOG_ORT_ERROR(ort_api, ort_api->SetInterOpNumThreads(session_options, 1));
  LOG_ORT_ERROR(ort_api, ort_api->SetSessionGraphOptimizationLevel(
                             session_options, ORT_ENABLE_ALL));
  LOG_ORT_ERROR(ort_api, ort_api->CreateCpuMemoryInfo(
                             OrtArenaAllocator, OrtMemTypeCPU, &memory_info));
  LOG_ORT_ERROR(ort_api, ort_api->GetAllocatorWithDefaultOptions(&allocator));
}

// Initializes threading settings.
void SileroVad::init_engine_threads(int inter_threads, int intra_threads) {
  LOG_ORT_ERROR(ort_api,
                ort_api->SetIntraOpNumThreads(session_options, intra_threads));
  LOG_ORT_ERROR(ort_api,
                ort_api->SetInterOpNumThreads(session_options, inter_threads));
  LOG_ORT_ERROR(ort_api, ort_api->SetSessionGraphOptimizationLevel(
                             session_options, ORT_ENABLE_ALL));
}

SileroVad::SileroVad(int sample_rate, int windows_frame_size, float threshold,
                     int min_silence_duration_ms, int speech_pad_ms,
                     int min_speech_duration_ms, float max_speech_duration_s)
    : ort_api(nullptr),
      env(nullptr),
      session_options(nullptr),
      session(nullptr),
      allocator(nullptr),
      memory_info(nullptr),
      threshold(threshold),
      speech_pad_samples(speech_pad_ms) {
  sr_per_ms = sample_rate / 1000;  // e.g., 16000 / 1000 = 16
  window_size_samples =
      windows_frame_size * sr_per_ms;  // e.g., 32ms * 16 = 512 samples
  effective_window_size =
      window_size_samples + context_samples;  // 512 + 64 = 576 samples
  input_node_dims[0] = 1;
  input_node_dims[1] = effective_window_size;
  _state.resize(size_state);
  _context.resize(context_samples, 0.0f);  // Initialize context to zeros
  sr = sample_rate;                        // scalar
  min_speech_samples = sr_per_ms * min_speech_duration_ms;
  max_speech_samples = (sample_rate * max_speech_duration_s -
                        window_size_samples - 2 * speech_pad_samples);
  min_silence_samples = sr_per_ms * min_silence_duration_ms;
  min_silence_samples_at_max_speech = sr_per_ms * 98;
  // Load model from embedded data
  load_from_memory(silero_vad_onnx, silero_vad_onnx_len);
}

int SileroVad::load_from_memory(const uint8_t *model_data,
                                size_t model_data_size) {
  init_onnx_env();
  return ort_session_from_memory(ort_api, env, session_options, model_data,
                                 model_data_size, &session);
}

SileroVad::~SileroVad() {
  if (session) ort_api->ReleaseSession(session);
  if (session_options) ort_api->ReleaseSessionOptions(session_options);
  if (env) ort_api->ReleaseEnv(env);
  if (memory_info) ort_api->ReleaseMemoryInfo(memory_info);
  // allocator is owned by ORT, do not release
}

// Inference: runs inference on one chunk of input data.
// data_chunk is expected to have window_size_samples samples (e.g., 512 for
// 16kHz).
void SileroVad::predict(const std::vector<float> &data_chunk,
                        float *out_probability, int *out_flag) {
  // Build input by prepending context (64 samples) to data_chunk (512 samples)
  // = 576 total
  input.resize(effective_window_size);
  std::copy(_context.begin(), _context.end(), input.begin());
  std::copy(data_chunk.begin(), data_chunk.end(),
            input.begin() + context_samples);

  // Create input tensor
  OrtValue *input_ort = nullptr;
  OrtValue *state_ort = nullptr;
  OrtValue *sr_ort = nullptr;

  OrtStatus *status = nullptr;

  status = ort_api->CreateTensorWithDataAsOrtValue(
      memory_info, input.data(), input.size() * sizeof(float), input_node_dims,
      2, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_ort);
  if (status != nullptr) {
    const char *msg = ort_api->GetErrorMessage(status);
    fprintf(stderr, "CreateTensorWithDataAsOrtValue (input) failed: %s\n", msg);
    ort_api->ReleaseStatus(status);
    return;
  }

  status = ort_api->CreateTensorWithDataAsOrtValue(
      memory_info, _state.data(), _state.size() * sizeof(float),
      state_node_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_ort);
  if (status != nullptr) {
    const char *msg = ort_api->GetErrorMessage(status);
    fprintf(stderr, "CreateTensorWithDataAsOrtValue (state) failed: %s\n", msg);
    ort_api->ReleaseStatus(status);
    ort_api->ReleaseValue(input_ort);
    return;
  }

  // Create scalar tensor for sample rate (empty shape = scalar)
  status = ort_api->CreateTensorWithDataAsOrtValue(
      memory_info, &sr, sizeof(int64_t), nullptr, 0,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_ort);
  if (status != nullptr) {
    const char *msg = ort_api->GetErrorMessage(status);
    fprintf(stderr, "CreateTensorWithDataAsOrtValue (sr) failed: %s\n", msg);
    ort_api->ReleaseStatus(status);
    ort_api->ReleaseValue(input_ort);
    ort_api->ReleaseValue(state_ort);
    return;
  }

  ort_inputs.clear();
  ort_inputs.push_back(input_ort);
  ort_inputs.push_back(state_ort);
  ort_inputs.push_back(sr_ort);

  // Prepare output OrtValue* array
  OrtValue *output_ort[2] = {nullptr, nullptr};
  status =
      ort_api->Run(session, nullptr, input_node_names.data(), ort_inputs.data(),
                   ort_inputs.size(), output_node_names.data(),
                   output_node_names.size(), output_ort);

  if (status != nullptr) {
    const char *msg = ort_api->GetErrorMessage(status);
    fprintf(stderr, "Run failed: %s\n", msg);
    ort_api->ReleaseStatus(status);
    ort_api->ReleaseValue(input_ort);
    ort_api->ReleaseValue(state_ort);
    ort_api->ReleaseValue(sr_ort);
    return;
  }

  float *speech_prob_ptr = nullptr;
  LOG_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                             output_ort[0], (void **)&speech_prob_ptr));
  float speech_prob = speech_prob_ptr[0];

  float *stateN = nullptr;
  LOG_ORT_ERROR(ort_api,
                ort_api->GetTensorMutableData(output_ort[1], (void **)&stateN));
  std::memcpy(_state.data(), stateN, size_state * sizeof(float));

  // Update context with last context_samples samples of the full input
  std::copy(input.end() - context_samples, input.end(), _context.begin());

  // Set output values
  if (out_probability) *out_probability = speech_prob;
  if (out_flag) *out_flag = (speech_prob >= threshold) ? 1 : 0;

  // Release OrtValues
  ort_api->ReleaseValue(input_ort);
  ort_api->ReleaseValue(state_ort);
  ort_api->ReleaseValue(sr_ort);
  ort_api->ReleaseValue(output_ort[0]);
  ort_api->ReleaseValue(output_ort[1]);
}
