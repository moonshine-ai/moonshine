#include "moonshine-streaming-model.h"

#include <fcntl.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <fstream>
#include <sstream>

#include "bin-tokenizer.h"
#include "moonshine-ort-allocator.h"
#include "string-utils.h"

#define DEBUG_ALLOC_ENABLED 1
#include "debug-utils.h"
#include "ort-utils.h"

// Streaming model constants
#define MOONSHINE_STREAMING_TINY_ENCODER_DIM 288
#define MOONSHINE_STREAMING_TINY_DECODER_DIM 288
#define MOONSHINE_STREAMING_TINY_DEPTH 6
#define MOONSHINE_STREAMING_TINY_NHEADS 8
#define MOONSHINE_STREAMING_TINY_HEAD_DIM 36

#define MOONSHINE_STREAMING_BASE_ENCODER_DIM 416
#define MOONSHINE_STREAMING_BASE_DECODER_DIM 416
#define MOONSHINE_STREAMING_BASE_DEPTH 8
#define MOONSHINE_STREAMING_BASE_NHEADS 8
#define MOONSHINE_STREAMING_BASE_HEAD_DIM 52

#define MOONSHINE_DECODER_START_TOKEN_ID 1
#define MOONSHINE_EOS_TOKEN_ID 2

/* ============================================================================
 * Helper Functions
 * ============================================================================
 */

static std::string read_file_to_string(const std::string &path) {
  std::ifstream f(path);
  if (!f.good()) return "";
  std::stringstream buffer;
  buffer << f.rdbuf();
  return buffer.str();
}

// TODO Use constants instead of loading config JSON
static bool parse_config_json(const std::string &json,
                              MoonshineStreamingConfig *config) {
  // Simple key-value extraction for our known config format
  auto get_int = [&json](const char *key) -> int {
    std::string search = std::string("\"") + key + "\":";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return 0;
    pos += search.length();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    int val = 0;
    bool negative = false;
    if (json[pos] == '-') {
      negative = true;
      pos++;
    }
    while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
      val = val * 10 + (json[pos] - '0');
      pos++;
    }
    return negative ? -val : val;
  };

  config->encoder_dim = get_int("encoder_dim");
  config->decoder_dim = get_int("decoder_dim");
  config->depth = get_int("depth");
  config->nheads = get_int("nheads");
  config->head_dim = get_int("head_dim");
  config->vocab_size = get_int("vocab_size");
  config->bos_id = get_int("bos_id");
  config->eos_id = get_int("eos_id");
  config->frame_len = get_int("frame_len");
  config->total_lookahead = get_int("total_lookahead");
  config->d_model_frontend = get_int("d_model_frontend");
  config->c1 = get_int("c1");
  config->c2 = get_int("c2");

  // max_seq_len defaults to 448 if not in config
  int max_seq = get_int("max_seq_len");
  config->max_seq_len = (max_seq > 0) ? max_seq : 448;

  // Validate essential fields
  return config->depth > 0 && config->decoder_dim > 0 && config->vocab_size > 0;
}

/* ============================================================================
 * MoonshineStreamingState Implementation
 * ============================================================================
 */

void MoonshineStreamingState::reset(const MoonshineStreamingConfig &cfg) {
  // Frontend state
  sample_buffer.assign(79, 0.0f);
  sample_len = 0;
  conv1_buffer.assign(cfg.d_model_frontend * 4, 0.0f);
  conv2_buffer.assign(cfg.c1 * 4, 0.0f);
  frame_count = 0;

  // Feature accumulator
  accumulated_features.clear();
  accumulated_feature_count = 0;

  // Encoder tracking
  encoder_frames_emitted = 0;

  // Adapter position
  adapter_pos_offset = 0;

  // Memory
  memory.clear();
  memory_len = 0;

  // Decoder cache
  k_self.clear();
  v_self.clear();
  cache_seq_len = 0;

  // Cross-attention KV cache
  k_cross.clear();
  v_cross.clear();
  cross_len = 0;
  cross_kv_valid = false;
}

/* ============================================================================
 * MoonshineStreamingModel Implementation
 * ============================================================================
 */

MoonshineStreamingModel::MoonshineStreamingModel(bool log_ort_run)
    : frontend_session(nullptr),
      encoder_session(nullptr),
      adapter_session(nullptr),
      cross_kv_session(nullptr),
      decoder_kv_session(nullptr),
      tokenizer(nullptr),
      log_ort_run(log_ort_run) {
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  LOG_ORT_ERROR(ort_api,
                ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                   "MoonshineStreamingModel", &ort_env));
  LOG_ORT_ERROR(ort_api,
                ort_api->CreateCpuMemoryInfo(
                    OrtDeviceAllocator, OrtMemTypeDefault, &ort_memory_info));
  ort_allocator = new MoonshineOrtAllocator(ort_memory_info);

  LOG_ORT_ERROR(ort_api, ort_api->CreateSessionOptions(&ort_session_options));
  LOG_ORT_ERROR(ort_api, ort_api->SetSessionGraphOptimizationLevel(
                             ort_session_options, ORT_ENABLE_ALL));

  memset(&config, 0, sizeof(config));
}

MoonshineStreamingModel::~MoonshineStreamingModel() {
  ort_api->ReleaseEnv(ort_env);
  ort_api->ReleaseMemoryInfo(ort_memory_info);
  ort_api->ReleaseSessionOptions(ort_session_options);
  if (frontend_session) ort_api->ReleaseSession(frontend_session);
  if (encoder_session) ort_api->ReleaseSession(encoder_session);
  if (adapter_session) ort_api->ReleaseSession(adapter_session);
  if (cross_kv_session) ort_api->ReleaseSession(cross_kv_session);
  if (decoder_kv_session) ort_api->ReleaseSession(decoder_kv_session);
  delete ort_allocator;
  delete tokenizer;
#ifndef _WIN32
  if (frontend_mmapped_data) {
    munmap(const_cast<char *>(frontend_mmapped_data),
           frontend_mmapped_data_size);
  }
  if (encoder_mmapped_data) {
    munmap(const_cast<char *>(encoder_mmapped_data), encoder_mmapped_data_size);
  }
  if (adapter_mmapped_data) {
    munmap(const_cast<char *>(adapter_mmapped_data), adapter_mmapped_data_size);
  }
#endif
}

int MoonshineStreamingModel::load_config(const char *config_path) {
  std::string config_json = read_file_to_string(config_path);
  if (config_json.empty()) {
    LOGF("Failed to read config file: %s\n", config_path);
    return 1;
  }
  return load_config_from_string(config_json);
}

int MoonshineStreamingModel::load_config_from_string(const std::string &json) {
  if (!parse_config_json(json, &config)) {
    LOG("Failed to parse streaming config JSON\n");
    return 1;
  }
  return 0;
}

int MoonshineStreamingModel::load(const char *model_dir,
                                  const char *tokenizer_path,
                                  int32_t /* model_type */) {
  if (model_dir == nullptr) {
    LOG("Model directory is null\n");
    return 1;
  }

  // Build paths
  std::string frontend_path = append_path_component(model_dir, "frontend.ort");
  std::string encoder_path = append_path_component(model_dir, "encoder.ort");
  std::string adapter_path = append_path_component(model_dir, "adapter.ort");
  std::string config_path =
      append_path_component(model_dir, "streaming_config.json");

  // Load config
  RETURN_ON_ERROR(load_config(config_path.c_str()));

  // Load sessions using ort_session_from_path (same as non-streaming)
  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, frontend_path.c_str(),
      &frontend_session, &frontend_mmapped_data, &frontend_mmapped_data_size));
  RETURN_ON_NULL(frontend_session);

  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, encoder_path.c_str(),
      &encoder_session, &encoder_mmapped_data, &encoder_mmapped_data_size));
  RETURN_ON_NULL(encoder_session);

  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, adapter_path.c_str(),
      &adapter_session, &adapter_mmapped_data, &adapter_mmapped_data_size));
  RETURN_ON_NULL(adapter_session);

  // Load cross_kv and decoder_kv sessions (required for decoding)
  std::string cross_kv_path = append_path_component(model_dir, "cross_kv.ort");
  std::string decoder_kv_path =
      append_path_component(model_dir, "decoder_kv.ort");

  // Check if .ort versions exist (prefer them over .onnx)
  std::string cross_kv_ort = append_path_component(model_dir, "cross_kv.ort");
  std::string decoder_kv_ort =
      append_path_component(model_dir, "decoder_kv.ort");
  if (std::ifstream(cross_kv_ort).good()) cross_kv_path = cross_kv_ort;
  if (std::ifstream(decoder_kv_ort).good()) decoder_kv_path = decoder_kv_ort;

  // Load cross_kv (required)
  {
    const char *cross_kv_mmap = nullptr;
    size_t cross_kv_mmap_size = 0;
    RETURN_ON_ERROR(ort_session_from_path(
        ort_api, ort_env, ort_session_options, cross_kv_path.c_str(),
        &cross_kv_session, &cross_kv_mmap, &cross_kv_mmap_size));
    RETURN_ON_NULL(cross_kv_session);
  }

  // Load decoder_kv (required)
  {
    const char *decoder_kv_mmap = nullptr;
    size_t decoder_kv_mmap_size = 0;
    RETURN_ON_ERROR(ort_session_from_path(
        ort_api, ort_env, ort_session_options, decoder_kv_path.c_str(),
        &decoder_kv_session, &decoder_kv_mmap, &decoder_kv_mmap_size));
    RETURN_ON_NULL(decoder_kv_session);
  }

  // Load tokenizer
  tokenizer = new BinTokenizer(tokenizer_path);
  RETURN_ON_NULL(tokenizer);

  return 0;
}

int MoonshineStreamingModel::load_from_memory(
    const uint8_t *frontend_model_data, size_t frontend_model_data_size,
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *adapter_model_data, size_t adapter_model_data_size,
    const uint8_t *cross_kv_model_data, size_t cross_kv_model_data_size,
    const uint8_t *decoder_kv_model_data, size_t decoder_kv_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    const MoonshineStreamingConfig &in_config, int32_t /* model_type */) {
  config = in_config;

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, frontend_model_data,
      frontend_model_data_size, &frontend_session));
  RETURN_ON_NULL(frontend_session);

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, encoder_model_data,
      encoder_model_data_size, &encoder_session));
  RETURN_ON_NULL(encoder_session);

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, adapter_model_data,
      adapter_model_data_size, &adapter_session));
  RETURN_ON_NULL(adapter_session);

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, cross_kv_model_data,
      cross_kv_model_data_size, &cross_kv_session));
  RETURN_ON_NULL(cross_kv_session);

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, decoder_kv_model_data,
      decoder_kv_model_data_size, &decoder_kv_session));
  RETURN_ON_NULL(decoder_kv_session);

  tokenizer = new BinTokenizer(tokenizer_data, tokenizer_data_size);
  RETURN_ON_NULL(tokenizer);

  return 0;
}

#if defined(ANDROID)
int MoonshineStreamingModel::load_from_assets(const char *model_dir,
                                              const char *tokenizer_path,
                                              int32_t /* model_type */,
                                              AAssetManager *assetManager) {
  if (model_dir == nullptr) {
    LOG("Model directory is null\n");
    return 1;
  }

  // Build paths
  std::string frontend_path = append_path_component(model_dir, "frontend.ort");
  std::string encoder_path = append_path_component(model_dir, "encoder.ort");
  std::string adapter_path = append_path_component(model_dir, "adapter.ort");
  std::string cross_kv_path = append_path_component(model_dir, "cross_kv.ort");
  std::string decoder_kv_path =
      append_path_component(model_dir, "decoder_kv.onnx");
  std::string config_path =
      append_path_component(model_dir, "streaming_config.json");

  // Load config from asset
  AAsset *config_asset =
      AAssetManager_open(assetManager, config_path.c_str(), AASSET_MODE_BUFFER);
  if (config_asset == nullptr) {
    LOGF("Failed to open config asset: %s\n", config_path.c_str());
    return 1;
  }
  size_t config_size = AAsset_getLength(config_asset);
  std::string config_json(config_size, '\0');
  AAsset_read(config_asset, &config_json[0], config_size);
  AAsset_close(config_asset);
  RETURN_ON_ERROR(load_config_from_string(config_json));

  // Load sessions
  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager,
      frontend_path.c_str(), &frontend_session, &frontend_mmapped_data,
      &frontend_mmapped_data_size));
  RETURN_ON_NULL(frontend_session);

  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager, encoder_path.c_str(),
      &encoder_session, &encoder_mmapped_data, &encoder_mmapped_data_size));
  RETURN_ON_NULL(encoder_session);

  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager, adapter_path.c_str(),
      &adapter_session, &adapter_mmapped_data, &adapter_mmapped_data_size));
  RETURN_ON_NULL(adapter_session);

  // Load cross_kv and decoder_kv sessions (required for decoding)
  const char *cross_kv_mmap = nullptr;
  size_t cross_kv_mmap_size = 0;
  RETURN_ON_ERROR(ort_session_from_asset(ort_api, ort_env, ort_session_options,
                                         assetManager, cross_kv_path.c_str(),
                                         &cross_kv_session, &cross_kv_mmap,
                                         &cross_kv_mmap_size));
  RETURN_ON_NULL(cross_kv_session);

  const char *decoder_kv_mmap = nullptr;
  size_t decoder_kv_mmap_size = 0;
  RETURN_ON_ERROR(ort_session_from_asset(ort_api, ort_env, ort_session_options,
                                         assetManager, decoder_kv_path.c_str(),
                                         &decoder_kv_session, &decoder_kv_mmap,
                                         &decoder_kv_mmap_size));
  RETURN_ON_NULL(decoder_kv_session);

  tokenizer = new BinTokenizer(tokenizer_path, assetManager);
  RETURN_ON_NULL(tokenizer);

  return 0;
}
#endif

MoonshineStreamingState *MoonshineStreamingModel::create_state() {
  MoonshineStreamingState *state = new MoonshineStreamingState();
  state->reset(config);
  return state;
}

std::string MoonshineStreamingModel::tokens_to_text(
    const std::vector<int64_t> &tokens) {
  return tokenizer->tokens_to_text(tokens);
}

/* ============================================================================
 * Streaming Inference Implementation
 * ============================================================================
 */

int MoonshineStreamingModel::process_audio_chunk(MoonshineStreamingState *state,
                                                 const float *audio_chunk,
                                                 size_t chunk_len,
                                                 int *features_out) {
  if (state == nullptr) {
    LOG("State is null\n");
    return 1;
  }
  if (audio_chunk == nullptr && chunk_len > 0) {
    LOG("Audio chunk is null but chunk_len > 0\n");
    return 1;
  }

  if (chunk_len == 0) {
    if (features_out) *features_out = 0;
    return 0;
  }

  std::lock_guard<std::mutex> lock(processing_mutex);

  // Prepare input tensors
  std::vector<float> audio_vec(audio_chunk, audio_chunk + chunk_len);
  std::vector<int64_t> audio_shape = {1, static_cast<int64_t>(chunk_len)};
  std::vector<int64_t> sample_buffer_shape = {1, 79};
  std::vector<int64_t> sample_len_shape = {1};
  std::vector<int64_t> conv1_shape = {1, config.d_model_frontend, 4};
  std::vector<int64_t> conv2_shape = {1, config.c1, 4};
  std::vector<int64_t> frame_count_shape = {1};

  // Create ORT values
  OrtValue *audio_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, audio_vec.data(), audio_vec.size() * sizeof(float),
          audio_shape.data(), audio_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &audio_tensor));

  OrtValue *sample_buffer_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, state->sample_buffer.data(),
                   state->sample_buffer.size() * sizeof(float),
                   sample_buffer_shape.data(), sample_buffer_shape.size(),
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &sample_buffer_tensor));

  OrtValue *sample_len_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, &state->sample_len, sizeof(int64_t),
                   sample_len_shape.data(), sample_len_shape.size(),
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sample_len_tensor));

  OrtValue *conv1_buffer_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, state->conv1_buffer.data(),
                   state->conv1_buffer.size() * sizeof(float),
                   conv1_shape.data(), conv1_shape.size(),
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &conv1_buffer_tensor));

  OrtValue *conv2_buffer_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, state->conv2_buffer.data(),
                   state->conv2_buffer.size() * sizeof(float),
                   conv2_shape.data(), conv2_shape.size(),
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &conv2_buffer_tensor));

  OrtValue *frame_count_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, &state->frame_count, sizeof(int64_t),
                   frame_count_shape.data(), frame_count_shape.size(),
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &frame_count_tensor));

  // Run frontend
  const char *input_names[] = {"audio_chunk",  "sample_buffer", "sample_len",
                               "conv1_buffer", "conv2_buffer",  "frame_count"};
  const char *output_names[] = {"features",         "sample_buffer_out",
                                "sample_len_out",   "conv1_buffer_out",
                                "conv2_buffer_out", "frame_count_out"};

  OrtValue *inputs[] = {audio_tensor,        sample_buffer_tensor,
                        sample_len_tensor,   conv1_buffer_tensor,
                        conv2_buffer_tensor, frame_count_tensor};
  OrtValue *outputs[6] = {nullptr};

  OrtStatus *status = ORT_RUN(ort_api, frontend_session, input_names, inputs, 6,
                              output_names, 6, outputs);

  // Release input tensors
  for (int i = 0; i < 6; i++) {
    ort_api->ReleaseValue(inputs[i]);
  }

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    return 1;
  }

  // Extract features
  OrtTensorTypeAndShapeInfo *features_info = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorTypeAndShape(outputs[0], &features_info));
  size_t num_dims = 0;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetDimensionsCount(features_info, &num_dims));
  std::vector<int64_t> feat_shape(num_dims);
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensions(
                                   features_info, feat_shape.data(), num_dims));
  ort_api->ReleaseTensorTypeAndShapeInfo(features_info);

  int num_features = static_cast<int>(feat_shape[1]);
  int feat_dim = static_cast<int>(feat_shape[2]);

  // Accumulate features
  if (num_features > 0) {
    float *feat_data = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                     outputs[0], (void **)&feat_data));
    size_t feat_size = num_features * feat_dim;
    state->accumulated_features.insert(state->accumulated_features.end(),
                                       feat_data, feat_data + feat_size);
    state->accumulated_feature_count += num_features;
  }

  // Update state from outputs
  float *sample_buffer_out = nullptr;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   outputs[1], (void **)&sample_buffer_out));
  memcpy(state->sample_buffer.data(), sample_buffer_out, 79 * sizeof(float));

  int64_t *sample_len_out = nullptr;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   outputs[2], (void **)&sample_len_out));
  state->sample_len = *sample_len_out;

  float *conv1_out = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[3], (void **)&conv1_out));
  memcpy(state->conv1_buffer.data(), conv1_out,
         config.d_model_frontend * 4 * sizeof(float));

  float *conv2_out = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[4], (void **)&conv2_out));
  memcpy(state->conv2_buffer.data(), conv2_out, config.c1 * 4 * sizeof(float));

  int64_t *frame_count_out = nullptr;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   outputs[5], (void **)&frame_count_out));
  state->frame_count = *frame_count_out;

  // Release outputs
  for (int i = 0; i < 6; i++) {
    ort_api->ReleaseValue(outputs[i]);
  }

  if (features_out) *features_out = num_features;
  return 0;
}

int MoonshineStreamingModel::encode(MoonshineStreamingState *state,
                                    bool is_final, int *new_frames_out) {
  if (state == nullptr) {
    LOG("State is null\n");
    return 1;
  }

  int total_features = state->accumulated_feature_count;
  if (log_ort_run) {
    LOGF(
        "streaming encode: total_features=%d, encoder_frames_emitted=%d, "
        "memory_len=%d, is_final=%d",
        total_features, state->encoder_frames_emitted, state->memory_len,
        is_final ? 1 : 0);
  }
  if (total_features == 0) {
    if (new_frames_out) *new_frames_out = 0;
    return 0;
  }

  const int stable_count =
      is_final ? total_features
               : std::max(0, total_features - config.total_lookahead);
  const int new_frames = stable_count - state->encoder_frames_emitted;
  if (log_ort_run) {
    LOGF("streaming encode: stable_count=%d, new_frames=%d", stable_count,
         new_frames);
  }
  if (new_frames <= 0) {
    if (new_frames_out) *new_frames_out = 0;
    return 0;
  }

  // Encoder uses a sliding window with fixed per-layer left context.
  const int left_context_frames = 16 * config.depth;
  int window_start = state->encoder_frames_emitted - left_context_frames;
  if (window_start < 0) {
    window_start = 0;
  }
  const int window_size = total_features - window_start;
  if (log_ort_run) {
    LOGF("streaming encode: window_start=%d, window_size=%d", window_start,
         window_size);
  }

  std::lock_guard<std::mutex> lock(processing_mutex);

  // Run encoder on windowed accumulated features.
  std::vector<int64_t> feat_shape = {1, window_size, config.encoder_dim};

  OrtValue *features_tensor = nullptr;
  const float *features_ptr =
      state->accumulated_features.data() + window_start * config.encoder_dim;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, const_cast<float *>(features_ptr),
          static_cast<size_t>(window_size) * config.encoder_dim * sizeof(float),
          feat_shape.data(), feat_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &features_tensor));

  const char *enc_input_names[] = {"features"};
  const char *enc_output_names[] = {"encoded"};
  OrtValue *enc_outputs[1] = {nullptr};

  OrtStatus *status =
      ORT_RUN(ort_api, encoder_session, enc_input_names, &features_tensor, 1,
              enc_output_names, 1, enc_outputs);

  ort_api->ReleaseValue(features_tensor);

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    return 1;
  }

  // Get encoded shape
  OrtTensorTypeAndShapeInfo *enc_info = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorTypeAndShape(enc_outputs[0], &enc_info));
  size_t num_dims = 0;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetDimensionsCount(enc_info, &num_dims));
  std::vector<int64_t> enc_shape(num_dims);
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetDimensions(enc_info, enc_shape.data(), num_dims));
  ort_api->ReleaseTensorTypeAndShapeInfo(enc_info);

  int total_encoded = static_cast<int>(enc_shape[1]);

  // Run adapter on new frames
  float *encoded_data = nullptr;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   enc_outputs[0], (void **)&encoded_data));
  int start_idx = state->encoder_frames_emitted - window_start;
  if (start_idx < 0 || start_idx + new_frames > total_encoded) {
    ort_api->ReleaseValue(enc_outputs[0]);
    LOGF("Encoder window misaligned: start_idx=%d, new_frames=%d, total=%d",
         start_idx, new_frames, total_encoded);
    return 1;
  }

  std::vector<float> new_encoded(new_frames * config.encoder_dim);
  for (int i = 0; i < new_frames; ++i) {
    for (int j = 0; j < config.encoder_dim; ++j) {
      new_encoded[i * config.encoder_dim + j] =
          encoded_data[(start_idx + i) * config.encoder_dim + j];
    }
  }

  ort_api->ReleaseValue(enc_outputs[0]);

  std::vector<int64_t> enc_slice_shape = {1, new_frames, config.encoder_dim};
  std::vector<int64_t> pos_shape = {1};

  OrtValue *encoded_slice_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, new_encoded.data(),
                   new_encoded.size() * sizeof(float), enc_slice_shape.data(),
                   enc_slice_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                   &encoded_slice_tensor));

  OrtValue *pos_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, &state->adapter_pos_offset, sizeof(int64_t),
                   pos_shape.data(), pos_shape.size(),
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &pos_tensor));

  const char *adapter_input_names[] = {"encoded", "pos_offset"};
  const char *adapter_output_names[] = {"memory"};
  OrtValue *adapter_inputs[] = {encoded_slice_tensor, pos_tensor};
  OrtValue *adapter_outputs[1] = {nullptr};

  status = ORT_RUN(ort_api, adapter_session, adapter_input_names,
                   adapter_inputs, 2, adapter_output_names, 1, adapter_outputs);

  ort_api->ReleaseValue(encoded_slice_tensor);
  ort_api->ReleaseValue(pos_tensor);

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    return 1;
  }

  // Append to memory
  float *mem_data = nullptr;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   adapter_outputs[0], (void **)&mem_data));
  size_t mem_size = new_frames * config.decoder_dim;
  state->memory.insert(state->memory.end(), mem_data, mem_data + mem_size);
  state->memory_len += new_frames;
  if (log_ort_run) {
    LOGF("streaming encode: memory_len_after=%d", state->memory_len);
  }

  // Invalidate cross K/V cache since memory changed
  state->cross_kv_valid = false;

  ort_api->ReleaseValue(adapter_outputs[0]);

  // Update tracking
  state->encoder_frames_emitted = stable_count;
  state->adapter_pos_offset += new_frames;

  if (new_frames_out) *new_frames_out = new_frames;
  return 0;
}

/* ============================================================================
 * Compute cross-attention K/V from memory (for optimized decoding)
 * ============================================================================
 */

int MoonshineStreamingModel::compute_cross_kv(MoonshineStreamingState *state) {
  if (state == nullptr || cross_kv_session == nullptr) {
    return 1;
  }
  if (state->memory_len == 0) {
    LOG("Memory is empty, cannot compute cross K/V\n");
    return 1;
  }

  // Input: memory [1, mem_len, decoder_dim]
  std::vector<int64_t> memory_shape = {1, state->memory_len,
                                       config.decoder_dim};

  OrtValue *memory_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, state->memory.data(),
                   state->memory.size() * sizeof(float), memory_shape.data(),
                   memory_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                   &memory_tensor));

  // Run cross_kv session
  const char *input_names[] = {"memory"};
  const char *output_names[] = {"k_cross", "v_cross"};

  OrtValue *inputs[] = {memory_tensor};
  OrtValue *outputs[2] = {nullptr, nullptr};

  OrtStatus *status = ORT_RUN(ort_api, cross_kv_session, input_names, inputs, 1,
                              output_names, 2, outputs);

  ort_api->ReleaseValue(memory_tensor);

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    return 1;
  }

  // Extract k_cross shape: [depth, 1, nheads, cross_len, head_dim]
  OrtTensorTypeAndShapeInfo *k_info = nullptr;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetTensorTypeAndShape(outputs[0], &k_info));
  size_t num_dims = 0;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensionsCount(k_info, &num_dims));
  std::vector<int64_t> k_shape(num_dims);
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetDimensions(k_info, k_shape.data(), num_dims));
  ort_api->ReleaseTensorTypeAndShapeInfo(k_info);

  if (num_dims != 5) {
    LOG("Expected 5D cross KV tensor\n");
    for (int i = 0; i < 2; i++) ort_api->ReleaseValue(outputs[i]);
    return 1;
  }

  int cross_len = static_cast<int>(k_shape[3]);
  size_t kv_size = static_cast<size_t>(config.depth) * config.nheads *
                   cross_len * config.head_dim;

  // Copy to state
  state->k_cross.resize(kv_size);
  state->v_cross.resize(kv_size);
  state->cross_len = cross_len;

  float *k_data = nullptr;
  float *v_data = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[0], (void **)&k_data));
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[1], (void **)&v_data));

  memcpy(state->k_cross.data(), k_data, kv_size * sizeof(float));
  memcpy(state->v_cross.data(), v_data, kv_size * sizeof(float));
  state->cross_kv_valid = true;

  // Release outputs
  for (int i = 0; i < 2; i++) {
    ort_api->ReleaseValue(outputs[i]);
  }

  return 0;
}

/* ============================================================================
 * Run decoder with precomputed cross K/V (more efficient path)
 * ============================================================================
 */

int MoonshineStreamingModel::run_decoder_with_cross_kv(
    MoonshineStreamingState *state, const std::vector<int64_t> &tokens,
    std::vector<float> &logits_out) {
  if (state == nullptr || decoder_kv_session == nullptr) {
    return 1;
  }
  if (!state->cross_kv_valid || state->cross_len == 0) {
    LOG("Cross K/V not valid, call compute_cross_kv first\n");
    return 1;
  }

  int token_len = static_cast<int>(tokens.size());
  if (token_len == 0) {
    return 1;
  }

  // Token input [1, token_len]
  std::vector<int64_t> token_shape = {1, static_cast<int64_t>(token_len)};
  std::vector<int64_t> token_data(tokens.begin(), tokens.end());

  OrtValue *token_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, token_data.data(),
                   token_data.size() * sizeof(int64_t), token_shape.data(),
                   token_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                   &token_tensor));

  // Self-attention KV cache [depth, 1, nheads, cache_len, head_dim]
  int cache_len = state->cache_seq_len;
  std::vector<int64_t> kv_self_shape = {config.depth, 1, config.nheads,
                                        cache_len, config.head_dim};
  size_t kv_self_size = static_cast<size_t>(config.depth) * config.nheads *
                        cache_len * config.head_dim;

  if (state->k_self.size() != kv_self_size) {
    state->k_self.resize(kv_self_size, 0.0f);
    state->v_self.resize(kv_self_size, 0.0f);
  }

  OrtValue *k_self_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, state->k_self.data(),
                   state->k_self.size() * sizeof(float), kv_self_shape.data(),
                   kv_self_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                   &k_self_tensor));

  OrtValue *v_self_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                   ort_memory_info, state->v_self.data(),
                   state->v_self.size() * sizeof(float), kv_self_shape.data(),
                   kv_self_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                   &v_self_tensor));

  // Cross-attention KV cache [depth, 1, nheads, cross_len, head_dim]
  std::vector<int64_t> kv_cross_shape = {config.depth, 1, config.nheads,
                                         state->cross_len, config.head_dim};
  size_t kv_cross_size = static_cast<size_t>(config.depth) * config.nheads *
                         state->cross_len * config.head_dim;

  OrtValue *k_cross_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, state->k_cross.data(), kv_cross_size * sizeof(float),
          kv_cross_shape.data(), kv_cross_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &k_cross_tensor));

  OrtValue *v_cross_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, state->v_cross.data(), kv_cross_size * sizeof(float),
          kv_cross_shape.data(), kv_cross_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &v_cross_tensor));

  // Run decoder_kv session
  // Note: decoder_kv expects cross K/V as "out_k_cross" and "out_v_cross"
  // inputs
  const char *input_names[] = {"token", "k_self", "v_self", "out_k_cross",
                               "out_v_cross"};
  const char *output_names[] = {"logits", "out_k_self", "out_v_self"};

  OrtValue *inputs[] = {token_tensor, k_self_tensor, v_self_tensor,
                        k_cross_tensor, v_cross_tensor};
  OrtValue *outputs[3] = {nullptr, nullptr, nullptr};

  OrtStatus *status = ORT_RUN(ort_api, decoder_kv_session, input_names, inputs,
                              5, output_names, 3, outputs);

  // Release inputs
  for (int i = 0; i < 5; i++) {
    ort_api->ReleaseValue(inputs[i]);
  }

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    return 1;
  }

  // Copy logits [1, token_len, vocab_size]
  float *logits_data = nullptr;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   outputs[0], (void **)&logits_data));

  size_t total_logits = token_len * config.vocab_size;
  logits_out.resize(total_logits);
  memcpy(logits_out.data(), logits_data, total_logits * sizeof(float));

  // Update self-attention KV cache
  OrtTensorTypeAndShapeInfo *k_info = nullptr;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetTensorTypeAndShape(outputs[1], &k_info));
  size_t num_dims = 0;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensionsCount(k_info, &num_dims));
  std::vector<int64_t> k_shape(num_dims);
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetDimensions(k_info, k_shape.data(), num_dims));
  ort_api->ReleaseTensorTypeAndShapeInfo(k_info);

  int new_cache_len = static_cast<int>(k_shape[3]);
  size_t new_cache_size = static_cast<size_t>(config.depth) * config.nheads *
                          new_cache_len * config.head_dim;

  state->k_self.resize(new_cache_size);
  state->v_self.resize(new_cache_size);

  float *k_out_data = nullptr;
  float *v_out_data = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[1], (void **)&k_out_data));
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[2], (void **)&v_out_data));

  memcpy(state->k_self.data(), k_out_data, new_cache_size * sizeof(float));
  memcpy(state->v_self.data(), v_out_data, new_cache_size * sizeof(float));
  state->cache_seq_len = new_cache_len;

  // Release outputs
  for (int i = 0; i < 3; i++) {
    ort_api->ReleaseValue(outputs[i]);
  }

  return 0;
}

/* ============================================================================
 * Single-token decode step
 * ============================================================================
 */

int MoonshineStreamingModel::decode_step(MoonshineStreamingState *state,
                                         int token, float *logits_out) {
  if (state == nullptr) {
    LOG("State is null\n");
    return 1;
  }
  if (logits_out == nullptr) {
    LOG("Logits output is null\n");
    return 1;
  }
  if (state->memory_len == 0) {
    LOG("Memory is empty\n");
    return 1;
  }

  std::lock_guard<std::mutex> lock(processing_mutex);

  std::vector<int64_t> tokens = {static_cast<int64_t>(token)};
  std::vector<float> logits;

  int err;

  // Compute cross K/V if not valid
  if (!state->cross_kv_valid) {
    err = compute_cross_kv(state);
    if (err != 0) {
      LOG("Failed to compute cross K/V\n");
      return err;
    }
  }

  err = run_decoder_with_cross_kv(state, tokens, logits);

  if (err != 0) {
    return err;
  }

  // Copy logits to output
  memcpy(logits_out, logits.data(), config.vocab_size * sizeof(float));
  return 0;
}

/* ============================================================================
 * Multi-token decode step
 * ============================================================================
 */

int MoonshineStreamingModel::decode_tokens(MoonshineStreamingState *state,
                                           const int *tokens, int tokens_len,
                                           float *logits_out) {
  if (state == nullptr) {
    LOG("State is null\n");
    return 1;
  }
  if (tokens == nullptr || tokens_len <= 0) {
    LOG("Tokens is null or empty\n");
    return 1;
  }
  if (logits_out == nullptr) {
    LOG("Logits output is null\n");
    return 1;
  }
  if (state->memory_len == 0) {
    LOG("Memory is empty\n");
    return 1;
  }

  std::lock_guard<std::mutex> lock(processing_mutex);

  std::vector<int64_t> token_vec(tokens_len);
  for (int i = 0; i < tokens_len; ++i) {
    token_vec[i] = static_cast<int64_t>(tokens[i]);
  }

  std::vector<float> logits;
  int err;

  // Compute cross K/V if not valid
  if (!state->cross_kv_valid) {
    err = compute_cross_kv(state);
    if (err != 0) {
      LOG("Failed to compute cross K/V\n");
      return err;
    }
  }

  err = run_decoder_with_cross_kv(state, token_vec, logits);

  if (err != 0) {
    return err;
  }

  // Copy all logits to output
  memcpy(logits_out, logits.data(),
         tokens_len * config.vocab_size * sizeof(float));
  return 0;
}

/* ============================================================================
 * Full decode with speculative decoding support
 * ============================================================================
 */

int MoonshineStreamingModel::decode_full(MoonshineStreamingState *state,
                                         const int *speculative_tokens,
                                         int speculative_len, int **tokens_out,
                                         int *tokens_len_out) {
  if (state == nullptr) {
    LOG("State is null\n");
    return 1;
  }
  if (tokens_out == nullptr || tokens_len_out == nullptr) {
    LOG("Output pointers are null\n");
    return 1;
  }
  if (state->memory_len == 0) {
    LOG("Memory is empty\n");
    *tokens_out = nullptr;
    *tokens_len_out = 0;
    return 0;
  }

  std::lock_guard<std::mutex> lock(processing_mutex);

  std::vector<int> result_tokens;

  // Compute max tokens based on memory length (similar to new decoder)
  float duration_sec = state->memory_len * 0.020f;
  int max_tokens = std::min(static_cast<int>(std::ceil(duration_sec * 6.5)),
                            config.max_seq_len);

  // Helper lambda for argmax
  auto argmax = [this](const float *logits) -> int {
    int best = 0;
    float best_score = logits[0];
    for (int i = 1; i < config.vocab_size; ++i) {
      if (logits[i] > best_score) {
        best_score = logits[i];
        best = i;
      }
    }
    return best;
  };

  // Helper to run decoder (requires cross_kv path)
  auto run_decoder = [this, state](const std::vector<int64_t> &tokens,
                                   std::vector<float> &logits) -> int {
    if (!state->cross_kv_valid) {
      int err = compute_cross_kv(state);
      if (err != 0) {
        LOG("Failed to compute cross K/V\n");
        return err;
      }
    }
    return run_decoder_with_cross_kv(state, tokens, logits);
  };

  // Compute cross K/V upfront
  if (!state->cross_kv_valid) {
    int err = compute_cross_kv(state);
    if (err != 0) {
      LOG("Failed to compute cross K/V\n");
      return err;
    }
  }

  // Helper lambda for auto-regressive decoding
  auto continue_ar_decoding = [&](int start_token) {
    int current_token = start_token;
    std::vector<float> logits;

    while (current_token != config.eos_id &&
           result_tokens.size() < static_cast<size_t>(max_tokens)) {
      result_tokens.push_back(current_token);

      std::vector<int64_t> next_input = {static_cast<int64_t>(current_token)};
      int err = run_decoder(next_input, logits);
      if (err != 0) break;

      current_token = argmax(logits.data());
    }
  };

  if (speculative_tokens != nullptr && speculative_len > 0) {
    // Speculative decoding: verify previous tokens
    std::vector<int64_t> tokens_with_bos;
    tokens_with_bos.push_back(config.bos_id);
    for (int i = 0; i < speculative_len; ++i) {
      tokens_with_bos.push_back(static_cast<int64_t>(speculative_tokens[i]));
    }

    std::vector<float> logits;
    int err = run_decoder(tokens_with_bos, logits);
    if (err != 0) {
      return err;
    }

    // Get predictions from logits
    std::vector<int> predictions;
    for (int t = 0; t < static_cast<int>(tokens_with_bos.size()); ++t) {
      predictions.push_back(argmax(logits.data() + t * config.vocab_size));
    }

    // Find divergence point
    int diverge_point = 0;
    for (int i = 0; i < speculative_len; ++i) {
      if (predictions[i] == speculative_tokens[i]) {
        diverge_point = i + 1;
      } else {
        break;
      }
    }

    // Accept verified tokens
    for (int i = 0; i < diverge_point; ++i) {
      result_tokens.push_back(speculative_tokens[i]);
    }

    if (diverge_point == speculative_len) {
      // All speculative tokens verified, continue from final prediction
      int final_pred = predictions[speculative_len];
      continue_ar_decoding(final_pred);
    } else {
      // Diverged: reset cache and re-run with only accepted tokens
      state->cache_seq_len = 0;
      state->k_self.clear();
      state->v_self.clear();

      std::vector<int64_t> accepted_tokens;
      accepted_tokens.push_back(config.bos_id);
      for (int i = 0; i < diverge_point; ++i) {
        accepted_tokens.push_back(static_cast<int64_t>(speculative_tokens[i]));
      }

      std::vector<float> logits2;
      err = run_decoder(accepted_tokens, logits2);
      if (err != 0) {
        return err;
      }

      int new_pred = argmax(logits2.data() + diverge_point * config.vocab_size);
      continue_ar_decoding(new_pred);
    }
  } else {
    // Regular decoding: start from BOS
    std::vector<int64_t> tokens = {static_cast<int64_t>(config.bos_id)};
    std::vector<float> logits;

    int err = run_decoder(tokens, logits);
    if (err != 0) {
      return err;
    }

    int first_pred = argmax(logits.data());
    continue_ar_decoding(first_pred);
  }

  // Allocate and copy output
  *tokens_len_out = static_cast<int>(result_tokens.size());

  if (result_tokens.empty()) {
    *tokens_out = nullptr;
  } else {
    *tokens_out =
        static_cast<int *>(malloc(result_tokens.size() * sizeof(int)));
    if (*tokens_out == nullptr) {
      return 1;
    }
    memcpy(*tokens_out, result_tokens.data(),
           result_tokens.size() * sizeof(int));
  }

  return 0;
}

void MoonshineStreamingModel::decoder_reset(MoonshineStreamingState *state) {
  if (state == nullptr) return;
  state->k_self.clear();
  state->v_self.clear();
  state->cache_seq_len = 0;
  // Note: We keep cross K/V valid since memory hasn't changed
  // It will be invalidated automatically when memory changes via encode()
}
