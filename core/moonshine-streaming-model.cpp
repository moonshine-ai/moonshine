#include "moonshine-streaming-model.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
 * ============================================================================ */

static std::string read_file_to_string(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) return "";
    std::stringstream buffer;
    buffer << f.rdbuf();
    return buffer.str();
}

// TODO Use constants instead of loading config JSON
static bool parse_config_json(const std::string& json, MoonshineStreamingConfig* config) {
    // Simple key-value extraction for our known config format
    auto get_int = [&json](const char* key) -> int {
        std::string search = std::string("\"") + key + "\":";
        size_t pos = json.find(search);
        if (pos == std::string::npos) return 0;
        pos += search.length();
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
        int val = 0;
        bool negative = false;
        if (json[pos] == '-') { negative = true; pos++; }
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

    // Validate essential fields
    return config->depth > 0 && config->decoder_dim > 0 && config->vocab_size > 0;
}

/* ============================================================================
 * MoonshineStreamingState Implementation
 * ============================================================================ */

void MoonshineStreamingState::reset(const MoonshineStreamingConfig& cfg) {
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
}

/* ============================================================================
 * MoonshineStreamingModel Implementation
 * ============================================================================ */

MoonshineStreamingModel::MoonshineStreamingModel()
    : frontend_session(nullptr), encoder_session(nullptr),
      adapter_session(nullptr), decoder_session(nullptr), tokenizer(nullptr) {
    ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    LOG_ORT_ERROR(ort_api, ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                              "MoonshineStreamingModel", &ort_env));
    LOG_ORT_ERROR(ort_api, ort_api->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                                        OrtMemTypeDefault,
                                                        &ort_memory_info));
    ort_allocator = new MoonshineOrtAllocator(ort_memory_info);

    LOG_ORT_ERROR(ort_api, ort_api->CreateSessionOptions(&ort_session_options));
    LOG_ORT_ERROR(ort_api, ort_api->SetSessionGraphOptimizationLevel(
                               ort_session_options, ORT_ENABLE_ALL));
    LOG_ORT_ERROR(ort_api, ort_api->SetIntraOpNumThreads(ort_session_options, 1));
    
    memset(&config, 0, sizeof(config));
}

MoonshineStreamingModel::~MoonshineStreamingModel() {
    ort_api->ReleaseEnv(ort_env);
    ort_api->ReleaseMemoryInfo(ort_memory_info);
    ort_api->ReleaseSessionOptions(ort_session_options);
    if (frontend_session) ort_api->ReleaseSession(frontend_session);
    if (encoder_session) ort_api->ReleaseSession(encoder_session);
    if (adapter_session) ort_api->ReleaseSession(adapter_session);
    if (decoder_session) ort_api->ReleaseSession(decoder_session);
    delete ort_allocator;
    delete tokenizer;
    
    if (frontend_mmapped_data) {
        munmap(const_cast<char *>(frontend_mmapped_data), frontend_mmapped_data_size);
    }
    if (encoder_mmapped_data) {
        munmap(const_cast<char *>(encoder_mmapped_data), encoder_mmapped_data_size);
    }
    if (adapter_mmapped_data) {
        munmap(const_cast<char *>(adapter_mmapped_data), adapter_mmapped_data_size);
    }
    if (decoder_mmapped_data) {
        munmap(const_cast<char *>(decoder_mmapped_data), decoder_mmapped_data_size);
    }
}

int MoonshineStreamingModel::load_config(const char *config_path) {
    std::string config_json = read_file_to_string(config_path);
    if (config_json.empty()) {
        LOGF("Failed to read config file: %s\n", config_path);
        return 1;
    }
    return load_config_from_string(config_json);
}

int MoonshineStreamingModel::load_config_from_string(const std::string& json) {
    if (!parse_config_json(json, &config)) {
        LOG("Failed to parse streaming config JSON\n");
        return 1;
    }
    return 0;
}

int MoonshineStreamingModel::load(const char *model_dir, const char *tokenizer_path,
                                   int32_t /* model_type */) {
    if (model_dir == nullptr) {
        LOG("Model directory is null\n");
        return 1;
    }

    // Build paths
    std::string frontend_path = append_path_component(model_dir, "frontend.onnx");
    std::string encoder_path = append_path_component(model_dir, "encoder.onnx");
    std::string adapter_path = append_path_component(model_dir, "adapter.onnx");
    std::string decoder_path = append_path_component(model_dir, "decoder.onnx");
    std::string config_path = append_path_component(model_dir, "streaming_config.json");

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

    RETURN_ON_ERROR(ort_session_from_path(
        ort_api, ort_env, ort_session_options, decoder_path.c_str(),
        &decoder_session, &decoder_mmapped_data, &decoder_mmapped_data_size));
    RETURN_ON_NULL(decoder_session);

    // Load tokenizer
    tokenizer = new BinTokenizer(tokenizer_path);
    RETURN_ON_NULL(tokenizer);

    return 0;
}

int MoonshineStreamingModel::load_from_memory(
    const uint8_t *frontend_model_data, size_t frontend_model_data_size,
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *adapter_model_data, size_t adapter_model_data_size,
    const uint8_t *decoder_model_data, size_t decoder_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    const MoonshineStreamingConfig& in_config, int32_t /* model_type */) {
    
    config = in_config;

    RETURN_ON_ERROR(ort_session_from_memory(
        ort_api, ort_env, ort_session_options, frontend_model_data, frontend_model_data_size,
        &frontend_session));
    RETURN_ON_NULL(frontend_session);

    RETURN_ON_ERROR(ort_session_from_memory(
        ort_api, ort_env, ort_session_options, encoder_model_data, encoder_model_data_size,
        &encoder_session));
    RETURN_ON_NULL(encoder_session);

    RETURN_ON_ERROR(ort_session_from_memory(
        ort_api, ort_env, ort_session_options, adapter_model_data, adapter_model_data_size,
        &adapter_session));
    RETURN_ON_NULL(adapter_session);

    RETURN_ON_ERROR(ort_session_from_memory(
        ort_api, ort_env, ort_session_options, decoder_model_data, decoder_model_data_size,
        &decoder_session));
    RETURN_ON_NULL(decoder_session);

    tokenizer = new BinTokenizer(tokenizer_data, tokenizer_data_size);
    RETURN_ON_NULL(tokenizer);

    return 0;
}

#if defined(ANDROID)
int MoonshineStreamingModel::load_from_assets(const char *model_dir,
                                               const char *tokenizer_path,
                                               int32_t model_type,
                                               AAssetManager *assetManager) {
    if (model_dir == nullptr) {
        LOG("Model directory is null\n");
        return 1;
    }

    // Build paths
    std::string frontend_path = append_path_component(model_dir, "frontend.onnx");
    std::string encoder_path = append_path_component(model_dir, "encoder.onnx");
    std::string adapter_path = append_path_component(model_dir, "adapter.onnx");
    std::string decoder_path = append_path_component(model_dir, "decoder.onnx");
    std::string config_path = append_path_component(model_dir, "streaming_config.json");

    // Load config from asset
    AAsset* config_asset = AAssetManager_open(assetManager, config_path.c_str(), AASSET_MODE_BUFFER);
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
        ort_api, ort_env, ort_session_options, assetManager, frontend_path.c_str(),
        &frontend_session, &frontend_mmapped_data, &frontend_mmapped_data_size));
    RETURN_ON_NULL(frontend_session);

    RETURN_ON_ERROR(ort_session_from_asset(
        ort_api, ort_env, ort_session_options, assetManager, encoder_path.c_str(),
        &encoder_session, &encoder_mmapped_data, &encoder_mmapped_data_size));
    RETURN_ON_NULL(encoder_session);

    RETURN_ON_ERROR(ort_session_from_asset(
        ort_api, ort_env, ort_session_options, assetManager, adapter_path.c_str(),
        &adapter_session, &adapter_mmapped_data, &adapter_mmapped_data_size));
    RETURN_ON_NULL(adapter_session);

    RETURN_ON_ERROR(ort_session_from_asset(
        ort_api, ort_env, ort_session_options, assetManager, decoder_path.c_str(),
        &decoder_session, &decoder_mmapped_data, &decoder_mmapped_data_size));
    RETURN_ON_NULL(decoder_session);

    tokenizer = new BinTokenizer(tokenizer_path, assetManager);
    RETURN_ON_NULL(tokenizer);

    return 0;
}
#endif

MoonshineStreamingState* MoonshineStreamingModel::create_state() {
    MoonshineStreamingState* state = new MoonshineStreamingState();
    state->reset(config);
    return state;
}

std::string MoonshineStreamingModel::tokens_to_text(const std::vector<int64_t>& tokens) {
    return tokenizer->tokens_to_text(tokens);
}

/* ============================================================================
 * Streaming Inference Implementation
 * ============================================================================ */

int MoonshineStreamingModel::process_audio_chunk(MoonshineStreamingState *state,
                                                  const float *audio_chunk,
                                                  int chunk_len,
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
    std::vector<int64_t> audio_shape = {1, chunk_len};
    std::vector<int64_t> sample_buffer_shape = {1, 79};
    std::vector<int64_t> sample_len_shape = {1};
    std::vector<int64_t> conv1_shape = {1, config.d_model_frontend, 4};
    std::vector<int64_t> conv2_shape = {1, config.c1, 4};
    std::vector<int64_t> frame_count_shape = {1};

    // Create ORT values
    OrtValue* audio_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, audio_vec.data(), audio_vec.size() * sizeof(float),
        audio_shape.data(), audio_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &audio_tensor));

    OrtValue* sample_buffer_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, state->sample_buffer.data(), state->sample_buffer.size() * sizeof(float),
        sample_buffer_shape.data(), sample_buffer_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &sample_buffer_tensor));

    OrtValue* sample_len_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, &state->sample_len, sizeof(int64_t),
        sample_len_shape.data(), sample_len_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &sample_len_tensor));

    OrtValue* conv1_buffer_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, state->conv1_buffer.data(), state->conv1_buffer.size() * sizeof(float),
        conv1_shape.data(), conv1_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &conv1_buffer_tensor));

    OrtValue* conv2_buffer_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, state->conv2_buffer.data(), state->conv2_buffer.size() * sizeof(float),
        conv2_shape.data(), conv2_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &conv2_buffer_tensor));

    OrtValue* frame_count_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, &state->frame_count, sizeof(int64_t),
        frame_count_shape.data(), frame_count_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &frame_count_tensor));

    // Run frontend
    const char* input_names[] = {
        "audio_chunk", "sample_buffer", "sample_len",
        "conv1_buffer", "conv2_buffer", "frame_count"
    };
    const char* output_names[] = {
        "features", "sample_buffer_out", "sample_len_out",
        "conv1_buffer_out", "conv2_buffer_out", "frame_count_out"
    };

    OrtValue* inputs[] = {
        audio_tensor, sample_buffer_tensor, sample_len_tensor,
        conv1_buffer_tensor, conv2_buffer_tensor, frame_count_tensor
    };
    OrtValue* outputs[6] = {nullptr};

    OrtStatus* status = ort_api->Run(
        frontend_session, nullptr,
        input_names, inputs, 6,
        output_names, 6, outputs
    );

    // Release input tensors
    for (int i = 0; i < 6; i++) {
        ort_api->ReleaseValue(inputs[i]);
    }

    if (status != nullptr) {
        LOG_ORT_ERROR(ort_api, status);
        return 1;
    }

    // Extract features
    OrtTensorTypeAndShapeInfo* features_info = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorTypeAndShape(outputs[0], &features_info));
    size_t num_dims = 0;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensionsCount(features_info, &num_dims));
    std::vector<int64_t> feat_shape(num_dims);
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensions(features_info, feat_shape.data(), num_dims));
    ort_api->ReleaseTensorTypeAndShapeInfo(features_info);

    int num_features = static_cast<int>(feat_shape[1]);
    int feat_dim = static_cast<int>(feat_shape[2]);

    // Accumulate features
    if (num_features > 0) {
        float* feat_data = nullptr;
        RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[0], (void**)&feat_data));
        size_t feat_size = num_features * feat_dim;
        state->accumulated_features.insert(
            state->accumulated_features.end(),
            feat_data, feat_data + feat_size
        );
        state->accumulated_feature_count += num_features;
    }

    // Update state from outputs
    float* sample_buffer_out = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[1], (void**)&sample_buffer_out));
    memcpy(state->sample_buffer.data(), sample_buffer_out, 79 * sizeof(float));

    int64_t* sample_len_out = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[2], (void**)&sample_len_out));
    state->sample_len = *sample_len_out;

    float* conv1_out = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[3], (void**)&conv1_out));
    memcpy(state->conv1_buffer.data(), conv1_out, config.d_model_frontend * 4 * sizeof(float));

    float* conv2_out = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[4], (void**)&conv2_out));
    memcpy(state->conv2_buffer.data(), conv2_out, config.c1 * 4 * sizeof(float));

    int64_t* frame_count_out = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[5], (void**)&frame_count_out));
    state->frame_count = *frame_count_out;

    // Release outputs
    for (int i = 0; i < 6; i++) {
        ort_api->ReleaseValue(outputs[i]);
    }

    if (features_out) *features_out = num_features;
    return 0;
}

int MoonshineStreamingModel::encode(MoonshineStreamingState *state,
                                     bool is_final,
                                     int *new_frames_out) {
    if (state == nullptr) {
        LOG("State is null\n");
        return 1;
    }

    int total_features = state->accumulated_feature_count;
    if (total_features == 0) {
        if (new_frames_out) *new_frames_out = 0;
        return 0;
    }

    std::lock_guard<std::mutex> lock(processing_mutex);

    // Run encoder on all accumulated features
    std::vector<int64_t> feat_shape = {1, total_features, config.encoder_dim};

    OrtValue* features_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, state->accumulated_features.data(),
        state->accumulated_features.size() * sizeof(float),
        feat_shape.data(), feat_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &features_tensor));

    const char* enc_input_names[] = {"features"};
    const char* enc_output_names[] = {"encoded"};
    OrtValue* enc_outputs[1] = {nullptr};

    OrtStatus* status = ort_api->Run(
        encoder_session, nullptr,
        enc_input_names, &features_tensor, 1,
        enc_output_names, 1, enc_outputs
    );

    ort_api->ReleaseValue(features_tensor);

    if (status != nullptr) {
        LOG_ORT_ERROR(ort_api, status);
        return 1;
    }

    // Get encoded shape
    OrtTensorTypeAndShapeInfo* enc_info = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorTypeAndShape(enc_outputs[0], &enc_info));
    size_t num_dims = 0;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensionsCount(enc_info, &num_dims));
    std::vector<int64_t> enc_shape(num_dims);
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensions(enc_info, enc_shape.data(), num_dims));
    ort_api->ReleaseTensorTypeAndShapeInfo(enc_info);

    int total_encoded = static_cast<int>(enc_shape[1]);

    // Compute stable frame count
    int stable_count;
    if (is_final) {
        stable_count = total_encoded;
    } else {
        stable_count = std::max(0, total_encoded - config.total_lookahead);
    }

    // Slice new stable frames
    int new_frames = stable_count - state->encoder_frames_emitted;
    if (new_frames <= 0) {
        ort_api->ReleaseValue(enc_outputs[0]);
        if (new_frames_out) *new_frames_out = 0;
        return 0;
    }

    // Run adapter on new frames
    float* encoded_data = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(enc_outputs[0], (void**)&encoded_data));
    int start_idx = state->encoder_frames_emitted;

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

    OrtValue* encoded_slice_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, new_encoded.data(), new_encoded.size() * sizeof(float),
        enc_slice_shape.data(), enc_slice_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &encoded_slice_tensor));

    OrtValue* pos_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, &state->adapter_pos_offset, sizeof(int64_t),
        pos_shape.data(), pos_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &pos_tensor));

    const char* adapter_input_names[] = {"encoded", "pos_offset"};
    const char* adapter_output_names[] = {"memory"};
    OrtValue* adapter_inputs[] = {encoded_slice_tensor, pos_tensor};
    OrtValue* adapter_outputs[1] = {nullptr};

    status = ort_api->Run(
        adapter_session, nullptr,
        adapter_input_names, adapter_inputs, 2,
        adapter_output_names, 1, adapter_outputs
    );

    ort_api->ReleaseValue(encoded_slice_tensor);
    ort_api->ReleaseValue(pos_tensor);

    if (status != nullptr) {
        LOG_ORT_ERROR(ort_api, status);
        return 1;
    }

    // Append to memory
    float* mem_data = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(adapter_outputs[0], (void**)&mem_data));
    size_t mem_size = new_frames * config.decoder_dim;
    state->memory.insert(state->memory.end(), mem_data, mem_data + mem_size);
    state->memory_len += new_frames;

    ort_api->ReleaseValue(adapter_outputs[0]);

    // Update tracking
    state->encoder_frames_emitted = stable_count;
    state->adapter_pos_offset += new_frames;

    if (new_frames_out) *new_frames_out = new_frames;
    return 0;
}

int MoonshineStreamingModel::decode_step(MoonshineStreamingState *state,
                                          int token,
                                          float *logits_out) {
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

    // Token input
    std::vector<int64_t> token_data = {static_cast<int64_t>(token)};
    std::vector<int64_t> token_shape = {1, 1};

    OrtValue* token_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, token_data.data(), sizeof(int64_t),
        token_shape.data(), token_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
        &token_tensor));

    // Memory input
    std::vector<int64_t> memory_shape = {1, state->memory_len, config.decoder_dim};

    OrtValue* memory_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, state->memory.data(), state->memory.size() * sizeof(float),
        memory_shape.data(), memory_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &memory_tensor));

    // KV cache
    int cache_len = state->cache_seq_len;
    std::vector<int64_t> cache_shape = {config.depth, 1, config.nheads, cache_len, config.head_dim};

    // Initialize empty cache if needed
    if (state->k_self.empty()) {
        state->k_self.resize(config.depth * config.nheads * cache_len * config.head_dim, 0.0f);
        state->v_self.resize(config.depth * config.nheads * cache_len * config.head_dim, 0.0f);
    }

    OrtValue* k_self_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, state->k_self.data(), state->k_self.size() * sizeof(float),
        cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &k_self_tensor));

    OrtValue* v_self_tensor = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateTensorWithDataAsOrtValue(
        ort_memory_info, state->v_self.data(), state->v_self.size() * sizeof(float),
        cache_shape.data(), cache_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &v_self_tensor));

    // Run decoder
    const char* input_names[] = {"token", "memory", "k_self", "v_self"};
    const char* output_names[] = {"logits", "out_k_self", "out_v_self", "out_k_cross", "out_v_cross"};

    OrtValue* inputs[] = {token_tensor, memory_tensor, k_self_tensor, v_self_tensor};
    OrtValue* outputs[5] = {nullptr};

    OrtStatus* status = ort_api->Run(
        decoder_session, nullptr,
        input_names, inputs, 4,
        output_names, 5, outputs
    );

    // Release inputs
    for (int i = 0; i < 4; i++) {
        ort_api->ReleaseValue(inputs[i]);
    }

    if (status != nullptr) {
        LOG_ORT_ERROR(ort_api, status);
        return 1;
    }

    // Copy logits
    float* logits_data = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[0], (void**)&logits_data));
    memcpy(logits_out, logits_data, config.vocab_size * sizeof(float));

    // Update KV cache
    OrtTensorTypeAndShapeInfo* k_info = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorTypeAndShape(outputs[1], &k_info));
    size_t num_dims = 0;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensionsCount(k_info, &num_dims));
    std::vector<int64_t> k_shape(num_dims);
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensions(k_info, k_shape.data(), num_dims));
    ort_api->ReleaseTensorTypeAndShapeInfo(k_info);

    int new_cache_len = static_cast<int>(k_shape[3]);
    size_t new_cache_size = config.depth * config.nheads * new_cache_len * config.head_dim;

    state->k_self.resize(new_cache_size);
    state->v_self.resize(new_cache_size);

    float* k_out_data = nullptr;
    float* v_out_data = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[1], (void**)&k_out_data));
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(outputs[2], (void**)&v_out_data));

    memcpy(state->k_self.data(), k_out_data, new_cache_size * sizeof(float));
    memcpy(state->v_self.data(), v_out_data, new_cache_size * sizeof(float));
    state->cache_seq_len = new_cache_len;

    // Release outputs
    for (int i = 0; i < 5; i++) {
        ort_api->ReleaseValue(outputs[i]);
    }

    return 0;
}

void MoonshineStreamingModel::decoder_reset(MoonshineStreamingState *state) {
    if (state == nullptr) return;
    state->k_self.clear();
    state->v_self.clear();
    state->cache_seq_len = 0;
}

/* ============================================================================
 * Batch Transcription (convenience method)
 * ============================================================================ */

int MoonshineStreamingModel::transcribe(const float *input_audio_data,
                                         size_t input_audio_data_size,
                                         char **out_text) {
    *out_text = nullptr;
    if (input_audio_data == nullptr || input_audio_data_size == 0) {
        LOG("Audio data is nullptr or empty\n");
        return 1;
    }

    // Create state
    MoonshineStreamingState* state = create_state();
    if (state == nullptr) {
        LOG("Failed to create streaming state\n");
        return 1;
    }

    // Process audio in chunks (80ms = 1280 samples at 16kHz)
    const int chunk_size = 1280;
    for (size_t offset = 0; offset < input_audio_data_size; offset += chunk_size) {
        int len = std::min(static_cast<size_t>(chunk_size), input_audio_data_size - offset);
        int err = process_audio_chunk(state, input_audio_data + offset, len, nullptr);
        if (err != 0) {
            delete state;
            return err;
        }
    }

    // Final encode (emit all frames)
    int err = encode(state, true, nullptr);
    if (err != 0) {
        delete state;
        return err;
    }

    if (state->memory_len == 0) {
        delete state;
        last_result = "";
        *out_text = (char*)(last_result.c_str());
        return 0;
    }

    // Decode
    const int max_tokens = 256;
    std::vector<int64_t> tokens;
    tokens.push_back(config.bos_id);

    std::vector<float> logits(config.vocab_size);
    int current_token = config.bos_id;

    const int ngram_size = 3;
    const int max_repeats = 2;

    for (int step = 0; step < max_tokens; ++step) {
        err = decode_step(state, current_token, logits.data());
        if (err != 0) {
            delete state;
            return err;
        }

        // Argmax
        int next_token = 0;
        float max_logit = logits[0];
        for (int i = 1; i < config.vocab_size; ++i) {
            if (logits[i] > max_logit) {
                max_logit = logits[i];
                next_token = i;
            }
        }

        tokens.push_back(next_token);
        current_token = next_token;

        // EOS check
        if (next_token == config.eos_id) break;

        // Repetition detection
        if (tokens.size() >= static_cast<size_t>(ngram_size * (max_repeats + 1))) {
            int repeat_count = 0;
            size_t end_pos = tokens.size();

            for (int r = 1; r <= max_repeats; ++r) {
                bool match = true;
                for (int j = 0; j < ngram_size; ++j) {
                    size_t cur_idx = end_pos - ngram_size + j;
                    size_t prev_idx = end_pos - ngram_size * (r + 1) + j;
                    if (tokens[cur_idx] != tokens[prev_idx]) {
                        match = false;
                        break;
                    }
                }
                if (match) ++repeat_count;
                else break;
            }

            if (repeat_count >= max_repeats) {
                for (int j = 0; j < ngram_size * max_repeats; ++j) {
                    tokens.pop_back();
                }
                tokens.push_back(config.eos_id);
                break;
            }
        }
    }

    delete state;

    // Convert tokens to text
    last_result = tokenizer->tokens_to_text(tokens);
    *out_text = (char*)(last_result.c_str());

    return 0;
}
