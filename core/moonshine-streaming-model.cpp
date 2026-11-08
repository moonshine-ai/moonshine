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
#include <filesystem>
#include <fstream>
#include <sstream>
#include <utility>

#include "bin-tokenizer.h"
#include "moonshine-ort-allocator.h"
#include "speculative-realign.h"
#include "string-utils.h"

#define DEBUG_ALLOC_ENABLED 1
#include "debug-utils.h"
#include "ort-utils.h"

#ifndef _WIN32
namespace {
void unmap_model(const char **data, size_t *size) {
  if (*data != nullptr) {
    munmap(const_cast<char *>(*data), *size);
    *data = nullptr;
    *size = 0;
  }
}
}  // namespace
#endif

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

namespace {

// The word-boundary marker these vocabularies use, U+2581. Passed explicitly
// only because the encoding follows it in the argument list.
const char *kSpaceString = "▁";

// These vocabularies are learned by merges, so text_to_tokens has to replay
// those merges to arrive at the subwords the decoder was trained to emit.
// Taking the longest entry at each position instead lands on a different
// spelling for most words needing more than one subword, which went unnoticed
// while the tokenizer was only ever used to turn tokens back into text.
// Contextual biasing was the first caller to encode text, and it was waiting on
// subwords the decoder never emits.
const BinTokenizerEncoding kTokenizerEncoding = BinTokenizerEncoding::kBpe;

// How many frames each encoder layer attends back over, for a stateless
// encoder graph that gives no other clue. Every published model uses 16. A
// cached encoder graph bakes its own width into its cache shape, and
// detect_encoder_kind reads it from there instead of trusting this.
constexpr int kDefaultEncoderCacheFrames = 16;

// When the encoder has not emitted memory yet (lookahead still filling, or a
// new segment just reset), decoding is a no-op. Point argmax at EOS so a
// caller that ignores the empty-memory case still stops instead of reading
// uninitialized logits.
void write_empty_decode_logits(const MoonshineStreamingConfig &config,
                               float *logits_out) {
  if (logits_out == nullptr || config.vocab_size <= 0) {
    return;
  }
  std::fill(logits_out, logits_out + config.vocab_size, 0.0f);
  if (config.eos_id >= 0 && config.eos_id < config.vocab_size) {
    logits_out[config.eos_id] = 1.0f;
  }
}

}  // namespace

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

  // Encoder layer cache
  encoder_cache.clear();
  encoder_cache_len = 0;
}

/* ============================================================================
 * MoonshineStreamingModel Implementation
 * ============================================================================
 */

MoonshineStreamingModel::MoonshineStreamingModel(
    bool log_ort_run, const std::vector<std::string> &ort_provider_names,
    const std::string &coreml_cache_dir)
    : frontend_session(nullptr),
      encoder_session(nullptr),
      adapter_session(nullptr),
      cross_kv_session(nullptr),
      decoder_kv_session(nullptr),
      tokenizer(nullptr),
      log_ort_run(log_ort_run) {
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  LOG_ORT_ERROR(ort_api, ort_create_env(ort_api, ORT_LOGGING_LEVEL_WARNING,
                                        "MoonshineStreamingModel", &ort_env));
  LOG_ORT_ERROR(ort_api,
                ort_api->CreateCpuMemoryInfo(
                    OrtDeviceAllocator, OrtMemTypeDefault, &ort_memory_info));
  ort_allocator = new MoonshineOrtAllocator(ort_memory_info);

  LOG_ORT_ERROR(ort_api, ort_api->CreateSessionOptions(&ort_session_options));
  ort_maybe_force_single_thread(ort_api, ort_session_options);
  ort_configure_ort_file_session(ort_api, ort_session_options);
  ort_configure_execution_providers(ort_api, ort_session_options,
                                    ort_provider_names, coreml_cache_dir);

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
  unmap_model(&frontend_mmapped_data, &frontend_mmapped_data_size);
  unmap_model(&encoder_mmapped_data, &encoder_mmapped_data_size);
  unmap_model(&adapter_mmapped_data, &adapter_mmapped_data_size);
  unmap_model(&cross_kv_mmapped_data, &cross_kv_mmapped_data_size);
  unmap_model(&decoder_kv_mmapped_data, &decoder_kv_mmapped_data_size);
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

int MoonshineStreamingModel::detect_encoder_kind() {
  encoder_has_cache = false;
  encoder_cache_frames = kDefaultEncoderCacheFrames;
  if (encoder_session == nullptr) return 1;
  size_t input_count = 0;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->SessionGetInputCount(encoder_session, &input_count));
  if (input_count == 1) {
    return 0;
  }
  if (input_count != 3) {
    LOGF("encoder graph has %zu inputs; expected 1 (stateless) or 3 (cached)",
         input_count);
    return 1;
  }
  // The cached graph is identified by its input names so a stray third input
  // on some future graph is an error rather than a misread cache.
  static const char *const kExpected[] = {"features", "cache", "cache_valid"};
  for (size_t i = 0; i < 3; ++i) {
    char *name = nullptr;
    RETURN_ON_ORT_ERROR(
        ort_api, ort_api->SessionGetInputName(encoder_session, i,
                                              &ort_allocator->base, &name));
    const bool matches = name != nullptr && strcmp(name, kExpected[i]) == 0;
    if (name != nullptr) {
      ort_allocator->base.Free(&ort_allocator->base, name);
    }
    if (!matches) {
      LOGF("encoder input %zu is not '%s'", i, kExpected[i]);
      return 1;
    }
  }
  // The cache input must be [depth, 1, cache_frames, encoder_dim]; the
  // cache_frames axis is how far each layer looks back and is taken from the
  // graph.
  OrtTypeInfo *type_info = nullptr;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetInputTypeInfo(encoder_session,
                                                                1, &type_info));
  const OrtTensorTypeAndShapeInfo *shape_info = nullptr;
  OrtStatus *cast_status =
      ort_api->CastTypeInfoToTensorInfo(type_info, &shape_info);
  std::vector<int64_t> dims;
  if (cast_status == nullptr && shape_info != nullptr) {
    size_t num_dims = 0;
    if (ort_api->GetDimensionsCount(shape_info, &num_dims) == nullptr) {
      dims.resize(num_dims);
      if (ort_api->GetDimensions(shape_info, dims.data(), num_dims) !=
          nullptr) {
        dims.clear();
      }
    }
  }
  ort_api->ReleaseTypeInfo(type_info);
  if (dims.size() != 4 || dims[0] != config.depth || dims[1] != 1 ||
      dims[2] <= 0 || dims[2] > 1024 || dims[3] != config.encoder_dim) {
    LOGF(
        "encoder cache input has %zu dims (%lld, %lld, %lld, %lld); expected "
        "(%d, 1, <frames>, %d)",
        dims.size(), dims.size() > 0 ? (long long)dims[0] : -1LL,
        dims.size() > 1 ? (long long)dims[1] : -1LL,
        dims.size() > 2 ? (long long)dims[2] : -1LL,
        dims.size() > 3 ? (long long)dims[3] : -1LL, config.depth,
        config.encoder_dim);
    return 1;
  }
  encoder_cache_frames = static_cast<int>(dims[2]);
  encoder_has_cache = true;
  return 0;
}

int MoonshineStreamingModel::collect_frontend_split_weights(
    OrtSession *weights_session) {
  frontend_split_weights.clear();
  size_t count = 0;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->SessionGetOutputCount(weights_session, &count));
  if (count == 0) {
    LOG("frontend weights model produced no outputs\n");
    return 1;
  }

  std::vector<char *> names_alloc(count, nullptr);
  std::vector<const char *> names(count, nullptr);
  for (size_t i = 0; i < count; ++i) {
    RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetOutputName(
                                     weights_session, i, &ort_allocator->base,
                                     &names_alloc[i]));
    names[i] = names_alloc[i];
  }

  std::vector<OrtValue *> outputs(count, nullptr);
  OrtStatus *status = ORT_RUN(ort_api, weights_session, nullptr, nullptr, 0,
                              names.data(), count, outputs.data());
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    for (char *n : names_alloc) {
      if (n != nullptr) {
        ort_allocator->base.Free(&ort_allocator->base, n);
      }
    }
    return 1;
  }

  frontend_split_weights.reserve(count);
  int err = 0;
  for (size_t i = 0; i < count; ++i) {
    OrtTensorTypeAndShapeInfo *info = nullptr;
    OrtStatus *info_status = ort_api->GetTensorTypeAndShape(outputs[i], &info);
    if (info_status != nullptr) {
      LOG_ORT_ERROR(ort_api, info_status);
      err = 1;
      break;
    }
    ONNXTensorElementDataType dtype = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorElementType(info, &dtype));
    size_t n_dims = 0;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensionsCount(info, &n_dims));
    FrontendSplitWeight weight;
    weight.name = names[i];
    weight.shape.resize(n_dims);
    RETURN_ON_ORT_ERROR(
        ort_api, ort_api->GetDimensions(info, weight.shape.data(), n_dims));
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      LOGF("frontend split weight %s is not float32\n", weight.name.c_str());
      err = 1;
      break;
    }
    float *src = nullptr;
    OrtStatus *data_status =
        ort_api->GetTensorMutableData(outputs[i], (void **)&src);
    if (data_status != nullptr) {
      LOG_ORT_ERROR(ort_api, data_status);
      err = 1;
      break;
    }
    size_t elements = 1;
    for (int64_t dim : weight.shape) {
      elements *= static_cast<size_t>(dim);
    }
    weight.data.assign(src, src + elements);
    frontend_split_weights.push_back(std::move(weight));
  }

  for (OrtValue *value : outputs) {
    if (value != nullptr) {
      ort_api->ReleaseValue(value);
    }
  }
  for (char *n : names_alloc) {
    if (n != nullptr) {
      ort_allocator->base.Free(&ort_allocator->base, n);
    }
  }
  return err;
}

int MoonshineStreamingModel::load_frontend_split_weights_from_memory(
    const uint8_t *data, size_t size) {
  OrtSession *weights_session = nullptr;
  RETURN_ON_ERROR(ort_session_from_memory(ort_api, ort_env, ort_session_options,
                                          data, size, &weights_session));
  const int err = collect_frontend_split_weights(weights_session);
  ort_api->ReleaseSession(weights_session);
  return err;
}

int MoonshineStreamingModel::load_frontend_split_weights_from_path(
    const char *path) {
  OrtSession *weights_session = nullptr;
  const char *mapped = nullptr;
  size_t mapped_size = 0;
  RETURN_ON_ERROR(ort_session_from_path(ort_api, ort_env, ort_session_options,
                                        path, &weights_session, &mapped,
                                        &mapped_size));
  const int err = collect_frontend_split_weights(weights_session);
  ort_api->ReleaseSession(weights_session);
#ifndef _WIN32
  unmap_model(&mapped, &mapped_size);
#endif
  return err;
}

int MoonshineStreamingModel::load(const char *model_dir,
                                  const char *tokenizer_path,
                                  int32_t /* model_type */) {
  if (model_dir == nullptr) {
    LOG("Model directory is null\n");
    return 1;
  }

  // Build paths
  std::string frontend_model_path =
      append_path_component(model_dir, "frontend.model.ort");
  std::string frontend_weights_path =
      append_path_component(model_dir, "frontend.weights.ort");
  std::string frontend_path = append_path_component(model_dir, "frontend.ort");
  std::string encoder_path = append_path_component(model_dir, "encoder.ort");
  std::string adapter_path = append_path_component(model_dir, "adapter.ort");
  std::string config_path =
      append_path_component(model_dir, "streaming_config.json");

  // Load config
  RETURN_ON_ERROR(load_config(config_path.c_str()));

  // Prefer the split pair so int8 weights survive ORT conversion. Fall back
  // to a single frontend.ort for pins published before the split.
  const bool split_frontend =
      std::filesystem::is_regular_file(frontend_model_path) &&
      std::filesystem::is_regular_file(frontend_weights_path);
  if (split_frontend) {
    RETURN_ON_ERROR(ort_session_from_path(
        ort_api, ort_env, ort_session_options, frontend_model_path.c_str(),
        &frontend_session, &frontend_mmapped_data,
        &frontend_mmapped_data_size));
    RETURN_ON_NULL(frontend_session);
    RETURN_ON_ERROR(
        load_frontend_split_weights_from_path(frontend_weights_path.c_str()));
  } else {
    RETURN_ON_ERROR(ort_session_from_path(
        ort_api, ort_env, ort_session_options, frontend_path.c_str(),
        &frontend_session, &frontend_mmapped_data,
        &frontend_mmapped_data_size));
    RETURN_ON_NULL(frontend_session);
  }

  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, encoder_path.c_str(),
      &encoder_session, &encoder_mmapped_data, &encoder_mmapped_data_size));
  RETURN_ON_NULL(encoder_session);
  RETURN_ON_ERROR(detect_encoder_kind());

  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, adapter_path.c_str(),
      &adapter_session, &adapter_mmapped_data, &adapter_mmapped_data_size));
  RETURN_ON_NULL(adapter_session);

  // Load cross_kv and decoder_kv sessions (required for decoding)
  std::string cross_kv_path = append_path_component(model_dir, "cross_kv.ort");
  std::string decoder_kv_path =
      append_path_component(model_dir, "decoder_kv.ort");

  // Load cross_kv (required). Keep the mapping on the model so the destructor
  // can munmap it; a local pointer here is how issue #216 leaked ~83 MB per
  // Transcriber.
  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, cross_kv_path.c_str(),
      &cross_kv_session, &cross_kv_mmapped_data, &cross_kv_mmapped_data_size));
  RETURN_ON_NULL(cross_kv_session);

  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, decoder_kv_path.c_str(),
      &decoder_kv_session, &decoder_kv_mmapped_data,
      &decoder_kv_mmapped_data_size));
  RETURN_ON_NULL(decoder_kv_session);

  // Load tokenizer
  tokenizer =
      new BinTokenizer(tokenizer_path, kSpaceString, kTokenizerEncoding);
  RETURN_ON_NULL(tokenizer);

  return 0;
}

int MoonshineStreamingModel::replace_decoder_kv_from_path(const char *path) {
  if (decoder_kv_session != nullptr) {
    ort_api->ReleaseSession(decoder_kv_session);
    decoder_kv_session = nullptr;
  }
#ifndef _WIN32
  unmap_model(&decoder_kv_mmapped_data, &decoder_kv_mmapped_data_size);
#endif
  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, path, &decoder_kv_session,
      &decoder_kv_mmapped_data, &decoder_kv_mmapped_data_size));
  RETURN_ON_NULL(decoder_kv_session);
  return 0;
}

int MoonshineStreamingModel::load_from_memory(
    const uint8_t *frontend_model_data, size_t frontend_model_data_size,
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *adapter_model_data, size_t adapter_model_data_size,
    const uint8_t *cross_kv_model_data, size_t cross_kv_model_data_size,
    const uint8_t *decoder_kv_model_data, size_t decoder_kv_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    const MoonshineStreamingConfig &in_config, int32_t /* model_type */,
    const uint8_t *frontend_weights_data, size_t frontend_weights_data_size) {
  config = in_config;

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, frontend_model_data,
      frontend_model_data_size, &frontend_session));
  RETURN_ON_NULL(frontend_session);
  if (frontend_weights_data != nullptr && frontend_weights_data_size > 0) {
    RETURN_ON_ERROR(load_frontend_split_weights_from_memory(
        frontend_weights_data, frontend_weights_data_size));
  }

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, encoder_model_data,
      encoder_model_data_size, &encoder_session));
  RETURN_ON_NULL(encoder_session);
  RETURN_ON_ERROR(detect_encoder_kind());

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

  tokenizer = new BinTokenizer(tokenizer_data, tokenizer_data_size,
                               kSpaceString, kTokenizerEncoding);
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
  std::string frontend_model_path =
      append_path_component(model_dir, "frontend.model.ort");
  std::string frontend_weights_path =
      append_path_component(model_dir, "frontend.weights.ort");
  std::string frontend_path = append_path_component(model_dir, "frontend.ort");
  std::string encoder_path = append_path_component(model_dir, "encoder.ort");
  std::string adapter_path = append_path_component(model_dir, "adapter.ort");
  std::string cross_kv_path = append_path_component(model_dir, "cross_kv.ort");
  std::string decoder_kv_path =
      append_path_component(model_dir, "decoder_kv.ort");
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

  AAsset *split_model_asset = AAssetManager_open(
      assetManager, frontend_model_path.c_str(), AASSET_MODE_BUFFER);
  AAsset *split_weights_asset = AAssetManager_open(
      assetManager, frontend_weights_path.c_str(), AASSET_MODE_BUFFER);
  const bool split_frontend =
      split_model_asset != nullptr && split_weights_asset != nullptr;
  if (split_model_asset != nullptr) {
    AAsset_close(split_model_asset);
  }
  if (split_weights_asset != nullptr) {
    AAsset_close(split_weights_asset);
  }

  if (split_frontend) {
    RETURN_ON_ERROR(ort_session_from_asset(
        ort_api, ort_env, ort_session_options, assetManager,
        frontend_model_path.c_str(), &frontend_session, &frontend_mmapped_data,
        &frontend_mmapped_data_size));
    RETURN_ON_NULL(frontend_session);
    OrtSession *weights_session = nullptr;
    const char *weights_mapped = nullptr;
    size_t weights_mapped_size = 0;
    RETURN_ON_ERROR(ort_session_from_asset(
        ort_api, ort_env, ort_session_options, assetManager,
        frontend_weights_path.c_str(), &weights_session, &weights_mapped,
        &weights_mapped_size));
    const int weights_err = collect_frontend_split_weights(weights_session);
    ort_api->ReleaseSession(weights_session);
    unmap_model(&weights_mapped, &weights_mapped_size);
    RETURN_ON_ERROR(weights_err);
  } else {
    RETURN_ON_ERROR(ort_session_from_asset(
        ort_api, ort_env, ort_session_options, assetManager,
        frontend_path.c_str(), &frontend_session, &frontend_mmapped_data,
        &frontend_mmapped_data_size));
    RETURN_ON_NULL(frontend_session);
  }

  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager, encoder_path.c_str(),
      &encoder_session, &encoder_mmapped_data, &encoder_mmapped_data_size));
  RETURN_ON_NULL(encoder_session);
  RETURN_ON_ERROR(detect_encoder_kind());

  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager, adapter_path.c_str(),
      &adapter_session, &adapter_mmapped_data, &adapter_mmapped_data_size));
  RETURN_ON_NULL(adapter_session);

  // Load cross_kv and decoder_kv sessions (required for decoding)
  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager,
      cross_kv_path.c_str(), &cross_kv_session, &cross_kv_mmapped_data,
      &cross_kv_mmapped_data_size));
  RETURN_ON_NULL(cross_kv_session);

  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager,
      decoder_kv_path.c_str(), &decoder_kv_session, &decoder_kv_mmapped_data,
      &decoder_kv_mmapped_data_size));
  RETURN_ON_NULL(decoder_kv_session);

  tokenizer = new BinTokenizer(tokenizer_path, assetManager, kSpaceString,
                               kTokenizerEncoding);
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

std::vector<int32_t> MoonshineStreamingModel::text_to_tokens(
    const std::string &text) {
  if (tokenizer == nullptr) {
    return {};
  }
  return tokenizer->text_to_tokens<int32_t>(text);
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

  // Run frontend. Split-pair weights are extra graph inputs, so the name
  // list is not a fixed 6-entry array.
  std::vector<const char *> input_names = {"audio_chunk",  "sample_buffer",
                                           "sample_len",   "conv1_buffer",
                                           "conv2_buffer", "frame_count"};
  const char *output_names[] = {"features",         "sample_buffer_out",
                                "sample_len_out",   "conv1_buffer_out",
                                "conv2_buffer_out", "frame_count_out"};

  std::vector<OrtValue *> inputs = {audio_tensor,        sample_buffer_tensor,
                                    sample_len_tensor,   conv1_buffer_tensor,
                                    conv2_buffer_tensor, frame_count_tensor};
  for (FrontendSplitWeight &weight : frontend_split_weights) {
    OrtValue *weight_tensor = nullptr;
    RETURN_ON_ORT_ERROR(
        ort_api, ort_api->CreateTensorWithDataAsOrtValue(
                     ort_memory_info, weight.data.data(),
                     weight.data.size() * sizeof(float), weight.shape.data(),
                     weight.shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                     &weight_tensor));
    inputs.push_back(weight_tensor);
    input_names.push_back(weight.name.c_str());
  }
  OrtValue *outputs[6] = {nullptr};

  OrtStatus *status =
      ORT_RUN(ort_api, frontend_session, input_names.data(), inputs.data(),
              inputs.size(), output_names, 6, outputs);

  // Release input tensors
  for (OrtValue *value : inputs) {
    ort_api->ReleaseValue(value);
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

namespace {

// Reads a [1, frames, dim] float tensor into a vector. Returns the frame
// count, or -1 on error.
int read_frames_tensor(const OrtApi *ort_api, OrtValue *value, int dim,
                       std::vector<float> *out) {
  OrtTensorTypeAndShapeInfo *info = nullptr;
  if (ort_api->GetTensorTypeAndShape(value, &info) != nullptr) return -1;
  size_t num_dims = 0;
  if (ort_api->GetDimensionsCount(info, &num_dims) != nullptr) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    return -1;
  }
  std::vector<int64_t> shape(num_dims);
  if (ort_api->GetDimensions(info, shape.data(), num_dims) != nullptr) {
    ort_api->ReleaseTensorTypeAndShapeInfo(info);
    return -1;
  }
  ort_api->ReleaseTensorTypeAndShapeInfo(info);
  if (num_dims != 3 || shape[2] != dim) return -1;
  const int frames = static_cast<int>(shape[1]);
  float *data = nullptr;
  if (ort_api->GetTensorMutableData(value, (void **)&data) != nullptr) {
    return -1;
  }
  out->assign(data, data + static_cast<size_t>(frames) * dim);
  return frames;
}

}  // namespace

int MoonshineStreamingModel::run_encoder_stateless(
    MoonshineStreamingState *state, int window_start, int total_features,
    std::vector<float> *encoded_out, int *encoded_frames_out) {
  const int window_size = total_features - window_start;
  std::vector<int64_t> feat_shape = {1, window_size, config.encoder_dim};

  OrtValue *features_tensor = nullptr;
  const float *features_ptr =
      state->accumulated_features.data() +
      static_cast<size_t>(window_start) * config.encoder_dim;
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

  stats.encoder_frames += window_size;
  OrtStatus *status =
      ORT_RUN(ort_api, encoder_session, enc_input_names, &features_tensor, 1,
              enc_output_names, 1, enc_outputs);
  ort_api->ReleaseValue(features_tensor);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    return 1;
  }

  const int frames = read_frames_tensor(ort_api, enc_outputs[0],
                                        config.encoder_dim, encoded_out);
  ort_api->ReleaseValue(enc_outputs[0]);
  if (frames < 0) {
    LOG("Unexpected encoder output shape\n");
    return 1;
  }
  *encoded_frames_out = frames;
  return 0;
}

int MoonshineStreamingModel::run_encoder_cached(MoonshineStreamingState *state,
                                                int window_start,
                                                int total_features,
                                                bool use_cache, bool is_final,
                                                std::vector<float> *encoded_out,
                                                int *encoded_frames_out) {
  const int window_size = total_features - window_start;
  const size_t cache_size = static_cast<size_t>(config.depth) *
                            encoder_cache_frames * config.encoder_dim;
  if (state->encoder_cache.size() != cache_size) {
    state->encoder_cache.assign(cache_size, 0.0f);
    state->encoder_cache_len = 0;
  }
  // Valid cache slots are the last encoder_cache_len of each layer's; with
  // use_cache false every slot is masked and the window carries the context.
  std::vector<float> cache_valid(encoder_cache_frames, 0.0f);
  if (use_cache) {
    for (int i = encoder_cache_frames - state->encoder_cache_len;
         i < encoder_cache_frames; ++i) {
      cache_valid[i] = 1.0f;
    }
  }

  std::vector<int64_t> feat_shape = {1, window_size, config.encoder_dim};
  std::vector<int64_t> cache_shape = {config.depth, 1, encoder_cache_frames,
                                      config.encoder_dim};
  std::vector<int64_t> valid_shape = {encoder_cache_frames};

  OrtValue *inputs[3] = {nullptr, nullptr, nullptr};
  const float *features_ptr =
      state->accumulated_features.data() +
      static_cast<size_t>(window_start) * config.encoder_dim;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, const_cast<float *>(features_ptr),
          static_cast<size_t>(window_size) * config.encoder_dim * sizeof(float),
          feat_shape.data(), feat_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[0]));
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, state->encoder_cache.data(),
          cache_size * sizeof(float), cache_shape.data(), cache_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[1]));
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, cache_valid.data(),
          cache_valid.size() * sizeof(float), valid_shape.data(),
          valid_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[2]));

  const char *input_names[] = {"features", "cache", "cache_valid"};
  const char *output_names[] = {"encoded", "cache_out"};
  OrtValue *outputs[2] = {nullptr, nullptr};
  stats.encoder_frames += window_size;
  OrtStatus *status = ORT_RUN(ort_api, encoder_session, input_names, inputs, 3,
                              output_names, 2, outputs);
  for (OrtValue *value : inputs) ort_api->ReleaseValue(value);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    return 1;
  }

  const int frames =
      read_frames_tensor(ort_api, outputs[0], config.encoder_dim, encoded_out);
  int err = 0;
  if (frames < 0) {
    LOG("Unexpected cached encoder output shape\n");
    err = 1;
  } else if (is_final) {
    // The graph cuts the next cache from before the lookahead, and a final
    // flush has none, so what it produced does not line up with any frame
    // boundary. Nothing follows a final flush in this stream anyway; mark the
    // cache unusable so a caller that keeps going gets the full-context path.
    state->encoder_cache_len = -1;
  } else {
    float *cache_out = nullptr;
    if (ort_api->GetTensorMutableData(outputs[1], (void **)&cache_out) !=
        nullptr) {
      err = 1;
    } else {
      memcpy(state->encoder_cache.data(), cache_out,
             cache_size * sizeof(float));
      // The new cache ends at the frame after the last one emitted this call.
      const int emitted_after = total_features - config.total_lookahead;
      state->encoder_cache_len = std::min(encoder_cache_frames, emitted_after);
    }
  }
  for (OrtValue *value : outputs) ort_api->ReleaseValue(value);
  if (err != 0) return err;
  *encoded_frames_out = frames;
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

  // Every layer looks back encoder_cache_frames frames, so an exact recompute
  // of the new frames needs that much left context per layer.
  const int left_context_frames = encoder_cache_frames * config.depth;
  const int emitted = state->encoder_frames_emitted;
  // The cached graph only needs the window to start at the first unemitted
  // frame, provided the cache really holds the frames before it. After a
  // reset, a final flush, or with the cache switched off, the cache is not
  // trusted and the full left context is fed instead, which the cached graph
  // computes identically to the stateless one.
  const bool cache_ready =
      encoder_has_cache && use_encoder_cache &&
      state->encoder_cache_len == std::min(encoder_cache_frames, emitted) &&
      !state->encoder_cache.empty();
  int window_start = cache_ready ? emitted : emitted - left_context_frames;
  if (window_start < 0) {
    window_start = 0;
  }
  const int window_size = total_features - window_start;
  if (log_ort_run) {
    LOGF("streaming encode: window_start=%d, window_size=%d, cached=%d",
         window_start, window_size, cache_ready ? 1 : 0);
  }

  std::lock_guard<std::mutex> lock(processing_mutex);

  std::vector<float> encoded;
  int total_encoded = 0;
  if (encoder_has_cache) {
    RETURN_ON_ERROR(run_encoder_cached(state, window_start, total_features,
                                       cache_ready, is_final, &encoded,
                                       &total_encoded));
  } else {
    RETURN_ON_ERROR(run_encoder_stateless(state, window_start, total_features,
                                          &encoded, &total_encoded));
  }

  // Run adapter on new frames
  const int start_idx = emitted - window_start;
  if (start_idx < 0 || start_idx + new_frames > total_encoded) {
    LOGF("Encoder window misaligned: start_idx=%d, new_frames=%d, total=%d",
         start_idx, new_frames, total_encoded);
    return 1;
  }
  std::vector<float> new_encoded(
      encoded.begin() + static_cast<size_t>(start_idx) * config.encoder_dim,
      encoded.begin() +
          static_cast<size_t>(start_idx + new_frames) * config.encoder_dim);

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

  OrtStatus *status =
      ORT_RUN(ort_api, adapter_session, adapter_input_names, adapter_inputs, 2,
              adapter_output_names, 1, adapter_outputs);

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
    return 1;
  }

  // Memory is append-only within a stream, so the K/V already held for
  // frames [0, cross_len) are still right; only the frames past them need
  // projecting. Anything inconsistent (cache longer than memory, sizes that
  // do not match) falls back to projecting everything.
  const size_t per_frame =
      static_cast<size_t>(config.depth) * config.nheads * config.head_dim;
  const bool can_append =
      append_cross_kv && state->cross_len > 0 &&
      state->cross_len < state->memory_len &&
      state->k_cross.size() == per_frame * state->cross_len &&
      state->v_cross.size() == per_frame * state->cross_len;
  const int first_frame = can_append ? state->cross_len : 0;
  if (state->cross_len == state->memory_len &&
      state->k_cross.size() == per_frame * state->cross_len &&
      state->v_cross.size() == per_frame * state->cross_len) {
    state->cross_kv_valid = true;
    return 0;
  }
  const int new_frames = state->memory_len - first_frame;

  // Input: memory [1, new_frames, decoder_dim]
  std::vector<int64_t> memory_shape = {1, new_frames, config.decoder_dim};

  OrtValue *memory_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info,
          state->memory.data() +
              static_cast<size_t>(first_frame) * config.decoder_dim,
          static_cast<size_t>(new_frames) * config.decoder_dim * sizeof(float),
          memory_shape.data(), memory_shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &memory_tensor));

  // Run cross_kv session
  const char *input_names[] = {"memory"};
  const char *output_names[] = {"k_cross", "v_cross"};

  OrtValue *inputs[] = {memory_tensor};
  OrtValue *outputs[2] = {nullptr, nullptr};

  stats.cross_kv_frames += new_frames;
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

  const int out_frames = static_cast<int>(k_shape[3]);
  if (out_frames != new_frames) {
    LOGF("cross_kv produced %d frames for %d memory frames", out_frames,
         new_frames);
    for (int i = 0; i < 2; i++) ort_api->ReleaseValue(outputs[i]);
    return 1;
  }

  float *k_data = nullptr;
  float *v_data = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[0], (void **)&k_data));
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[1], (void **)&v_data));

  const int total_frames = first_frame + new_frames;
  const size_t kv_size = per_frame * total_frames;
  if (first_frame == 0) {
    state->k_cross.resize(kv_size);
    state->v_cross.resize(kv_size);
    memcpy(state->k_cross.data(), k_data, kv_size * sizeof(float));
    memcpy(state->v_cross.data(), v_data, kv_size * sizeof(float));
  } else {
    // The decoder wants a dense [depth, 1, nheads, frames, head_dim] tensor,
    // so growing the frames axis means every (layer, head) block moves. That
    // is one pass of memcpy over the cache, versus a matmul over all of
    // memory to recompute it.
    const size_t old_block = static_cast<size_t>(first_frame) * config.head_dim;
    const size_t new_block = static_cast<size_t>(new_frames) * config.head_dim;
    const size_t blocks = static_cast<size_t>(config.depth) * config.nheads;
    std::vector<float> k_next(kv_size);
    std::vector<float> v_next(kv_size);
    for (size_t b = 0; b < blocks; ++b) {
      float *k_dst = k_next.data() + b * (old_block + new_block);
      float *v_dst = v_next.data() + b * (old_block + new_block);
      memcpy(k_dst, state->k_cross.data() + b * old_block,
             old_block * sizeof(float));
      memcpy(v_dst, state->v_cross.data() + b * old_block,
             old_block * sizeof(float));
      memcpy(k_dst + old_block, k_data + b * new_block,
             new_block * sizeof(float));
      memcpy(v_dst + old_block, v_data + b * new_block,
             new_block * sizeof(float));
    }
    state->k_cross.swap(k_next);
    state->v_cross.swap(v_next);
  }
  state->cross_len = total_frames;
  state->cross_kv_valid = true;

  // Release outputs
  for (int i = 0; i < 2; i++) {
    ort_api->ReleaseValue(outputs[i]);
  }

  return 0;
}

void MoonshineStreamingModel::truncate_self_cache(
    MoonshineStreamingState *state, int keep_len) {
  if (state == nullptr) return;
  if (keep_len <= 0) {
    state->k_self.clear();
    state->v_self.clear();
    state->cache_seq_len = 0;
    return;
  }
  if (keep_len >= state->cache_seq_len) return;
  // Layout is [depth, 1, nheads, cache_seq_len, head_dim]: shrinking the
  // sequence axis packs each (layer, head) block up against the previous one.
  const size_t blocks = static_cast<size_t>(config.depth) * config.nheads;
  const size_t old_block =
      static_cast<size_t>(state->cache_seq_len) * config.head_dim;
  const size_t new_block = static_cast<size_t>(keep_len) * config.head_dim;
  for (size_t b = 1; b < blocks; ++b) {
    memmove(state->k_self.data() + b * new_block,
            state->k_self.data() + b * old_block, new_block * sizeof(float));
    memmove(state->v_self.data() + b * new_block,
            state->v_self.data() + b * old_block, new_block * sizeof(float));
  }
  state->k_self.resize(blocks * new_block);
  state->v_self.resize(blocks * new_block);
  state->cache_seq_len = keep_len;
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
  // Build output names dynamically (base 5 + optional cross_attentions)
  size_t decoder_output_count = 0;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetOutputCount(
                                   decoder_kv_session, &decoder_output_count));

  std::vector<const char *> output_names_vec;
  std::vector<char *> output_names_alloc(decoder_output_count);
  for (size_t i = 0; i < decoder_output_count; i++) {
    RETURN_ON_ORT_ERROR(ort_api,
                        ort_api->SessionGetOutputName(decoder_kv_session, i,
                                                      &ort_allocator->base,
                                                      &output_names_alloc[i]));
    output_names_vec.push_back(output_names_alloc[i]);
  }

  const char *input_names[] = {"token", "k_self", "v_self", "out_k_cross",
                               "out_v_cross"};
  OrtValue *inputs[] = {token_tensor, k_self_tensor, v_self_tensor,
                        k_cross_tensor, v_cross_tensor};
  std::vector<OrtValue *> outputs(decoder_output_count, nullptr);

  stats.decoder_calls += 1;
  stats.decoder_tokens += token_len;
  OrtStatus *status =
      ORT_RUN(ort_api, decoder_kv_session, input_names, inputs, 5,
              output_names_vec.data(), decoder_output_count, outputs.data());

  // Release inputs
  for (int i = 0; i < 5; i++) {
    ort_api->ReleaseValue(inputs[i]);
  }

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    for (auto *n : output_names_alloc)
      ort_allocator->base.Free(&ort_allocator->base, n);
    return 1;
  }

  // Build name→index map for outputs
  std::map<std::string, size_t> output_index;
  for (size_t i = 0; i < decoder_output_count; i++) {
    output_index[output_names_vec[i]] = i;
  }

  // Copy logits [1, token_len, vocab_size]
  float *logits_data = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->GetTensorMutableData(outputs[output_index["logits"]],
                                             (void **)&logits_data));

  size_t total_logits = token_len * config.vocab_size;
  logits_out.resize(total_logits);
  memcpy(logits_out.data(), logits_data, total_logits * sizeof(float));

  // Update self-attention KV cache
  size_t k_idx = output_index["out_k_self"];
  OrtTensorTypeAndShapeInfo *k_info = nullptr;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetTensorTypeAndShape(outputs[k_idx], &k_info));
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
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   outputs[k_idx], (void **)&k_out_data));
  size_t v_idx = output_index["out_v_self"];
  RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                   outputs[v_idx], (void **)&v_out_data));

  memcpy(state->k_self.data(), k_out_data, new_cache_size * sizeof(float));
  memcpy(state->v_self.data(), v_out_data, new_cache_size * sizeof(float));
  state->cache_seq_len = new_cache_len;

  // Collect cross-attention weights if decoder has them
  for (int layer = 0; layer < config.depth; layer++) {
    std::string attn_name = "cross_attentions." + std::to_string(layer);
    auto it = output_index.find(attn_name);
    if (it == output_index.end()) break;

    OrtTensorTypeAndShapeInfo *attn_info = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorTypeAndShape(
                                     outputs[it->second], &attn_info));
    size_t attn_ndims = 0;
    RETURN_ON_ORT_ERROR(ort_api,
                        ort_api->GetDimensionsCount(attn_info, &attn_ndims));
    std::vector<int64_t> attn_shape(attn_ndims);
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetDimensions(
                                     attn_info, attn_shape.data(), attn_ndims));
    ort_api->ReleaseTensorTypeAndShapeInfo(attn_info);

    // Shape: [batch, heads, seq_len, enc_len] or similar
    int heads =
        (attn_ndims >= 2) ? static_cast<int>(attn_shape[1]) : config.nheads;
    int enc_len = (attn_ndims >= 4)   ? static_cast<int>(attn_shape[3])
                  : (attn_ndims >= 3) ? static_cast<int>(attn_shape[2])
                                      : state->cross_len;

    float *attn_data = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                     outputs[it->second], (void **)&attn_data));

    size_t step_size = 1;
    for (size_t d = 0; d < attn_ndims; d++) step_size *= attn_shape[d];

    size_t old_size = cross_attention_buffer.size();
    cross_attention_buffer.resize(old_size + step_size);
    memcpy(cross_attention_buffer.data() + old_size, attn_data,
           step_size * sizeof(float));

    if (cross_attn_steps == 0) {
      cross_attn_heads = heads;
      cross_attn_enc_len = enc_len;
    }
    cross_attn_steps++;
  }

  // Release outputs
  for (size_t i = 0; i < decoder_output_count; i++) {
    if (outputs[i]) ort_api->ReleaseValue(outputs[i]);
  }
  for (auto *n : output_names_alloc) {
    ort_allocator->base.Free(&ort_allocator->base, n);
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
    write_empty_decode_logits(config, logits_out);
    return 0;
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
    for (int i = 0; i < tokens_len; ++i) {
      write_empty_decode_logits(config, logits_out + i * config.vocab_size);
    }
    return 0;
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
                                         int *tokens_len_out,
                                         ContextBiaser *biaser) {
  if (state == nullptr) {
    LOG("State is null\n");
    return 1;
  }
  if (tokens_out == nullptr || tokens_len_out == nullptr) {
    LOG("Output pointers are null\n");
    return 1;
  }
  if (state->memory_len == 0) {
    *tokens_out = nullptr;
    *tokens_len_out = 0;
    return 0;
  }

  std::lock_guard<std::mutex> lock(processing_mutex);

  std::vector<int> result_tokens;

  // Compute max tokens based on memory length (similar to new decoder)
  float duration_sec = state->memory_len * 0.020f;
  const int max_tokens = std::min(
      static_cast<int>(std::ceil(duration_sec * 6.5)), config.max_seq_len);

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

  // Contextual biasing, if the caller supplied key terms. The bonuses go into
  // the logits row in place, which is safe because every logits buffer here is
  // local scratch. The walk starts at the root: this function always decodes
  // from BOS, even when it is verifying a draft. It is advanced exactly once
  // per accepted token, so at every argmax it reflects the tokens that
  // position is conditioned on.
  if (biaser != nullptr) {
    biaser->reset();
  }
  auto biased_argmax = [&](float *logits_row) -> int {
    if (biaser != nullptr) {
      biaser->apply(logits_row, config.vocab_size);
    }
    return argmax(logits_row);
  };

  // Compute cross K/V upfront
  if (!state->cross_kv_valid) {
    int err = compute_cross_kv(state);
    if (err != 0) {
      LOG("Failed to compute cross K/V\n");
      return err;
    }
  }

  // Emit one token the model has chosen. Returns false when decoding is over.
  auto accept = [&](int token) -> bool {
    if (token == config.eos_id ||
        result_tokens.size() >= static_cast<size_t>(max_tokens)) {
      return false;
    }
    result_tokens.push_back(token);
    if (biaser != nullptr) {
      biaser->advance(token);
    }
    return true;
  };

  // Feed ``head`` followed by ``cand`` through the decoder in one call and
  // accept the longest prefix of ``cand`` the model agrees with. ``head`` has
  // already been accepted (or is BOS); row i of the logits predicts the token
  // after head + cand[0, i). The self-attention cache is cut back to the
  // tokens that were accepted, so a rejected tail costs nothing beyond this
  // call. On return ``*next`` holds the model's token for the first position
  // it disagreed on (or the position after the whole candidate). Returns the
  // number of candidate tokens accepted, or -1 on error.
  auto verify = [&](int head, const int *cand, int cand_len, int *next) -> int {
    std::vector<int64_t> tokens;
    tokens.reserve(static_cast<size_t>(cand_len) + 1);
    tokens.push_back(head);
    for (int i = 0; i < cand_len; ++i) {
      tokens.push_back(cand[i]);
    }
    const int cache_before = state->cache_seq_len;
    std::vector<float> logits;
    if (run_decoder_with_cross_kv(state, tokens, logits) != 0) {
      return -1;
    }
    int accepted = 0;
    *next = config.eos_id;
    for (int i = 0; i <= cand_len; ++i) {
      const int pred = biased_argmax(logits.data() + static_cast<size_t>(i) *
                                                         config.vocab_size);
      if (i == cand_len || pred != cand[i]) {
        *next = pred;
        break;
      }
      if (!accept(pred)) {
        // Hit EOS or the token budget inside the draft: stop here. ``next``
        // is EOS so the caller ends the line.
        *next = config.eos_id;
        break;
      }
      ++accepted;
    }
    truncate_self_cache(state, cache_before + 1 + accepted);
    return accepted;
  };

  SpeculativeDecodeHooks hooks;
  hooks.verify = verify;
  hooks.accept = accept;
  const SpeculativeDecodeOutcome outcome =
      run_speculative_decode(speculative_tokens, speculative_len, config.bos_id,
                             max_realign_attempts, hooks);
  stats.realign_attempts += outcome.realign_attempts;
  if (!outcome.ok) {
    return 1;
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
