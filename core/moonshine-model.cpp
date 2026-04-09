#include "moonshine-model.h"

#if defined(ANDROID)
#include "nnapi_provider_factory.h"
#endif

#include <fcntl.h>

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cerrno>  // For errno
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstring>  // For strerror

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <numeric>
#include <vector>

#include "bin-tokenizer.h"
#include "moonshine-ort-allocator.h"
#include "moonshine-tensor-view.h"
#include "string-utils.h"

#define DEBUG_ALLOC_ENABLED 1
#include "debug-utils.h"
#include "ort-utils.h"

// Tiny settings.
#define MOONSHINE_TINY_NUM_LAYERS 6
#define MOONSHINE_TINY_NUM_KV_HEADS 8
#define MOONSHINE_TINY_HEAD_DIM 36

#define MOONSHINE_TINY_PAST_ELEMENT_COUNT \
  (MOONSHINE_TINY_NUM_KV_HEADS * MOONSHINE_TINY_HEAD_DIM)

// Base settings.
#define MOONSHINE_BASE_NUM_LAYERS 8
#define MOONSHINE_BASE_NUM_KV_HEADS 8
#define MOONSHINE_BASE_HEAD_DIM 52

#define MOONSHINE_BASE_PAST_ELEMENT_COUNT \
  (MOONSHINE_BASE_NUM_KV_HEADS * MOONSHINE_BASE_HEAD_DIM)

#define MOONSHINE_DECODER_START_TOKEN_ID 1
#define MOONSHINE_EOS_TOKEN_ID 2

namespace {
int set_model_options_from_arch(MoonshineModel *model, int32_t model_arch) {
  if (model_arch == MOONSHINE_MODEL_ARCH_TINY) {
    model->num_layers = MOONSHINE_TINY_NUM_LAYERS;
    model->num_kv_heads = MOONSHINE_TINY_NUM_KV_HEADS;
    model->head_dim = MOONSHINE_TINY_HEAD_DIM;
    model->past_element_count = MOONSHINE_TINY_PAST_ELEMENT_COUNT;
  } else if (model_arch == MOONSHINE_MODEL_ARCH_BASE) {
    model->num_layers = MOONSHINE_BASE_NUM_LAYERS;
    model->num_kv_heads = MOONSHINE_BASE_NUM_KV_HEADS;
    model->head_dim = MOONSHINE_BASE_HEAD_DIM;
    model->past_element_count = MOONSHINE_BASE_PAST_ELEMENT_COUNT;
  } else {
    LOGF(
        "Invalid model architecture: %d, must be MOONSHINE_MODEL_ARCH_TINY (0) "
        "or MOONSHINE_MODEL_ARCH_BASE (1)\n",
        model_arch);
    return 1;
  }
  return 0;
}
}  // namespace

MoonshineModel::MoonshineModel(bool log_ort_run, float max_tokens_per_second,
                               bool use_nnapi)
    : encoder_session(nullptr),
      decoder_session(nullptr),
      tokenizer(nullptr),
      max_tokens_per_second(max_tokens_per_second),
      log_ort_run(log_ort_run) {
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  LOG_ORT_ERROR(ort_api, ort_api->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                            "MoonshineModel", &ort_env));
  LOG_ORT_ERROR(ort_api,
                ort_api->CreateCpuMemoryInfo(
                    OrtDeviceAllocator, OrtMemTypeDefault, &ort_memory_info));
  // Use a custom allocator that tracks memory usage.
  ort_session_allocator = new MoonshineOrtAllocator(ort_memory_info);
  // LOG_ORT_ERROR(ort_api, ort_api->RegisterAllocator(
  //                              ort_env, &ort_session_allocator->base));

  // Used only for our string allocations, to avoid memory fragmentation.
  ort_string_allocator = new MoonshineOrtAllocator(ort_memory_info);

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
                    ort_session_options, "session.use_env_allocators", "1"));
  LOG_ORT_ERROR(ort_api,
                ort_api->AddSessionConfigEntry(
                    ort_session_options, "session.disable_prepacking", "1"));
  LOG_ORT_ERROR(ort_api, ort_api->DisableCpuMemArena(ort_session_options));

#if defined(ANDROID)
  if (use_nnapi) {
    uint32_t nnapi_flags = NNAPI_FLAG_CPU_DISABLED;
    OrtStatus *status = OrtSessionOptionsAppendExecutionProvider_Nnapi(
        ort_session_options, nnapi_flags);
    if (status != nullptr) {
      fprintf(stderr, "Warning: NNAPI EP not available, falling back to CPU\n");
      ort_api->ReleaseStatus(status);
    }
  }
#else
  (void)use_nnapi;
#endif
}

MoonshineModel::~MoonshineModel() {
  ort_api->ReleaseEnv(ort_env);
  ort_api->ReleaseMemoryInfo(ort_memory_info);
  ort_api->ReleaseSessionOptions(ort_session_options);
  ort_api->ReleaseSession(encoder_session);
  ort_api->ReleaseSession(decoder_session);
  delete ort_session_allocator;
  delete ort_string_allocator;
  delete tokenizer;
#ifndef _WIN32
  if (encoder_mmapped_data) {
    munmap(const_cast<char *>(encoder_mmapped_data), encoder_mmapped_data_size);
  }
  if (decoder_mmapped_data) {
    munmap(const_cast<char *>(decoder_mmapped_data), decoder_mmapped_data_size);
  }
#endif
}

int MoonshineModel::load(const char *encoder_model_path,
                         const char *decoder_model_path,
                         const char *tokenizer_path, int32_t model_type) {
  RETURN_ON_ERROR(set_model_options_from_arch(this, model_type));

  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, encoder_model_path,
      &encoder_session, &encoder_mmapped_data, &encoder_mmapped_data_size));
  RETURN_ON_NULL(encoder_session);
  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, decoder_model_path,
      &decoder_session, &decoder_mmapped_data, &decoder_mmapped_data_size));
  RETURN_ON_NULL(decoder_session);
  tokenizer = new BinTokenizer(tokenizer_path);
  RETURN_ON_NULL(tokenizer);
  return 0;
}

int MoonshineModel::load_from_memory(const uint8_t *encoder_model_data,
                                     size_t encoder_model_data_size,
                                     const uint8_t *decoder_model_data,
                                     size_t decoder_model_data_size,
                                     const uint8_t *tokenizer_data,
                                     size_t tokenizer_data_size,
                                     int32_t model_type) {
  RETURN_ON_ERROR(set_model_options_from_arch(this, model_type));

  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, encoder_model_data,
      encoder_model_data_size, &encoder_session));
  RETURN_ON_NULL(encoder_session);
  RETURN_ON_ERROR(ort_session_from_memory(
      ort_api, ort_env, ort_session_options, decoder_model_data,
      decoder_model_data_size, &decoder_session));
  RETURN_ON_NULL(decoder_session);
  tokenizer = new BinTokenizer(tokenizer_data, tokenizer_data_size);
  RETURN_ON_NULL(tokenizer);
  return 0;
}

#if defined(ANDROID)
int MoonshineModel::load_from_assets(const char *encoder_model_path,
                                     const char *decoder_model_path,
                                     const char *tokenizer_path,
                                     int32_t model_type,
                                     AAssetManager *assetManager) {
  RETURN_ON_ERROR(set_model_options_from_arch(this, model_type));

  //      session_options->EnableProfiling("encoder");

  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager, encoder_model_path,
      &encoder_session, &encoder_mmapped_data, &encoder_mmapped_data_size));
  RETURN_ON_NULL(encoder_session);

  //      session_options->EnableProfiling("decoder");

  RETURN_ON_ERROR(ort_session_from_asset(
      ort_api, ort_env, ort_session_options, assetManager, decoder_model_path,
      &decoder_session, &decoder_mmapped_data, &decoder_mmapped_data_size));
  RETURN_ON_NULL(decoder_session);

  tokenizer = new BinTokenizer(tokenizer_path, assetManager);
  if (tokenizer == nullptr) {
    LOGF("Failed to load tokenizer from '%s'\n", tokenizer_path);
    return 1;
  }
  return 0;
}
#endif

int MoonshineModel::transcribe(const float *input_audio_data,
                               size_t input_audio_data_size, char **out_text) {
  // TIMER_START(moonshine_transcribe);

  *out_text = nullptr;
  if (input_audio_data == nullptr || input_audio_data_size == 0) {
    LOG("Audio data is nullptr or empty");
    return 1;
  }
  size_t encoder_input_count = 0;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetInputCount(
                                   encoder_session, &encoder_input_count));
  size_t encoder_output_count = 0;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetOutputCount(
                                   encoder_session, &encoder_output_count));
  std::vector<char *> encoder_input_names(encoder_input_count);
  std::vector<char *> encoder_output_names(encoder_output_count);
  for (size_t i = 0; i < encoder_input_count; i++) {
    RETURN_ON_ORT_ERROR(ort_api,
                        ort_api->SessionGetInputName(
                            encoder_session, i, &ort_string_allocator->base,
                            &(encoder_input_names.data())[i]));
  }
  for (size_t i = 0; i < encoder_output_count; i++) {
    RETURN_ON_ORT_ERROR(ort_api,
                        ort_api->SessionGetOutputName(
                            encoder_session, i, &ort_string_allocator->base,
                            &(encoder_output_names.data())[i]));
  }
  std::vector<int64_t> encoder_input_shape =
      ort_get_input_shape(ort_api, encoder_session, 0);
  encoder_input_shape[0] = 1;
  encoder_input_shape[1] = input_audio_data_size;
  MoonshineTensorView *encoder_input_tensor = new MoonshineTensorView(
      encoder_input_shape, ort_get_input_type(ort_api, encoder_session, 0),
      const_cast<float *>(input_audio_data), "encoder_input_tensor");
  std::vector<OrtValue *> encoder_inputs;
  encoder_inputs.push_back(
      encoder_input_tensor->create_ort_value(ort_api, ort_memory_info));
  // Newer versions of the optimum onnx converter include an attention mask
  // input.
  MoonshineTensorView *encoder_attention_mask_tensor = nullptr;
  if (encoder_input_count > 1) {
    encoder_attention_mask_tensor = new MoonshineTensorView(
        {1, static_cast<int64_t>(input_audio_data_size)}, MOONSHINE_DTYPE_INT64,
        nullptr, "encoder_attention_mask");
    for (size_t i = 0; i < input_audio_data_size; i++) {
      encoder_attention_mask_tensor->data<int64_t>()[i] = 1;
    }
    encoder_inputs.push_back(encoder_attention_mask_tensor->create_ort_value(
        ort_api, ort_memory_info));
  }
  std::vector<OrtValue *> encoder_outputs(encoder_output_count);
  // TIMER_START(moonshine_encoder_run);
  RETURN_ON_ORT_ERROR(
      ort_api, ORT_RUN(ort_api, encoder_session, encoder_input_names.data(),
                       encoder_inputs.data(), encoder_input_count,
                       encoder_output_names.data(), encoder_output_count,
                       encoder_outputs.data()));
  // TIMER_END(moonshine_encoder_run);
  for (size_t i = 0; i < encoder_inputs.size(); i++) {
    ort_api->ReleaseValue(encoder_inputs[i]);
  }
  for (char *encoder_input_name : encoder_input_names) {
    ort_string_allocator->base.Free(&ort_string_allocator->base,
                                    encoder_input_name);
  }
  for (char *encoder_output_name : encoder_output_names) {
    ort_string_allocator->base.Free(&ort_string_allocator->base,
                                    encoder_output_name);
  }
  OrtValue *last_hidden_state = encoder_outputs[0];
  MoonshineTensorView *last_hidden_state_tensor = new MoonshineTensorView(
      ort_api, last_hidden_state, "last_hidden_state_tensor");
  ort_api->ReleaseValue(last_hidden_state);

  // Save encoder hidden states for word alignment (if alignment model loaded)
  if (alignment_session != nullptr) {
    auto &enc_shape = last_hidden_state_tensor->shape();
    // Shape: [1, enc_frames, encoder_dim]
    last_encoder_frames = static_cast<int>(enc_shape[1]);
    encoder_dim = static_cast<int>(enc_shape[2]);
    size_t total = last_encoder_frames * encoder_dim;
    last_encoder_hidden_states.resize(total);
    memcpy(last_encoder_hidden_states.data(),
           last_hidden_state_tensor->data<float>(), total * sizeof(float));
  }

  delete encoder_input_tensor;
  encoder_input_tensor = nullptr;

  size_t decoder_input_count;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetInputCount(
                                   decoder_session, &decoder_input_count));

  const size_t expected_decoder_input_count_v1 = (num_layers * 4) + 3;
  const size_t expected_decoder_input_count_v2 = (num_layers * 4) + 4;
  if ((decoder_input_count != expected_decoder_input_count_v1) &&
      (decoder_input_count != expected_decoder_input_count_v2)) {
    LOGF(
        "Expected decoder input count to be %zu or "
        "%zu, but got %zu. This "
        "often indicates you're specifying the "
        "wrong model architecture "
        "(for example tiny instead of base).\n",
        expected_decoder_input_count_v1, expected_decoder_input_count_v2,
        decoder_input_count);
    return 1;
  }

  std::vector<const char *> decoder_input_names(decoder_input_count);
  for (size_t i = 0; i < decoder_input_count; i++) {
    char *decoder_input_name = nullptr;
    RETURN_ON_ORT_ERROR(
        ort_api, ort_api->SessionGetInputName(decoder_session, i,
                                              &ort_string_allocator->base,
                                              &decoder_input_name));
    decoder_input_names[i] = decoder_input_name;
  }
  size_t decoder_output_count;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetOutputCount(
                                   decoder_session, &decoder_output_count));
  std::vector<const char *> decoder_output_names(decoder_output_count);
  for (size_t i = 0; i < decoder_output_count; i++) {
    char *decoder_output_name = nullptr;
    RETURN_ON_ORT_ERROR(
        ort_api, ort_api->SessionGetOutputName(decoder_session, i,
                                               &ort_string_allocator->base,
                                               &decoder_output_name));
    decoder_output_names[i] = decoder_output_name;
  }
  const float audio_duration = input_audio_data_size / 16000.0f;
  const int max_len =
      static_cast<int>(std::ceil(audio_duration * this->max_tokens_per_second));
  auto decoder_input_name_to_index = name_to_index(decoder_input_names);
  auto decoder_output_name_to_index = name_to_index(decoder_output_names);
  std::vector<std::string> layer_suffixes;
  std::map<std::string, MoonshineTensorView *> past_key_values_by_name;
  for (int i = 0; i < num_layers; i++) {
    for (const char *a : {"decoder", "encoder"}) {
      for (const char *b : {"key", "value"}) {
        std::vector<int64_t> past_key_values_shape = {1, num_kv_heads, 1,
                                                      head_dim};
        std::string layer_suffix = std::to_string(i) + "." + a + "." + b;
        layer_suffixes.push_back(layer_suffix);
        std::string past_key_values_name = "past_key_values." + layer_suffix;
        MoonshineTensorView *tensor_and_data = new MoonshineTensorView(
            past_key_values_shape, MOONSHINE_DTYPE_FLOAT32, nullptr,
            TENSOR_NAME(past_key_values_name));
        past_key_values_by_name[past_key_values_name] = tensor_and_data;
      }
    }
  }

  std::vector<int64_t> tokens = {MOONSHINE_DECODER_START_TOKEN_ID};
  std::vector<int64_t> inputIDs = tokens;

  // Buffer for collecting cross-attention weights during single-pass decoding.
  // Filled when the decoder model has cross_attentions.* outputs.
  std::vector<float> cross_attention_buffer;
  int cross_attn_heads = 0;
  int cross_attn_enc_len = 0;
  int cross_attn_steps = 0;

  for (int token_index = 0; token_index < max_len; token_index++) {
    const bool use_cache_branch = token_index > 0;
    std::vector<MoonshineTensorView *> decoder_inputs_data(decoder_input_count);

    std::vector<int64_t> input_ids_shape = {
        1, static_cast<int64_t>(inputIDs.size())};
    const int64_t input_ids_index = decoder_input_name_to_index["input_ids"];
    decoder_inputs_data[input_ids_index] =
        new MoonshineTensorView(input_ids_shape, MOONSHINE_DTYPE_INT64,
                                inputIDs.data(), TENSOR_NAME("input_ids"));

    const int64_t encoder_hidden_states_index =
        decoder_input_name_to_index["encoder_hidden_states"];
    decoder_inputs_data[encoder_hidden_states_index] =
        new MoonshineTensorView(*last_hidden_state_tensor);

    // The attention mask is optional: the attention-enabled decoder
    // (decoder_with_attention.ort) omits this input since it was not
    // present in the original HuggingFace export. Only pass it if
    // the decoder actually accepts it.
    if (encoder_attention_mask_tensor != nullptr &&
        decoder_input_name_to_index.find("encoder_attention_mask") !=
            decoder_input_name_to_index.end()) {
      const int64_t encoder_attention_mask_index =
          decoder_input_name_to_index["encoder_attention_mask"];
      decoder_inputs_data[encoder_attention_mask_index] =
          new MoonshineTensorView(*encoder_attention_mask_tensor);
    }

    std::vector<uint8_t> use_cache_branch_vector = {
        use_cache_branch ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0)};
    std::vector<int64_t> use_cache_branch_shape = {1};
    const int64_t use_cache_branch_index =
        decoder_input_name_to_index["use_cache_branch"];
    decoder_inputs_data[use_cache_branch_index] = new MoonshineTensorView(
        use_cache_branch_shape, MOONSHINE_DTYPE_BOOL,
        use_cache_branch_vector.data(), TENSOR_NAME("use_cache_branch"));
    for (const auto &[key, value] : past_key_values_by_name) {
      RETURN_ON_FALSE(decoder_input_name_to_index.find(key) !=
                      decoder_input_name_to_index.end());
      const int64_t key_index = decoder_input_name_to_index[key];
      if (decoder_inputs_data[key_index] != nullptr) {
        LOGF(
            "Decoder input data for key %s is not "
            "nullptr\n",
            key.c_str());
        return 1;
      }
      decoder_inputs_data[key_index] = new MoonshineTensorView(*value);
    }
    std::vector<OrtValue *> decoder_inputs;
    for (size_t i = 0; i < decoder_input_count; i++) {
      MoonshineTensorView *decoder_view = decoder_inputs_data[i];
      if (decoder_view == nullptr) {
        LOGF("Decoder input %s is nullptr\n", decoder_input_names[i]);
        return 1;
      }
      decoder_inputs.push_back(
          decoder_view->create_ort_value(ort_api, ort_memory_info));
    }

    // TIMER_START(moonshine_decoder_run);
    std::vector<OrtValue *> decoder_outputs(decoder_output_count);
    RETURN_ON_ORT_ERROR(
        ort_api, ORT_RUN(ort_api, decoder_session, decoder_input_names.data(),
                         decoder_inputs.data(), decoder_inputs.size(),
                         decoder_output_names.data(),
                         decoder_output_names.size(), decoder_outputs.data()));
    // TIMER_END(moonshine_decoder_run);
    for (size_t i = 0; i < decoder_inputs_data.size(); i++) {
      delete decoder_inputs_data[i];
      ort_api->ReleaseValue(decoder_inputs[i]);
    }
    decoder_inputs_data.clear();

    OrtValue *logits_tensor = decoder_outputs[0];
    MoonshineTensorView logits_tensor_view =
        MoonshineTensorView(ort_api, logits_tensor, "logits_tensor");

    // Copy over the values output from the last run
    // into the corresponding inputs for the next
    // decoding run.
    for (const auto &layer_suffix : layer_suffixes) {
      if (!use_cache_branch ||
          layer_suffix.find("decoder") != std::string::npos) {
        std::string past_key_values_name = "past_key_values." + layer_suffix;
        std::string present_key_values_name = "present." + layer_suffix;
        assert(decoder_output_name_to_index.find(present_key_values_name) !=
               decoder_output_name_to_index.end());
        const int64_t present_key_values_index =
            decoder_output_name_to_index[present_key_values_name];

        delete past_key_values_by_name[past_key_values_name];
        past_key_values_by_name[past_key_values_name] = new MoonshineTensorView(
            ort_api, decoder_outputs[present_key_values_index],
            TENSOR_NAME(past_key_values_name));
      }
    }
    // Collect cross-attention weights if available (single-pass mode)
    if (alignment_session == nullptr) {
      // Check if decoder outputs include cross_attentions (modified decoder)
      for (int layer_idx = 0; layer_idx < num_layers; layer_idx++) {
        std::string attn_name = "cross_attentions." + std::to_string(layer_idx);
        if (decoder_output_name_to_index.find(attn_name) !=
            decoder_output_name_to_index.end()) {
          const int64_t attn_index = decoder_output_name_to_index[attn_name];
          MoonshineTensorView attn_view(ort_api, decoder_outputs[attn_index],
                                        "cross_attn");
          auto &attn_shape = attn_view.shape();
          // Shape: [1, heads, dec_step_len, enc_len]
          int heads = static_cast<int>(attn_shape[1]);
          int step_len = static_cast<int>(attn_shape[2]);
          int enc_len = static_cast<int>(attn_shape[3]);
          size_t step_size = heads * step_len * enc_len;
          size_t old_size = cross_attention_buffer.size();
          cross_attention_buffer.resize(old_size + step_size);
          memcpy(cross_attention_buffer.data() + old_size,
                 attn_view.data<float>(), step_size * sizeof(float));
          if (token_index == 0) {
            cross_attn_heads = heads;
            cross_attn_enc_len = enc_len;
          }
          cross_attn_steps++;
        }
      }
    }

    for (size_t i = 0; i < decoder_output_count; i++) {
      ort_api->ReleaseValue(decoder_outputs[i]);
    }

    int64_t next_token = logits_tensor_view.argmax();
    tokens.push_back(next_token);
    if (next_token == MOONSHINE_EOS_TOKEN_ID) {
      break;
    }
    inputIDs = {next_token};
  }

  for (const char *decoder_input_name : decoder_input_names) {
    ort_string_allocator->base.Free(&ort_string_allocator->base,
                                    (char *)decoder_input_name);
  }
  decoder_input_names.clear();

  for (const char *decoder_output_name : decoder_output_names) {
    ort_string_allocator->base.Free(&ort_string_allocator->base,
                                    (char *)decoder_output_name);
  }
  decoder_output_names.clear();

  delete last_hidden_state_tensor;
  last_hidden_state_tensor = nullptr;

  if (encoder_attention_mask_tensor != nullptr) {
    delete encoder_attention_mask_tensor;
    encoder_attention_mask_tensor = nullptr;
  }

  for (const auto &[key, value] : past_key_values_by_name) {
    delete value;
  }
  past_key_values_by_name.clear();

  // Save tokens and attention for word alignment (only when needed)
  if (!cross_attention_buffer.empty() || alignment_session != nullptr) {
    last_tokens = tokens;
  }

  // If we collected attention during single-pass decoding, save it
  if (!cross_attention_buffer.empty() && cross_attn_steps > 0) {
    last_cross_attention_buffer = std::move(cross_attention_buffer);
    last_cross_attn_heads = cross_attn_heads;
    last_cross_attn_enc_len = cross_attn_enc_len;
    last_cross_attn_steps = cross_attn_steps;
  }

  last_result = tokenizer->tokens_to_text(tokens);
  *out_text = (char *)(last_result.c_str());

  // TIMER_END(moonshine_transcribe);
  // log_leaked_tensor_views();
  return 0;
}

int MoonshineModel::transcribe_wav(const char *wav_path, char **out_text) {
  *out_text = nullptr;
  if (wav_path == nullptr) {
    LOG("WAV path is nullptr\n");
    return 1;
  }
  float *wav_data = nullptr;
  size_t wav_data_size = 0;
  if (!load_wav_data(wav_path, &wav_data, &wav_data_size)) {
    LOGF("Failed to load WAV file '%s'\n", wav_path);
    return 1;
  }
  return transcribe(wav_data, wav_data_size, out_text);
}

int MoonshineModel::load_alignment_model(const char *alignment_model_path) {
  const char *alignment_mmapped_data = nullptr;
  size_t alignment_mmapped_data_size = 0;
  RETURN_ON_ERROR(ort_session_from_path(
      ort_api, ort_env, ort_session_options, alignment_model_path,
      &alignment_session, &alignment_mmapped_data,
      &alignment_mmapped_data_size));
  RETURN_ON_NULL(alignment_session);
  return 0;
}

int MoonshineModel::compute_word_timestamps(
    float audio_duration, std::vector<TranscriberWord> &words_out) {
  words_out.clear();

  if (last_tokens.size() < 2) {
    return 1;
  }

  // Prefer single-pass attention (collected during decode) over alignment model
  if (!last_cross_attention_buffer.empty() && last_cross_attn_steps > 0) {
    std::vector<int> tokens_int(last_tokens.begin(), last_tokens.end());
    int total_steps = last_cross_attn_steps / num_layers;
    int H = last_cross_attn_heads;
    int E = last_cross_attn_enc_len;
    int L = num_layers;

    // The buffer was collected as: for each decode step, for each layer:
    //   [heads, 1, enc_len] → H*E floats
    // So layout is: [step0_L0, step0_L1, ..., step0_L5, step1_L0, ...]
    // We need: [L*H, total_steps, E] (layers*heads contiguous, then steps, then enc)
    // Rearrange from [total_steps, L, H, E] to [L, H, total_steps, E]
    size_t per_layer_step = H * E;
    std::vector<float> rearranged(L * H * total_steps * E);
    for (int s = 0; s < total_steps; s++) {
      for (int l = 0; l < L; l++) {
        // Source: step s, layer l → offset (s*L + l) * H * E
        // Dest: layer l, all heads, step s → offset (l * H * total_steps + s) * E
        // Actually for align_words: [L*H, total_steps, E]
        // So dest for layer l, head h, step s = ((l*H + h) * total_steps + s) * E
        const float *src =
            last_cross_attention_buffer.data() + (s * L + l) * per_layer_step;
        for (int h = 0; h < H; h++) {
          float *dst =
              rearranged.data() + ((l * H + h) * total_steps + s) * E;
          memcpy(dst, src + h * E, E * sizeof(float));
        }
      }
    }

    float time_per_frame = audio_duration / static_cast<float>(E);
    words_out = align_words(rearranged.data(), L, H, total_steps, E,
                            tokens_int, time_per_frame, tokenizer);
    last_cross_attention_buffer.clear();
    last_cross_attn_steps = 0;
    return 0;
  }

  // Fallback: use alignment model (two-pass)
  if (alignment_session == nullptr) {
    return 1;
  }
  if (last_encoder_hidden_states.empty()) {
    return 1;
  }

  // Prepare inputs for alignment model:
  //   input_ids: [1, num_tokens]
  //   encoder_hidden_states: [1, enc_frames, encoder_dim]

  int64_t num_tokens = static_cast<int64_t>(last_tokens.size());
  std::vector<int64_t> input_ids_shape = {1, num_tokens};
  std::vector<int64_t> enc_shape = {1, static_cast<int64_t>(last_encoder_frames),
                                    static_cast<int64_t>(encoder_dim)};

  OrtValue *input_ids_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, last_tokens.data(),
          last_tokens.size() * sizeof(int64_t), input_ids_shape.data(),
          input_ids_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
          &input_ids_tensor));

  OrtValue *enc_tensor = nullptr;
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          ort_memory_info, last_encoder_hidden_states.data(),
          last_encoder_hidden_states.size() * sizeof(float), enc_shape.data(),
          enc_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &enc_tensor));

  // Run alignment model
  // Outputs: logits, cross_attentions.0 .. cross_attentions.{num_layers-1}
  size_t align_output_count = 0;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->SessionGetOutputCount(
                                   alignment_session, &align_output_count));

  std::vector<const char *> input_names = {"input_ids", "encoder_hidden_states"};
  std::vector<char *> output_names_alloc(align_output_count);
  for (size_t i = 0; i < align_output_count; i++) {
    RETURN_ON_ORT_ERROR(ort_api,
                        ort_api->SessionGetOutputName(
                            alignment_session, i, &ort_string_allocator->base,
                            &output_names_alloc[i]));
  }
  std::vector<const char *> output_names(align_output_count);
  for (size_t i = 0; i < align_output_count; i++) {
    output_names[i] = output_names_alloc[i];
  }

  OrtValue *inputs[] = {input_ids_tensor, enc_tensor};
  std::vector<OrtValue *> outputs(align_output_count, nullptr);

  OrtStatus *status = ORT_RUN(ort_api, alignment_session, input_names.data(),
                              inputs, 2, output_names.data(),
                              align_output_count, outputs.data());

  ort_api->ReleaseValue(input_ids_tensor);
  ort_api->ReleaseValue(enc_tensor);

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api, status);
    for (auto *o : outputs) {
      if (o) ort_api->ReleaseValue(o);
    }
    for (auto *n : output_names_alloc) {
      ort_string_allocator->base.Free(&ort_string_allocator->base, n);
    }
    return 1;
  }

  // Extract cross-attention tensors (outputs[1] .. outputs[num_layers])
  // Each has shape [1, num_heads, num_tokens, enc_frames]
  int attn_layers = static_cast<int>(align_output_count) - 1;  // minus logits
  if (attn_layers <= 0) {
    for (auto *o : outputs) {
      if (o) ort_api->ReleaseValue(o);
    }
    for (auto *n : output_names_alloc) {
      ort_string_allocator->base.Free(&ort_string_allocator->base, n);
    }
    return 1;
  }

  // Get attention head count from first attention output shape
  OrtTensorTypeAndShapeInfo *attn_info = nullptr;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetTensorTypeAndShape(outputs[1], &attn_info));
  size_t attn_ndims = 0;
  RETURN_ON_ORT_ERROR(ort_api,
                      ort_api->GetDimensionsCount(attn_info, &attn_ndims));
  std::vector<int64_t> attn_shape(attn_ndims);
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->GetDimensions(attn_info, attn_shape.data(), attn_ndims));
  ort_api->ReleaseTensorTypeAndShapeInfo(attn_info);

  int num_heads = static_cast<int>(attn_shape[1]);
  int dec_len = static_cast<int>(attn_shape[2]);
  int enc_len = static_cast<int>(attn_shape[3]);

  // Stack all attention layers into a flat buffer:
  // [attn_layers * num_heads, dec_len, enc_len]
  size_t per_layer = num_heads * dec_len * enc_len;
  std::vector<float> cross_attention_data(attn_layers * per_layer);

  for (int layer = 0; layer < attn_layers; layer++) {
    float *layer_data = nullptr;
    RETURN_ON_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(
                                     outputs[1 + layer], (void **)&layer_data));
    // Copy from [1, heads, dec_len, enc_len] to flat buffer
    memcpy(cross_attention_data.data() + layer * per_layer, layer_data,
           per_layer * sizeof(float));
  }

  // Release ONNX outputs
  for (auto *o : outputs) {
    if (o) ort_api->ReleaseValue(o);
  }
  for (auto *n : output_names_alloc) {
    ort_string_allocator->base.Free(&ort_string_allocator->base, n);
  }

  // Convert tokens to int vector for align_words
  std::vector<int> tokens_int(last_tokens.begin(), last_tokens.end());

  float time_per_frame = audio_duration / static_cast<float>(enc_len);

  words_out = align_words(cross_attention_data.data(), attn_layers, num_heads,
                          dec_len, enc_len, tokens_int, time_per_frame,
                          tokenizer);

  return 0;
}
