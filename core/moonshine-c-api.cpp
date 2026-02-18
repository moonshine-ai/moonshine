/*
MIT License

Copyright (c) 2025 Moonshine AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "moonshine-c-api.h"

#include <fcntl.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cctype>
#include <cerrno>
#include <cerrno>  // For errno
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstring>  // For strerror
#include <map>
#include <mutex>
#include <numeric>
#include <vector>

#include "bin-tokenizer.h"
#include "debug-utils.h"
#include "gemma-embedding-model.h"
#include "intent-recognizer.h"
#include "moonshine-model.h"
#include "moonshine-ort-allocator.h"
#include "moonshine-tensor-view.h"
#include "ort-utils.h"
#include "string-utils.h"
#include "transcriber.h"

// Defined as a macro to ensure we get meaningful line numbers in the error
// message.
#define CHECK_TRANSCRIBER_HANDLE(handle)                                  \
  do {                                                                    \
    if (handle < 0 || !transcriber_map.contains(handle)) {                \
      LOGF("Moonshine transcriber handle is invalid: handle %d", handle); \
      return MOONSHINE_ERROR_INVALID_HANDLE;                              \
    }                                                                     \
  } while (0)

namespace {

bool log_api_calls = false;

void parse_transcriber_options(const transcriber_option_t *in_options,
                               uint64_t in_options_count,
                               TranscriberOptions &out_options) {
  for (uint64_t i = 0; i < in_options_count; i++) {
    const transcriber_option_t &in_option = in_options[i];
    std::string option_name = to_lowercase(in_option.name);
    if (option_name == "skip_transcription") {
      out_options.model_source = TranscriberOptions::ModelSource::NONE;
    } else if (option_name == "transcription_interval") {
      out_options.transcription_interval = float_from_string(in_option.value);
    } else if (option_name == "vad_threshold") {
      out_options.vad_threshold = float_from_string(in_option.value);
    } else if (option_name == "save_input_wav_path") {
      out_options.save_input_wav_path = std::string(in_option.value);
    } else if (option_name == "log_api_calls") {
      log_api_calls = bool_from_string(in_option.value);
    } else if (option_name == "log_ort_run") {
      out_options.log_ort_run = bool_from_string(in_option.value);
    } else if (option_name == "vad_window_duration") {
      out_options.vad_window_duration = float_from_string(in_option.value);
    } else if (option_name == "vad_hop_size") {
      out_options.vad_hop_size = int32_from_string(in_option.value);
    } else if (option_name == "vad_look_behind_sample_count") {
      out_options.vad_look_behind_sample_count = size_t_from_string(in_option.value);
    } else if (option_name == "vad_max_segment_duration") {
      out_options.vad_max_segment_duration = float_from_string(in_option.value);
    } else if (option_name == "max_tokens_per_second") {
      out_options.max_tokens_per_second = float_from_string(in_option.value);
    } else if (option_name == "identify_speakers") {
      out_options.identify_speakers = bool_from_string(in_option.value);
    } else if (option_name == "speaker_id_cluster_threshold") {
      out_options.speaker_id_cluster_threshold = float_from_string(in_option.value);
    } else if (option_name == "return_audio_data") {
      out_options.return_audio_data = bool_from_string(in_option.value);
    } else {
      throw std::runtime_error("Unknown transcriber option: '" +
                               std::string(in_option.name) + "'");
    }
  }
}

std::mutex transcriber_map_mutex;
std::map<int32_t, Transcriber *> transcriber_map;
int32_t next_transcriber_handle = 0;

int32_t allocate_transcriber_handle(Transcriber *transcriber) {
  std::lock_guard<std::mutex> lock(transcriber_map_mutex);
  int32_t transcriber_handle = next_transcriber_handle++;
  transcriber_map[transcriber_handle] = transcriber;
  return transcriber_handle;
}

void free_transcriber_handle(int32_t handle) {
  std::lock_guard<std::mutex> lock(transcriber_map_mutex);
  delete transcriber_map[handle];
  transcriber_map[handle] = nullptr;
  transcriber_map.erase(handle);
}

}  // namespace

extern "C" int32_t moonshine_get_version(void) {
  if (log_api_calls) {
    LOG("moonshine_get_version");
  }
  return MOONSHINE_HEADER_VERSION;
}

/* Converts an error code number returned from an API call into a
   human-readable string. */
extern "C" const char *moonshine_error_to_string(int32_t error) {
  if (error == MOONSHINE_ERROR_NONE) {
    return "Success";
  }
  if (error == MOONSHINE_ERROR_INVALID_HANDLE) {
    return "Invalid handle";
  }
  if (error == MOONSHINE_ERROR_INVALID_ARGUMENT) {
    return "Invalid argument";
  }
  return "Unknown error";
}

extern "C" int32_t moonshine_load_transcriber_from_files(
    const char *path, uint32_t model_arch, const transcriber_option_t *options,
    uint64_t options_count, int32_t moonshine_version) {
  if (log_api_calls) {
    LOGF(
        "moonshine_load_transcriber_from_files(path=%s, model_arch=%d, "
        "options=%p, options_count=%" PRIu64 ", moonshine_version=%d)",
        path, model_arch, (void *)(options), options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const transcriber_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  Transcriber *transcriber = nullptr;
  try {
    TranscriberOptions transcriber_options;
    transcriber_options.model_source = TranscriberOptions::ModelSource::FILES;
    transcriber_options.model_path = path;
    transcriber_options.model_arch = model_arch;
    parse_transcriber_options(options, options_count, transcriber_options);
    transcriber = new Transcriber(transcriber_options);
  } catch (const std::exception &e) {
    LOGF("Failed to load transcriber: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  int32_t transcriber_handle = allocate_transcriber_handle(transcriber);
  return transcriber_handle;
}

int32_t moonshine_load_transcriber_from_memory(
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *decoder_model_data, size_t decoder_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    uint32_t model_arch, const transcriber_option_t *options,
    uint64_t options_count, int32_t moonshine_version) {
  if (log_api_calls) {
    LOGF(
        "moonshine_load_transcriber_from_memory(encoder_model_data=%p, "
        "encoder_model_data_size=%zu, decoder_model_data=%p, "
        "decoder_model_data_size=%zu, tokenizer_data=%p, "
        "tokenizer_data_size=%zu, model_arch=%d, options=%p, "
        "options_count=%" PRIu64 ", moonshine_version=%d)",
        (void *)(encoder_model_data), encoder_model_data_size,
        (void *)(decoder_model_data), decoder_model_data_size,
        (void *)(tokenizer_data), tokenizer_data_size, model_arch,
        (void *)(options), options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const transcriber_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }

  Transcriber *transcriber = nullptr;
  try {
    TranscriberOptions transcriber_options;
    transcriber_options.model_source = TranscriberOptions::ModelSource::MEMORY;
    transcriber_options.encoder_model_data = encoder_model_data;
    transcriber_options.encoder_model_data_size = encoder_model_data_size;
    transcriber_options.decoder_model_data = decoder_model_data;
    transcriber_options.decoder_model_data_size = decoder_model_data_size;
    transcriber_options.tokenizer_data = tokenizer_data;
    transcriber_options.tokenizer_data_size = tokenizer_data_size;
    transcriber_options.model_arch = model_arch;
    parse_transcriber_options(options, options_count, transcriber_options);
    transcriber = new Transcriber(transcriber_options);
  } catch (const std::exception &e) {
    LOGF("Failed to load transcriber: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  int32_t transcriber_handle = allocate_transcriber_handle(transcriber);
  return transcriber_handle;
}

void moonshine_free_transcriber(int32_t transcriber_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_transcriber(transcriber_handle=%d)",
         transcriber_handle);
  }
  free_transcriber_handle(transcriber_handle);
}

int32_t moonshine_transcribe_without_streaming(
    int32_t transcriber_handle, float *audio_data, uint64_t audio_length,
    int32_t sample_rate, uint32_t flags, struct transcript_t **out_transcript) {
  if (log_api_calls) {
    LOGF(
        "moonshine_transcribe_without_streaming(transcriber_handle=%d, "
        "audio_data=%p, audio_length=%" PRIu64
        ", sample_rate=%d, flags=%d, "
        "out_transcript=%p)",
        transcriber_handle, (void *)(audio_data), audio_length, sample_rate,
        flags, (void *)(out_transcript));
  }
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->transcribe_without_streaming(
        audio_data, audio_length, sample_rate, flags, out_transcript);
  } catch (const std::exception &e) {
    LOGF("Failed to transcribe without streaming: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

int32_t moonshine_create_stream(int32_t transcriber_handle, uint32_t flags) {
  if (log_api_calls) {
    LOGF("moonshine_create_stream(transcriber_handle=%d, flags=%d)",
         transcriber_handle, flags);
  }
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    return transcriber_map[transcriber_handle]->create_stream();
  } catch (const std::exception &e) {
    LOGF("Failed to create stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int moonshine_free_stream(int32_t transcriber_handle, int32_t stream_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_stream(transcriber_handle=%d, stream_handle=%d)",
         transcriber_handle, stream_handle);
  }
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->free_stream(stream_handle);
  } catch (const std::exception &e) {
    LOGF("Failed to free stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

int32_t moonshine_start_stream(int32_t transcriber_handle,
                               int32_t stream_handle) {
  if (log_api_calls) {
    LOGF("moonshine_start_stream(transcriber_handle=%d, stream_handle=%d)",
         transcriber_handle, stream_handle);
  }
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->start_stream(stream_handle);
  } catch (const std::exception &e) {
    LOGF("Failed to start stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

int32_t moonshine_stop_stream(int32_t transcriber_handle,
                              int32_t stream_handle) {
  if (log_api_calls) {
    LOGF("moonshine_stop_stream(transcriber_handle=%d, stream_handle=%d)",
         transcriber_handle, stream_handle);
  }
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->stop_stream(stream_handle);
  } catch (const std::exception &e) {
    LOGF("Failed to stop stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

const char *moonshine_transcript_to_string(
    const struct transcript_t *transcript) {
  if (log_api_calls) {
    LOGF("moonshine_transcript_to_string(transcript=%p)", (void *)(transcript));
  }
  static std::string description;
  description = Transcriber::transcript_to_string(transcript);
  return description.c_str();
}

int32_t moonshine_transcribe_add_audio_to_stream(int32_t transcriber_handle,
                                                 int32_t stream_handle,
                                                 const float *new_audio_data,
                                                 uint64_t audio_length,
                                                 int32_t sample_rate,
                                                 uint32_t flags) {
  if (log_api_calls) {
    LOGF(
        "moonshine_transcribe_add_audio_to_stream(transcriber_handle=%d, "
        "stream_handle=%d, new_audio_data=%p, audio_length=%" PRIu64
        ", "
        "sample_rate=%d, flags=%d)",
        transcriber_handle, stream_handle, (void *)(new_audio_data),
        audio_length, sample_rate, flags);
  }
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->add_audio_to_stream(
        stream_handle, new_audio_data, audio_length, sample_rate);
  } catch (const std::exception &e) {
    LOGF("Failed to add audio to stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

int32_t moonshine_transcribe_stream(int32_t transcriber_handle,
                                    int32_t stream_handle, uint32_t flags,
                                    struct transcript_t **out_transcript) {
  if (log_api_calls) {
    LOGF(
        "moonshine_transcribe_stream(transcriber_handle=%d, stream_handle=%d, "
        "flags=%d, out_transcript=%p)",
        transcriber_handle, stream_handle, flags, (void *)(out_transcript));
  }
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->transcribe_stream(stream_handle, flags,
                                                           out_transcript);
  } catch (const std::exception &e) {
    LOGF("Failed to transcribe stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

/* ------------------------------ INTENT RECOGNIZER ------------------------- */

namespace {

// Structure to hold callback info for C API
struct IntentCallbackInfo {
  moonshine_intent_callback callback;
  void *user_data;
  std::string trigger_phrase;
};

std::mutex intent_recognizer_map_mutex;
std::map<int32_t, IntentRecognizer *> intent_recognizer_map;
std::map<int32_t, std::vector<IntentCallbackInfo>> intent_callback_map;
int32_t next_intent_recognizer_handle = 0;

int32_t allocate_intent_recognizer_handle(IntentRecognizer *recognizer) {
  std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
  int32_t handle = next_intent_recognizer_handle++;
  intent_recognizer_map[handle] = recognizer;
  intent_callback_map[handle] = std::vector<IntentCallbackInfo>();
  return handle;
}

void free_intent_recognizer_handle(int32_t handle) {
  // Note: Caller must hold intent_recognizer_map_mutex
  delete intent_recognizer_map[handle];
  intent_recognizer_map[handle] = nullptr;
  intent_recognizer_map.erase(handle);
  intent_callback_map.erase(handle);
}

#define CHECK_INTENT_RECOGNIZER_HANDLE(handle)                                  \
  do {                                                                          \
    if (handle < 0 || !intent_recognizer_map.contains(handle)) {                \
      LOGF("Moonshine intent recognizer handle is invalid: handle %d", handle); \
      return MOONSHINE_ERROR_INVALID_HANDLE;                                    \
    }                                                                           \
  } while (0)

}  // namespace

int32_t moonshine_create_intent_recognizer(const char *model_path,
                                           uint32_t model_arch,
                                           const char *model_variant,
                                           float threshold) {
  if (log_api_calls) {
    LOGF("moonshine_create_intent_recognizer(model_path=%s, model_arch=%d, "
         "model_variant=%s, threshold=%f)",
         model_path, model_arch, model_variant ? model_variant : "q4",
         threshold);
  }

  if (model_path == nullptr) {
    LOGF("%s", "Invalid model_path: nullptr");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }

  IntentRecognizer *recognizer = nullptr;
  try {
    IntentRecognizerOptions options;
    options.model_path = model_path;
    options.model_arch = static_cast<EmbeddingModelArch>(model_arch);
    options.model_variant = model_variant ? model_variant : "q4";
    options.threshold = threshold;

    recognizer = new IntentRecognizer(options);
  } catch (const std::exception &e) {
    delete recognizer;
    LOGF("Failed to create intent recognizer: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return allocate_intent_recognizer_handle(recognizer);
}

void moonshine_free_intent_recognizer(int32_t intent_recognizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_intent_recognizer(handle=%d)", intent_recognizer_handle);
  }
  std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
  if (intent_recognizer_map.contains(intent_recognizer_handle)) {
    free_intent_recognizer_handle(intent_recognizer_handle);
  }
}

int32_t moonshine_register_intent(int32_t intent_recognizer_handle,
                                  const char *trigger_phrase,
                                  moonshine_intent_callback callback,
                                  void *user_data) {
  if (log_api_calls) {
    LOGF("moonshine_register_intent(handle=%d, trigger_phrase=%s)",
         intent_recognizer_handle, trigger_phrase);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  try {
    // Store callback info
    IntentCallbackInfo info;
    info.callback = callback;
    info.user_data = user_data;
    info.trigger_phrase = trigger_phrase;

    {
      std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
      intent_callback_map[intent_recognizer_handle].push_back(info);
    }

    // Register with the C++ IntentRecognizer using a lambda that invokes the C
    // callback. We capture a std::string copy of trigger_phrase since the
    // original pointer may become invalid after this function returns.
    std::string trigger_copy = trigger_phrase;
    intent_recognizer_map[intent_recognizer_handle]->register_intent(
        trigger_phrase,
        [callback, user_data, trigger_copy](const std::string &utterance,
                                            float similarity) {
          if (callback) {
            callback(user_data, trigger_copy.c_str(), utterance.c_str(),
                     similarity);
          }
        });
  } catch (const std::exception &e) {
    LOGF("Failed to register intent: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

int32_t moonshine_unregister_intent(int32_t intent_recognizer_handle,
                                    const char *trigger_phrase) {
  if (log_api_calls) {
    LOGF("moonshine_unregister_intent(handle=%d, trigger_phrase=%s)",
         intent_recognizer_handle, trigger_phrase);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  try {
    bool result = intent_recognizer_map[intent_recognizer_handle]
                      ->unregister_intent(trigger_phrase);
    if (!result) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    // Remove from callback map
    {
      std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
      auto &callbacks = intent_callback_map[intent_recognizer_handle];
      callbacks.erase(
          std::remove_if(callbacks.begin(), callbacks.end(),
                         [trigger_phrase](const IntentCallbackInfo &info) {
                           return info.trigger_phrase == trigger_phrase;
                         }),
          callbacks.end());
    }
  } catch (const std::exception &e) {
    LOGF("Failed to unregister intent: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

int32_t moonshine_process_utterance(int32_t intent_recognizer_handle,
                                    const char *utterance) {
  if (log_api_calls) {
    LOGF("moonshine_process_utterance(handle=%d, utterance=%s)",
         intent_recognizer_handle, utterance);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  try {
    bool result = intent_recognizer_map[intent_recognizer_handle]
                      ->process_utterance(utterance);
    return result ? 1 : 0;
  } catch (const std::exception &e) {
    LOGF("Failed to process utterance: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int32_t moonshine_set_intent_threshold(int32_t intent_recognizer_handle,
                                       float threshold) {
  if (log_api_calls) {
    LOGF("moonshine_set_intent_threshold(handle=%d, threshold=%f)",
         intent_recognizer_handle, threshold);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  try {
    intent_recognizer_map[intent_recognizer_handle]->set_threshold(threshold);
  } catch (const std::exception &e) {
    LOGF("Failed to set intent threshold: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

float moonshine_get_intent_threshold(int32_t intent_recognizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_get_intent_threshold(handle=%d)", intent_recognizer_handle);
  }
  std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
  if (!intent_recognizer_map.contains(intent_recognizer_handle)) {
    return -1.0f;
  }
  return intent_recognizer_map[intent_recognizer_handle]->get_threshold();
}

int32_t moonshine_get_intent_count(int32_t intent_recognizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_get_intent_count(handle=%d)", intent_recognizer_handle);
  }
  std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
  if (!intent_recognizer_map.contains(intent_recognizer_handle)) {
    return MOONSHINE_ERROR_INVALID_HANDLE;
  }
  return static_cast<int32_t>(
      intent_recognizer_map[intent_recognizer_handle]->get_intent_count());
}

int32_t moonshine_clear_intents(int32_t intent_recognizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_clear_intents(handle=%d)", intent_recognizer_handle);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  try {
    intent_recognizer_map[intent_recognizer_handle]->clear_intents();
    {
      std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
      intent_callback_map[intent_recognizer_handle].clear();
    }
  } catch (const std::exception &e) {
    LOGF("Failed to clear intents: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}