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
#include <filesystem>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bin-tokenizer.h"
#include "debug-utils.h"
#include "intent-recognizer.h"
#include "moonshine-asset-catalog.h"
#include "moonshine-g2p.h"
#include "moonshine-model.h"
#include "moonshine-ort-allocator.h"
#include "moonshine-tensor-view.h"
#include "moonshine-tts.h"
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

typedef std::pair<std::string, std::string> OptionPair;
typedef std::vector<OptionPair> OptionVector;

OptionVector parse_option_vector(const moonshine_option_t *options,
                                 uint64_t options_count) {
  OptionVector option_vector;
  option_vector.reserve(options_count);
  for (uint64_t i = 0; i < options_count; i++) {
    const moonshine_option_t &option = options[i];
    std::string option_name = to_lowercase(option.name);
    option_vector.emplace_back(option_name, option.value);
  }
  return option_vector;
}

bool log_api_calls = false;

// Handles common options that are not specific to any particular API.
OptionVector parse_common_options(const OptionVector &options) {
  OptionVector uncommon_options;
  for (const auto &option : options) {
    if (option.first == "log_api_calls") {
      log_api_calls = bool_from_string(option.second);
    } else {
      uncommon_options.push_back(option);
    }
  }
  return uncommon_options;
}

void parse_transcriber_options(const OptionVector &options,
                               TranscriberOptions &out_options) {
  for (const auto &option : options) {
    const std::string &option_name = option.first;
    const std::string &option_value = option.second;
    if (option_name == "skip_transcription") {
      out_options.model_source = TranscriberOptions::ModelSource::NONE;
    } else if (option_name == "transcription_interval") {
      out_options.transcription_interval = float_from_string(option_value);
    } else if (option.first == "vad_threshold") {
      out_options.vad_threshold = float_from_string(option_value);
    } else if (option_name == "save_input_wav_path") {
      out_options.save_input_wav_path = std::string(option_value);
    } else if (option_name == "log_api_calls") {
      log_api_calls = bool_from_string(option_value);
    } else if (option_name == "log_ort_run") {
      out_options.log_ort_run = bool_from_string(option_value);
    } else if (option_name == "vad_window_duration") {
      out_options.vad_window_duration = float_from_string(option_value);
    } else if (option_name == "vad_hop_size") {
      out_options.vad_hop_size = int32_from_string(option_value);
    } else if (option_name == "vad_look_behind_sample_count") {
      out_options.vad_look_behind_sample_count =
          size_t_from_string(option.second);
    } else if (option_name == "vad_max_segment_duration") {
      out_options.vad_max_segment_duration = float_from_string(option_value);
    } else if (option_name == "max_tokens_per_second") {
      out_options.max_tokens_per_second = float_from_string(option_value);
    } else if (option_name == "identify_speakers") {
      out_options.identify_speakers = bool_from_string(option_value);
    } else if (option_name == "speaker_id_cluster_threshold") {
      out_options.speaker_id_cluster_threshold =
          float_from_string(option_value);
    } else if (option_name == "return_audio_data") {
      out_options.return_audio_data = bool_from_string(option_value);
    } else if (option_name == "log_output_text") {
      out_options.log_output_text = bool_from_string(option_value);
    } else if (option_name == "word_timestamps") {
      out_options.word_timestamps = bool_from_string(option_value);
    } else {
      throw std::runtime_error("Unknown transcriber option: '" + option_name +
                               "', value=" + option_value);
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
    const char *path, uint32_t model_arch, const moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version) {
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (log_api_calls) {
    LOGF(
        "moonshine_load_transcriber_from_files(path=%s, model_arch=%d, "
        "options=%p, options_count=%" PRIu64 ", moonshine_version=%d)",
        path, model_arch, (void *)(options), options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  Transcriber *transcriber = nullptr;
  try {
    TranscriberOptions transcriber_options;
    transcriber_options.model_source = TranscriberOptions::ModelSource::FILES;
    transcriber_options.model_path = path;
    transcriber_options.model_arch = model_arch;
    parse_transcriber_options(uncommon_options, transcriber_options);
    transcriber = new Transcriber(transcriber_options);
  } catch (const std::exception &e) {
    LOGF("Failed to load transcriber: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  int32_t transcriber_handle = allocate_transcriber_handle(transcriber);
  return transcriber_handle;
}

extern "C" int32_t moonshine_load_transcriber_from_memory(
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *decoder_model_data, size_t decoder_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    uint32_t model_arch, const moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version) {
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
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
      const moonshine_option_t &option = options[i];
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
    parse_transcriber_options(uncommon_options, transcriber_options);
    transcriber = new Transcriber(transcriber_options);
  } catch (const std::exception &e) {
    LOGF("Failed to load transcriber: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  int32_t transcriber_handle = allocate_transcriber_handle(transcriber);
  return transcriber_handle;
}

extern "C" void moonshine_free_transcriber(int32_t transcriber_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_transcriber(transcriber_handle=%d)",
         transcriber_handle);
  }
  free_transcriber_handle(transcriber_handle);
}

extern "C" int32_t moonshine_transcribe_without_streaming(
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

extern "C" int32_t moonshine_create_stream(int32_t transcriber_handle, uint32_t flags) {
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

extern "C" int32_t moonshine_free_stream(int32_t transcriber_handle, int32_t stream_handle) {
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

extern "C" int32_t moonshine_start_stream(int32_t transcriber_handle,
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

extern "C" int32_t moonshine_stop_stream(int32_t transcriber_handle,
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

extern "C" const char *moonshine_transcript_to_string(
    const struct transcript_t *transcript) {
  if (log_api_calls) {
    LOGF("moonshine_transcript_to_string(transcript=%p)", (void *)(transcript));
  }
  static std::string description;
  description = Transcriber::transcript_to_string(transcript);
  return description.c_str();
}

extern "C" int32_t moonshine_transcribe_add_audio_to_stream(int32_t transcriber_handle,
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

extern "C" int32_t moonshine_transcribe_stream(int32_t transcriber_handle,
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

std::mutex intent_recognizer_map_mutex;
std::map<int32_t, IntentRecognizer *> intent_recognizer_map;
int32_t next_intent_recognizer_handle = 0;

int32_t allocate_intent_recognizer_handle(IntentRecognizer *recognizer) {
  std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
  int32_t handle = next_intent_recognizer_handle++;
  intent_recognizer_map[handle] = recognizer;
  return handle;
}

void free_intent_recognizer_handle(int32_t handle) {
  // Note: Caller must hold intent_recognizer_map_mutex
  delete intent_recognizer_map[handle];
  intent_recognizer_map[handle] = nullptr;
  intent_recognizer_map.erase(handle);
}

#define CHECK_INTENT_RECOGNIZER_HANDLE(handle)                         \
  do {                                                                 \
    if (handle < 0 || !intent_recognizer_map.contains(handle)) {       \
      LOGF("Moonshine intent recognizer handle is invalid: handle %d", \
           handle);                                                    \
      return MOONSHINE_ERROR_INVALID_HANDLE;                           \
    }                                                                  \
  } while (0)

char *duplicate_c_string(const char *s) {
  if (s == nullptr) {
    return nullptr;
  }
  std::size_t n = std::strlen(s) + 1;
  char *out = static_cast<char *>(std::malloc(n));
  if (out != nullptr) {
    std::memcpy(out, s, n);
  }
  return out;
}

}  // namespace

extern "C" int32_t moonshine_create_intent_recognizer(const char *model_path,
                                           uint32_t model_arch,
                                           const char *model_variant) {
  if (log_api_calls) {
    LOGF(
        "moonshine_create_intent_recognizer(model_path=%s, model_arch=%d, "
        "model_variant=%s)",
        model_path, model_arch, model_variant ? model_variant : "q4");
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

    recognizer = new IntentRecognizer(options);
  } catch (const std::exception &e) {
    delete recognizer;
    LOGF("Failed to create intent recognizer: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return allocate_intent_recognizer_handle(recognizer);
}

extern "C" void moonshine_free_intent_recognizer(int32_t intent_recognizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_intent_recognizer(handle=%d)",
         intent_recognizer_handle);
  }
  std::lock_guard<std::mutex> lock(intent_recognizer_map_mutex);
  if (intent_recognizer_map.contains(intent_recognizer_handle)) {
    free_intent_recognizer_handle(intent_recognizer_handle);
  }
}

extern "C" int32_t moonshine_register_intent(int32_t intent_recognizer_handle,
                                  const char *canonical_phrase,
                                  float *embedding, uint64_t embedding_size,
                                  int32_t priority) {
  if (log_api_calls) {
    LOGF(
        "moonshine_register_intent(handle=%d, canonical_phrase=%s, "
        "embedding=%p, embedding_size=%" PRIu64 ", priority=%d)",
        intent_recognizer_handle, canonical_phrase,
        static_cast<void *>(embedding), embedding_size, priority);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  if (canonical_phrase == nullptr) {
    LOGF("%s", "moonshine_register_intent: canonical_phrase is nullptr");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  try {
    intent_recognizer_map[intent_recognizer_handle]->register_intent(
        canonical_phrase, embedding, embedding_size, priority);
  } catch (const std::exception &e) {
    LOGF("Failed to register intent: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

extern "C" int32_t moonshine_unregister_intent(int32_t intent_recognizer_handle,
                                    const char *canonical_phrase) {
  if (log_api_calls) {
    LOGF("moonshine_unregister_intent(handle=%d, canonical_phrase=%s)",
         intent_recognizer_handle, canonical_phrase);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  if (canonical_phrase == nullptr) {
    LOGF("%s", "moonshine_unregister_intent: canonical_phrase is nullptr");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  try {
    bool result =
        intent_recognizer_map[intent_recognizer_handle]->unregister_intent(
            canonical_phrase);
    if (!result) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
  } catch (const std::exception &e) {
    LOGF("Failed to unregister intent: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

extern "C" int32_t moonshine_get_closest_intents(int32_t intent_recognizer_handle,
                                      const char *utterance,
                                      float tolerance_threshold,
                                      moonshine_intent_match_t **out_matches,
                                      uint64_t *out_count) {
  if (log_api_calls) {
    LOGF("moonshine_get_closest_intents(handle=%d, utterance=%s, tolerance=%f)",
         intent_recognizer_handle, utterance ? utterance : "(null)",
         tolerance_threshold);
  }
  if (out_matches == nullptr || out_count == nullptr) {
    LOGF("%s",
         "moonshine_get_closest_intents: out_matches or out_count is nullptr");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_matches = nullptr;
  *out_count = 0;

  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  if (utterance == nullptr) {
    LOGF("%s", "moonshine_get_closest_intents: utterance is nullptr");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }

  try {
    auto ranked = intent_recognizer_map[intent_recognizer_handle]->rank_intents(
        utterance, tolerance_threshold,
        static_cast<size_t>(MOONSHINE_INTENT_MAX_MATCHES));
    if (ranked.empty()) {
      return MOONSHINE_ERROR_NONE;
    }

    const uint64_t n = static_cast<uint64_t>(ranked.size());
    auto *arr = static_cast<moonshine_intent_match_t *>(
        std::malloc(n * sizeof(moonshine_intent_match_t)));
    if (arr == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    for (uint64_t i = 0; i < n; ++i) {
      arr[i].canonical_phrase = duplicate_c_string(ranked[i].first.c_str());
      arr[i].similarity = ranked[i].second;
      if (arr[i].canonical_phrase == nullptr) {
        for (uint64_t j = 0; j < i; ++j) {
          std::free(arr[j].canonical_phrase);
        }
        std::free(arr);
        return MOONSHINE_ERROR_UNKNOWN;
      }
    }
    *out_matches = arr;
    *out_count = n;
  } catch (const std::exception &e) {
    LOGF("Failed to get closest intents: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

extern "C" void moonshine_free_intent_matches(moonshine_intent_match_t *matches,
                                   uint64_t count) {
  if (matches == nullptr) {
    return;
  }
  for (uint64_t i = 0; i < count; ++i) {
    std::free(matches[i].canonical_phrase);
  }
  std::free(matches);
}

extern "C" int32_t moonshine_get_intent_count(int32_t intent_recognizer_handle) {
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

extern "C" int32_t moonshine_clear_intents(int32_t intent_recognizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_clear_intents(handle=%d)", intent_recognizer_handle);
  }
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  try {
    intent_recognizer_map[intent_recognizer_handle]->clear_intents();
  } catch (const std::exception &e) {
    LOGF("Failed to clear intents: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

extern "C" int32_t moonshine_calculate_intent_embedding(
    int32_t intent_recognizer_handle, const char *sentence,
    float **out_embedding, uint64_t *out_embedding_size,
    const char *model_name) {
  (void)model_name;
  if (log_api_calls) {
    LOGF(
        "moonshine_calculate_intent_embedding(handle=%d, sentence=%s, "
        "out_embedding=%p, out_embedding_size=%p, model_name=%s)",
        intent_recognizer_handle, sentence ? sentence : "(null)",
        static_cast<void *>(out_embedding),
        static_cast<void *>(out_embedding_size),
        model_name ? model_name : "(null)");
  }
  if (sentence == nullptr || out_embedding == nullptr ||
      out_embedding_size == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_embedding = nullptr;
  *out_embedding_size = 0;
  CHECK_INTENT_RECOGNIZER_HANDLE(intent_recognizer_handle);
  try {
    std::vector<float> emb =
        intent_recognizer_map[intent_recognizer_handle]->calculate_embedding(
            sentence);
    const uint64_t n = static_cast<uint64_t>(emb.size());
    auto *buf = static_cast<float *>(std::malloc(n * sizeof(float)));
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    std::memcpy(buf, emb.data(), n * sizeof(float));
    *out_embedding = buf;
    *out_embedding_size = n;
  } catch (const std::exception &e) {
    LOGF("Failed to calculate intent embedding: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

extern "C" void moonshine_free_intent_embedding(float *embedding) {
  std::free(embedding);
}

/* ------------------------------ TEXT TO SPEECH ------------------------- */

namespace {

std::mutex text_to_speech_synthesizer_map_mutex;
std::map<int32_t, moonshine_tts::MoonshineTTS *> text_to_speech_synthesizer_map;
int32_t next_text_to_speech_synthesizer_handle = 0;

int32_t allocate_text_to_speech_synthesizer_handle(
    moonshine_tts::MoonshineTTS *synthesizer) {
  std::lock_guard<std::mutex> lock(text_to_speech_synthesizer_map_mutex);
  int32_t handle = next_text_to_speech_synthesizer_handle++;
  text_to_speech_synthesizer_map[handle] = synthesizer;
  return handle;
}

void parse_tts_options(const OptionVector &options,
                       moonshine_tts::MoonshineTTSOptions &out_options,
                       std::string &cli_language_out,
                       bool &language_was_set_out) {
  language_was_set_out = false;
  out_options.parse_options(options, &cli_language_out, &language_was_set_out);
}

#define CHECK_TTS_SYNTHESIZER_HANDLE(synth_handle)                             \
  do {                                                                         \
    if ((synth_handle) < 0 ||                                                  \
        !text_to_speech_synthesizer_map.contains((synth_handle))) {            \
      LOGF(                                                                    \
          "Moonshine text to speech synthesizer handle is invalid: handle %d", \
          (int)(synth_handle));                                                \
      return MOONSHINE_ERROR_INVALID_HANDLE;                                   \
    }                                                                          \
  } while (0)

}  // namespace

extern "C" int32_t moonshine_create_tts_synthesizer_from_files(
    const char *language, const char **filenames, uint64_t filenames_count,
    const struct moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version) {
  (void)filenames;
  (void)filenames_count;
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (log_api_calls) {
    LOGF(
        "moonshine_create_tts_synthesizer_from_files(language=%s, "
        "filenames=%p, filenames_count=%" PRIu64 ", options=%p, options_count=%" PRIu64 ", "
        "moonshine_version=%d)",
        language, reinterpret_cast<const void *>(filenames),
        filenames_count,
        static_cast<const void *>(options),
        options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  moonshine_tts::MoonshineTTSOptions tts_options;
  std::string lang_from_options;
  bool lang_from_options_set = false;
  parse_tts_options(uncommon_options, tts_options, lang_from_options,
                    lang_from_options_set);
  std::string lang = (language != nullptr && language[0] != '\0')
                         ? std::string(language)
                         : std::string("en_us");
  if (lang_from_options_set) {
    lang = std::move(lang_from_options);
  }
  try {
    auto *synthesizer = new moonshine_tts::MoonshineTTS(lang, tts_options);
    return allocate_text_to_speech_synthesizer_handle(synthesizer);
  } catch (const std::exception &e) {
    LOGF("Failed to create TTS synthesizer: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int32_t moonshine_create_tts_synthesizer_from_memory(
    const char *language, const char **filenames, uint64_t filenames_count,
    const uint8_t **memory, const uint64_t *memory_sizes,
    const struct moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version) {
  (void)moonshine_version;
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (filenames_count > 0) {
    if (filenames == nullptr || memory == nullptr || memory_sizes == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
  }
  if (log_api_calls) {
    LOGF(
        "moonshine_create_tts_synthesizer_from_memory(language=%s, "
        "filenames=%p, "
        "filenames_count=%" PRIu64 ", memory=%p, memory_sizes=%p, options=%p, "
        "options_count=%" PRIu64 ", moonshine_version=%d)",
        language, reinterpret_cast<const void *>(filenames),
        filenames_count,
        reinterpret_cast<const void *>(memory),
        reinterpret_cast<const void *>(memory_sizes),
        static_cast<const void *>(options),
        options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  try {
    moonshine_tts::MoonshineTTSOptions tts_options;
    for (uint64_t i = 0; i < filenames_count; ++i) {
      if (filenames[i] == nullptr) {
        return MOONSHINE_ERROR_INVALID_ARGUMENT;
      }
      const std::string key(filenames[i]);
      const bool is_tts_only =
          (key.size() >= 7 && key.compare(0, 7, "kokoro/") == 0) ||
          (key.size() >= 6 && key.compare(0, 6, "piper/") == 0);
      moonshine_tts::FileInformationMap &dest =
          is_tts_only ? tts_options.files : tts_options.g2p_options.files;
      if (memory[i] != nullptr && memory_sizes[i] > 0) {
        dest.set_memory(key, memory[i], static_cast<size_t>(memory_sizes[i]));
      } else {
        dest.set_path(key, std::filesystem::path(key));
      }
    }
    {
      // Legacy callers used in-memory key ``kokoro/model.ort``. Canonical key
      // is ``kokoro/model.onnx``.
      moonshine_tts::FileInformationMap &tf = tts_options.files;
      const std::string canon_k{moonshine_tts::kTtsKokoroModelOnnxKey};
      const auto canon_it = tf.entries.find(canon_k);
      const bool canon_ok = canon_it != tf.entries.end() &&
                            canon_it->second.memory != nullptr &&
                            canon_it->second.memory_size > 0;
      if (!canon_ok) {
        const auto leg = tf.entries.find("kokoro/model.ort");
        if (leg != tf.entries.end() && leg->second.memory != nullptr &&
            leg->second.memory_size > 0) {
          const moonshine_tts::FileInformation &src = leg->second;
          tf.entries[canon_k] = moonshine_tts::FileInformation{
              std::filesystem::path(canon_k), src.memory, src.memory_size};
        }
      }
    }
    std::string lang_from_options;
    bool lang_from_options_set = false;
    parse_tts_options(uncommon_options, tts_options, lang_from_options,
                      lang_from_options_set);
    std::string lang = (language != nullptr && language[0] != '\0')
                           ? std::string(language)
                           : std::string("en_us");
    if (lang_from_options_set) {
      lang = std::move(lang_from_options);
    }
    auto *synthesizer = new moonshine_tts::MoonshineTTS(lang, tts_options);
    return allocate_text_to_speech_synthesizer_handle(synthesizer);
  } catch (const std::exception &e) {
    LOGF("Failed to create TTS synthesizer from memory: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

/* Releases the resources used by a text to speech synthesizer.
 Returns zero on success, or a non-zero error code on failure.
*/
extern "C" void moonshine_free_tts_synthesizer(int32_t tts_synthesizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_tts_synthesizer(handle=%d)", tts_synthesizer_handle);
  }
  std::lock_guard<std::mutex> lock(text_to_speech_synthesizer_map_mutex);
  if (text_to_speech_synthesizer_map.contains(tts_synthesizer_handle)) {
    delete text_to_speech_synthesizer_map[tts_synthesizer_handle];
    text_to_speech_synthesizer_map[tts_synthesizer_handle] = nullptr;
    text_to_speech_synthesizer_map.erase(tts_synthesizer_handle);
  }
}

/* Synthesizes text to speech.
 Returns zero on success, or a non-zero error code on failure.
*/
extern "C" int32_t moonshine_text_to_speech(int32_t tts_synthesizer_handle,
                                 const char *text,
                                 const struct moonshine_option_t *options,
                                 uint64_t options_count, float **out_audio_data,
                                 uint64_t *out_audio_data_size,
                                 int32_t *out_sample_rate) {
  if (log_api_calls) {
    LOGF(
        "moonshine_text_to_speech(handle=%d, text=%s, options=%p, "
        "options_count=%" PRIu64 ", out_audio_data=%p, out_audio_data_size=%p, "
        "out_sample_rate=%p)",
        tts_synthesizer_handle, text, static_cast<const void *>(options),
        options_count,
        static_cast<void *>(out_audio_data),
        static_cast<void *>(out_audio_data_size),
        static_cast<void *>(out_sample_rate));
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  CHECK_TTS_SYNTHESIZER_HANDLE(tts_synthesizer_handle);
  try {
    moonshine_tts::MoonshineTTS *synth =
        text_to_speech_synthesizer_map[tts_synthesizer_handle];
    std::vector<std::pair<std::string, std::string>> tts_pairs;
    if (options != nullptr && options_count > 0) {
      tts_pairs.reserve(static_cast<size_t>(options_count));
      for (uint64_t i = 0; i < options_count; i++) {
        const moonshine_option_t &option = options[i];
        const std::string name =
            option.name != nullptr ? std::string(option.name) : std::string();
        const std::string value =
            option.value != nullptr ? std::string(option.value) : std::string();
        tts_pairs.emplace_back(std::move(name), std::move(value));
      }
    }
    const std::vector<float> wave = tts_pairs.empty()
                                        ? synth->synthesize(text)
                                        : synth->synthesize(text, tts_pairs);
    *out_sample_rate = moonshine_tts::MoonshineTTS::kSampleRateHz;
    *out_audio_data_size = wave.size();
    *out_audio_data = nullptr;
    if (!wave.empty()) {
      *out_audio_data =
          static_cast<float *>(std::malloc(wave.size() * sizeof(float)));
      if (*out_audio_data == nullptr) {
        return MOONSHINE_ERROR_UNKNOWN;
      }
      std::memcpy(*out_audio_data, wave.data(), wave.size() * sizeof(float));
    }
  } catch (const std::exception &e) {
    LOGF("Failed to synthesize text to speech: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

namespace {

char *malloc_string_copy(const std::string &s) {
  char *p = static_cast<char *>(std::malloc(s.size() + 1));
  if (p == nullptr) {
    return nullptr;
  }
  std::memcpy(p, s.c_str(), s.size() + 1);
  return p;
}

std::vector<std::string> split_comma_nonempty_language_tokens(const char *s) {
  std::vector<std::string> parts;
  if (s == nullptr) {
    return parts;
  }
  std::string cur;
  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(s);
       *p != '\0'; ++p) {
    if (*p == ',') {
      const std::string t = trim(cur);
      if (!t.empty()) {
        parts.push_back(t);
      }
      cur.clear();
    } else {
      cur += static_cast<char>(*p);
    }
  }
  const std::string t = trim(cur);
  if (!t.empty()) {
    parts.push_back(t);
  }
  return parts;
}

void append_unique_in_order(std::vector<std::string> &acc,
                            const std::vector<std::string> &more) {
  std::unordered_set<std::string> seen(acc.begin(), acc.end());
  for (const std::string &x : more) {
    if (seen.insert(x).second) {
      acc.push_back(x);
    }
  }
}

std::string json_utf8_string_literal(const std::string &s) {
  std::string r;
  r.push_back('"');
  for (unsigned char c : s) {
    switch (c) {
      case '"':
        r += "\\\"";
        break;
      case '\\':
        r += "\\\\";
        break;
      case '\b':
        r += "\\b";
        break;
      case '\f':
        r += "\\f";
        break;
      case '\n':
        r += "\\n";
        break;
      case '\r':
        r += "\\r";
        break;
      case '\t':
        r += "\\t";
        break;
      default:
        if (c < 0x20U) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "\\u%04x",
                        static_cast<unsigned int>(c));
          r += buf;
        } else {
          r += static_cast<char>(c);
        }
        break;
    }
  }
  r.push_back('"');
  return r;
}

std::string json_flat_string_array(const std::vector<std::string> &items) {
  std::string o;
  o.push_back('[');
  for (size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      o.push_back(',');
    }
    o += json_utf8_string_literal(items[i]);
  }
  o.push_back(']');
  return o;
}

std::string json_tts_voice_entry(
    const moonshine_tts::MoonshineTtsVoiceAvailability &v) {
  std::string o = "{\"id\":";
  o += json_utf8_string_literal(v.id);
  o += ",\"state\":";
  o += json_utf8_string_literal(v.available ? "found" : "missing");
  o += "}";
  return o;
}

std::string json_tts_voices_lang_array(
    const std::vector<moonshine_tts::MoonshineTtsVoiceAvailability> &voices) {
  std::string o;
  o.push_back('[');
  for (size_t i = 0; i < voices.size(); ++i) {
    if (i > 0) {
      o.push_back(',');
    }
    o += json_tts_voice_entry(voices[i]);
  }
  o.push_back(']');
  return o;
}

std::string json_tts_voices_root_object(
    const std::vector<std::pair<
        std::string, std::vector<moonshine_tts::MoonshineTtsVoiceAvailability>>>
        &rows) {
  std::string o;
  o.push_back('{');
  for (size_t i = 0; i < rows.size(); ++i) {
    if (i > 0) {
      o.push_back(',');
    }
    o += json_utf8_string_literal(rows[i].first);
    o.push_back(':');
    o += json_tts_voices_lang_array(rows[i].second);
  }
  o.push_back('}');
  return o;
}

void apply_g2p_dependency_query_c_options(
    const moonshine_option_t *options, uint64_t options_count,
    moonshine_tts::MoonshineG2POptions &g2p_options) {
  if (options == nullptr || options_count == 0) {
    return;
  }
  std::vector<std::pair<std::string, std::string>> g2p_pairs;
  g2p_pairs.reserve(options_count);
  for (uint64_t i = 0; i < options_count; i++) {
    const moonshine_option_t &option = options[i];
    const std::string name =
        option.name != nullptr ? std::string(option.name) : std::string();
    const std::string value =
        option.value != nullptr ? std::string(option.value) : std::string();
    const std::string key = replace_all(to_lowercase(name), "-", "_");
    if (key == "tts_root" || key == "path_root" || key == "model_root") {
      const std::string t = trim(value);
      if (!t.empty()) {
        g2p_options.g2p_root = std::filesystem::path(t);
      }
    } else if (key == "g2p_root") {
      g2p_options.g2p_root = std::filesystem::path(trim(value));
    } else if (key == "lang" || key == "language") {
      continue;
    } else if (key == "use_bundled_cpp_g2p_data" || key == "bundle_g2p_data") {
      (void)value;
    } else if (key == "log_api_calls") {
      log_api_calls = bool_from_string(value.c_str());
    } else if (key == "voice" || key == "speed" || key == "vocoder_engine" ||
               key == "engine" || key == "output" || key == "o" ||
               key == "piper_normalize_audio" || key == "piper_output_volume" ||
               key == "kokoro_dir" || key == "kokoro_model" ||
               key == "kokoro_model_onnx" || key == "kokoro_config" ||
               key == "kokoro_config_json" || key == "piper_onnx" ||
               key == "piper_model_onnx" || key == "piper_onnx_json" ||
               key == "piper_model_json" || key == "piper_onnx_config" ||
               key == "piper_voices_dir" || key == "voices_dir" ||
               key == "piper_voices_json_dir" || key == "voices_json_dir" ||
               key == "piper_noise_scale" ||
               key == "piper_noise_scale_override" || key == "piper_noise_w" ||
               key == "piper_noise_w_override") {
      continue;
    } else {
      g2p_pairs.emplace_back(name, value);
    }
  }
  g2p_options.parse_options(g2p_pairs);
}

void append_g2p_explicit_override_keys_from_c_options(
    const moonshine_option_t *options, uint64_t options_count,
    std::vector<std::string> &keys) {
  if (options == nullptr || options_count == 0) {
    return;
  }
  for (uint64_t i = 0; i < options_count; ++i) {
    if (options[i].name == nullptr || options[i].value == nullptr) {
      continue;
    }
    const std::string v = trim(std::string(options[i].value));
    if (v.empty()) {
      continue;
    }
    const std::string key =
        replace_all(to_lowercase(std::string(options[i].name)), "-", "_");
    if (key == "oov_onnx_override") {
      append_unique_in_order(
          keys, {std::string(moonshine_tts::kG2pOovOnnxOverrideKey)});
    } else if (key == "oov_onnx_config") {
      append_unique_in_order(
          keys, {std::string(moonshine_tts::kG2pOovOnnxConfigOverrideKey)});
    } else if (key == "portuguese_dict_path") {
      append_unique_in_order(
          keys, {std::string(moonshine_tts::kG2pPortugueseDictOverrideKey)});
    }
  }
}

}  // namespace

extern "C" int32_t moonshine_get_g2p_dependencies(const char *languages, const moonshine_option_t *options,
                                       uint64_t options_count, char **out_dependencies_json) {
  if (out_dependencies_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_dependencies_json = nullptr;
  const bool all_langs = (languages == nullptr || languages[0] == '\0');
  try {
    moonshine_tts::MoonshineG2POptions g2p_opts;
    apply_g2p_dependency_query_c_options(options, options_count, g2p_opts);
    if (g2p_opts.g2p_root.empty()) {
      g2p_opts.g2p_root = std::filesystem::current_path();
    }
    (void)g2p_opts;
    std::vector<std::string> keys;
    if (all_langs) {
      keys = moonshine_tts::
          moonshine_asset_catalog_all_g2p_dependency_keys_union();
    } else {
      const std::vector<std::string> parts =
          split_comma_nonempty_language_tokens(languages);
      if (parts.empty()) {
        keys = moonshine_tts::
            moonshine_asset_catalog_all_g2p_dependency_keys_union();
      } else {
        for (const std::string &part : parts) {
          const std::optional<std::vector<std::string>> chunk =
              moonshine_tts::moonshine_asset_catalog_g2p_dependency_keys(part);
          if (!chunk.has_value()) {
            LOGF(
                "moonshine_get_g2p_dependencies: unsupported language \"%s\"\n",
                part.c_str());
            return MOONSHINE_ERROR_INVALID_ARGUMENT;
          }
          append_unique_in_order(keys, *chunk);
        }
      }
    }
    append_g2p_explicit_override_keys_from_c_options(options, options_count,
                                                     keys);
    std::string joined;
    for (size_t i = 0; i < keys.size(); ++i) {
      if (i > 0) {
        joined += ',';
      }
      joined += keys[i];
    }
    char *buf = malloc_string_copy(joined);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_dependencies_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_g2p_dependencies failed: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" int32_t moonshine_get_tts_dependencies(const char *languages, const moonshine_option_t *options,
                                       uint64_t options_count, char **out_dependencies_json) {
  if (out_dependencies_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (log_api_calls) {
    LOGF(
        "moonshine_get_tts_dependencies(languages=%s, options=%p, "
        "options_count=%" PRIu64 ", out_dependencies_json=%p)",
        languages, reinterpret_cast<const void *>(options), options_count,
        reinterpret_cast<const void *>(out_dependencies_json));
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  *out_dependencies_json = nullptr;
  const bool all_langs = (languages == nullptr || languages[0] == '\0');
  try {
    moonshine_tts::MoonshineTTSOptions tts_opt;
    std::string cli_lang;
    bool lang_set = false;
    parse_tts_options(uncommon_options, tts_opt, cli_lang, lang_set);
    if (tts_opt.g2p_options.g2p_root.empty()) {
      tts_opt.g2p_options.g2p_root = std::filesystem::current_path();
    }
    std::vector<std::string> merged;
    if (all_langs) {
      merged = moonshine_tts::
          moonshine_asset_catalog_all_g2p_dependency_keys_union();
      const std::vector<std::string> tags =
          moonshine_tts::moonshine_asset_catalog_all_registered_language_tags();
      for (const std::string &tag : tags) {
        append_unique_in_order(
            merged,
            moonshine_tts::moonshine_catalog_tts_vocoder_only_dependency_keys(
                tag, tts_opt));
      }
    } else {
      const std::vector<std::string> parts =
          split_comma_nonempty_language_tokens(languages);
      if (parts.empty()) {
        merged = moonshine_tts::
            moonshine_asset_catalog_all_g2p_dependency_keys_union();
        const std::vector<std::string> tags = moonshine_tts::
            moonshine_asset_catalog_all_registered_language_tags();
        for (const std::string &tag : tags) {
          append_unique_in_order(
              merged,
              moonshine_tts::moonshine_catalog_tts_vocoder_only_dependency_keys(
                  tag, tts_opt));
        }
      } else {
        for (const std::string &part : parts) {
          const std::optional<std::vector<std::string>> g2p =
              moonshine_tts::moonshine_asset_catalog_g2p_dependency_keys(part);
          if (!g2p.has_value()) {
            LOGF(
                "moonshine_get_tts_dependencies: unsupported language \"%s\"\n",
                part.c_str());
            return MOONSHINE_ERROR_INVALID_ARGUMENT;
          }
          const std::vector<std::string> voc =
              moonshine_tts::moonshine_catalog_tts_vocoder_only_dependency_keys(
                  part, tts_opt);
          if (voc.empty()) {
            LOGF(
                "moonshine_get_tts_dependencies: no TTS layout for \"%s\" "
                "(voice prefix / paths?)\n",
                part.c_str());
            return MOONSHINE_ERROR_INVALID_ARGUMENT;
          }
          append_unique_in_order(merged, *g2p);
          append_unique_in_order(merged, voc);
        }
      }
    }
    const std::string dumped = json_flat_string_array(merged);
    char *buf = malloc_string_copy(dumped);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_dependencies_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_tts_dependencies failed: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

extern "C" int32_t moonshine_get_tts_voices(const char *languages, const moonshine_option_t *options,
                                 uint64_t options_count, char **out_voices_json) {
  if (out_voices_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_voices_json = nullptr;
  const bool all_langs = (languages == nullptr || languages[0] == '\0');
  try {
    OptionVector option_vector = parse_option_vector(options, options_count);
    OptionVector uncommon_options = parse_common_options(option_vector);
    moonshine_tts::MoonshineTTSOptions tts_opt;
    std::string cli_lang;
    bool lang_set = false;
    parse_tts_options(uncommon_options, tts_opt, cli_lang, lang_set);
    if (tts_opt.g2p_options.g2p_root.empty()) {
      tts_opt.g2p_options.g2p_root = std::filesystem::current_path();
    }
    (void)lang_set;
    (void)cli_lang;

    std::vector<std::pair<
        std::string, std::vector<moonshine_tts::MoonshineTtsVoiceAvailability>>>
        rows;

    if (all_langs) {
      const std::vector<std::string> tags =
          moonshine_tts::moonshine_asset_catalog_all_registered_language_tags();
      for (const std::string &tag : tags) {
        const std::vector<std::string> voc =
            moonshine_tts::moonshine_catalog_tts_vocoder_only_dependency_keys(
                tag, tts_opt);
        if (voc.empty()) {
          continue;
        }
        std::vector<moonshine_tts::MoonshineTtsVoiceAvailability> voices =
            moonshine_tts::moonshine_list_tts_voices_with_availability(tag,
                                                                       tts_opt);
        rows.emplace_back(tag, std::move(voices));
      }
    } else {
      const std::vector<std::string> parts =
          split_comma_nonempty_language_tokens(languages);
      if (parts.empty()) {
        const std::vector<std::string> tags = moonshine_tts::
            moonshine_asset_catalog_all_registered_language_tags();
        for (const std::string &tag : tags) {
          const std::vector<std::string> voc =
              moonshine_tts::moonshine_catalog_tts_vocoder_only_dependency_keys(
                  tag, tts_opt);
          if (voc.empty()) {
            continue;
          }
          std::vector<moonshine_tts::MoonshineTtsVoiceAvailability> voices =
              moonshine_tts::moonshine_list_tts_voices_with_availability(
                  tag, tts_opt);
          rows.emplace_back(tag, std::move(voices));
        }
      } else {
        for (const std::string &part : parts) {
          const std::optional<std::vector<std::string>> g2p =
              moonshine_tts::moonshine_asset_catalog_g2p_dependency_keys(part);
          if (!g2p.has_value()) {
            LOGF("moonshine_get_tts_voices: unsupported language \"%s\"\n",
                 part.c_str());
            return MOONSHINE_ERROR_INVALID_ARGUMENT;
          }
          const std::vector<std::string> voc =
              moonshine_tts::moonshine_catalog_tts_vocoder_only_dependency_keys(
                  part, tts_opt);
          if (voc.empty()) {
            LOGF(
                "moonshine_get_tts_voices: no TTS layout for \"%s\" (voice "
                "prefix / paths?)\n",
                part.c_str());
            return MOONSHINE_ERROR_INVALID_ARGUMENT;
          }
          std::vector<moonshine_tts::MoonshineTtsVoiceAvailability> voices =
              moonshine_tts::moonshine_list_tts_voices_with_availability(
                  part, tts_opt);
          rows.emplace_back(part, std::move(voices));
        }
      }
    }

    const std::string dumped = json_tts_voices_root_object(rows);
    char *buf = malloc_string_copy(dumped);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_voices_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_tts_voices failed: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

/* ------------------------------ GRAPHEME TO PHONEMIZER ------------------- */

namespace {

std::mutex grapheme_phonemizer_map_mutex;
std::map<int32_t, moonshine_tts::MoonshineG2P *> grapheme_phonemizer_map;
int32_t next_grapheme_phonemizer_handle = 0;

int32_t allocate_grapheme_phonemizer_handle(moonshine_tts::MoonshineG2P *g2p) {
  std::lock_guard<std::mutex> lock(grapheme_phonemizer_map_mutex);
  int32_t handle = next_grapheme_phonemizer_handle++;
  grapheme_phonemizer_map[handle] = g2p;
  return handle;
}

void parse_grapheme_phonemizer_options(
    const moonshine_option_t *in_options, uint64_t in_options_count,
    moonshine_tts::MoonshineG2POptions &g2p_options,
    std::string &cli_language_out, bool &language_was_set_out) {
  language_was_set_out = false;
  cli_language_out.clear();
  std::vector<std::pair<std::string, std::string>> g2p_pairs;
  g2p_pairs.reserve(in_options_count);
  for (uint64_t i = 0; i < in_options_count; i++) {
    const moonshine_option_t &option = in_options[i];
    const std::string name =
        option.name != nullptr ? std::string(option.name) : std::string();
    const std::string value =
        option.value != nullptr ? std::string(option.value) : std::string();
    const std::string key = replace_all(to_lowercase(name), "-", "_");
    if (key == "tts_root" || key == "path_root" || key == "model_root") {
      const std::string t = trim(value);
      if (!t.empty()) {
        g2p_options.g2p_root = std::filesystem::path(t);
      }
    } else if (key == "g2p_root") {
      g2p_options.g2p_root = std::filesystem::path(trim(value));
    } else if (key == "lang" || key == "language") {
      cli_language_out = trim(value);
      language_was_set_out = true;
    } else if (key == "use_bundled_cpp_g2p_data" || key == "bundle_g2p_data") {
      // Deprecated: cwd-based discovery removed; value ignored.
      (void)value;
    } else if (key == "log_api_calls") {
      log_api_calls = bool_from_string(value.c_str());
    } else {
      g2p_pairs.emplace_back(name, value);
    }
  }
  g2p_options.parse_options(g2p_pairs);
}

void finalize_g2p_options_for_phonemizer_create(
    moonshine_tts::MoonshineG2POptions &g2p_opt) {
  if (g2p_opt.g2p_root.empty()) {
    g2p_opt.g2p_root = std::filesystem::current_path();
  }
}

#define CHECK_GRAPHEME_PHONEMIZER_HANDLE(g2p_handle)                           \
  do {                                                                         \
    if ((g2p_handle) < 0 || !grapheme_phonemizer_map.contains((g2p_handle))) { \
      LOGF("Moonshine grapheme phonemizer handle is invalid: handle %d",       \
           (int)(g2p_handle));                                                 \
      return MOONSHINE_ERROR_INVALID_HANDLE;                                   \
    }                                                                          \
  } while (0)

}  // namespace

/* Creates a grapheme to phonemizer from files on disk.
 Returns a non-negative handle on success, or a negative error code on
 failure. The error code can be converted to a human-readable string using
 moonshine_error_to_string.
*/
extern "C" int32_t moonshine_create_grapheme_to_phonemizer_from_files(
    const char *language, const char **filenames, uint64_t filenames_count,
    const struct moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version) {
  (void)moonshine_version;
  if (filenames_count > 0 && filenames == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (log_api_calls) {
    LOGF(
        "moonshine_create_grapheme_to_phonemizer_from_files(language=%s, "
        "filenames=%p, filenames_count=%" PRIu64 ", options=%p, options_count=%" PRIu64 ", "
        "moonshine_version=%d)",
        language != nullptr ? language : "",
        reinterpret_cast<const void *>(filenames),
        filenames_count,
        static_cast<const void *>(options),
        options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  moonshine_tts::MoonshineG2POptions g2p_options;
  std::string lang_from_options;
  bool lang_from_options_set = false;
  try {
    parse_grapheme_phonemizer_options(options, options_count, g2p_options,
                                      lang_from_options, lang_from_options_set);
    for (uint64_t i = 0; i < filenames_count; ++i) {
      if (filenames[i] == nullptr) {
        return MOONSHINE_ERROR_INVALID_ARGUMENT;
      }
      const std::string key(filenames[i]);
      g2p_options.files.set_path(key, std::filesystem::path(key));
    }
    finalize_g2p_options_for_phonemizer_create(g2p_options);
    std::string lang = (language != nullptr && language[0] != '\0')
                           ? std::string(language)
                           : std::string("en_us");
    if (lang_from_options_set) {
      lang = std::move(lang_from_options);
    }
    auto *g2p = new moonshine_tts::MoonshineG2P(lang, std::move(g2p_options));
    return allocate_grapheme_phonemizer_handle(g2p);
  } catch (const std::exception &e) {
    LOGF("Failed to create grapheme phonemizer from files: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

/* Creates a grapheme to phonemizer from memory.
 Returns a non-negative handle on success, or a negative error code on
 failure. The error code can be converted to a human-readable string using
 moonshine_error_to_string.
*/
extern "C" int32_t moonshine_create_grapheme_to_phonemizer_from_memory(
    const char *language, const char **filenames,
    const uint64_t filenames_count, const uint8_t **memory,
    const uint64_t *memory_sizes, const struct moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version) {
  (void)moonshine_version;
  if (filenames_count > 0) {
    if (filenames == nullptr || memory == nullptr || memory_sizes == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (log_api_calls) {
    LOGF(
        "moonshine_create_grapheme_to_phonemizer_from_memory(language=%s, "
        "filenames=%p, "
        "filenames_count=%" PRIu64 ", memory=%p, memory_sizes=%p, options=%p, "
        "options_count=%" PRIu64 ", moonshine_version=%d)",
        language != nullptr ? language : "",
        reinterpret_cast<const void *>(filenames),
        filenames_count,
        reinterpret_cast<const void *>(memory),
        reinterpret_cast<const void *>(memory_sizes),
        static_cast<const void *>(options),
        options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  try {
    moonshine_tts::MoonshineG2POptions g2p_options;
    for (uint64_t i = 0; i < filenames_count; ++i) {
      if (filenames[i] == nullptr) {
        return MOONSHINE_ERROR_INVALID_ARGUMENT;
      }
      const std::string key(filenames[i]);
      if (memory[i] != nullptr && memory_sizes[i] > 0) {
        g2p_options.files.set_memory(key, memory[i],
                                     static_cast<size_t>(memory_sizes[i]));
      } else {
        g2p_options.files.set_path(key, std::filesystem::path(key));
      }
    }
    std::string lang_from_options;
    bool lang_from_options_set = false;
    parse_grapheme_phonemizer_options(options, options_count, g2p_options,
                                      lang_from_options, lang_from_options_set);
    finalize_g2p_options_for_phonemizer_create(g2p_options);
    std::string lang = (language != nullptr && language[0] != '\0')
                           ? std::string(language)
                           : std::string("en_us");
    if (lang_from_options_set) {
      lang = std::move(lang_from_options);
    }
    auto *g2p = new moonshine_tts::MoonshineG2P(lang, std::move(g2p_options));
    return allocate_grapheme_phonemizer_handle(g2p);
  } catch (const std::exception &e) {
    LOGF("Failed to create grapheme phonemizer from memory: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

/* Releases the resources used by a grapheme to phonemizer.
 Returns zero on success, or a non-zero error code on failure.
*/
extern "C" void moonshine_free_grapheme_to_phonemizer(
    int32_t grapheme_to_phonemizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_grapheme_to_phonemizer(handle=%d)",
         grapheme_to_phonemizer_handle);
  }
  std::lock_guard<std::mutex> lock(grapheme_phonemizer_map_mutex);
  if (grapheme_phonemizer_map.contains(grapheme_to_phonemizer_handle)) {
    delete grapheme_phonemizer_map[grapheme_to_phonemizer_handle];
    grapheme_phonemizer_map[grapheme_to_phonemizer_handle] = nullptr;
    grapheme_phonemizer_map.erase(grapheme_to_phonemizer_handle);
  }
}

/* Converts a text into the equivalent International Phonetic Alphabet (IPA)
 phonemes. Returns zero on success, or a non-zero error code on failure.
*/
extern "C" int32_t moonshine_text_to_phonemes(int32_t grapheme_to_phonemizer_handle,
                                   const char *text,
                                   const struct moonshine_option_t *options,
                                   uint64_t options_count,
                                   const char **out_phonemes,
                                   uint64_t *out_phonemes_count) {
  (void)options;
  (void)options_count;
  if (log_api_calls) {
    LOGF(
        "moonshine_text_to_phonemes(handle=%d, text=%s, options=%p, "
        "options_count=%" PRIu64 ", out_phonemes=%p, out_phonemes_count=%p)",
        grapheme_to_phonemizer_handle, text != nullptr ? text : "",
        static_cast<const void *>(options),
        options_count,
        static_cast<void *>(out_phonemes),
        static_cast<void *>(out_phonemes_count));
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  if (text == nullptr || out_phonemes == nullptr ||
      out_phonemes_count == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  CHECK_GRAPHEME_PHONEMIZER_HANDLE(grapheme_to_phonemizer_handle);
  try {
    moonshine_tts::MoonshineG2P *g2p =
        grapheme_phonemizer_map[grapheme_to_phonemizer_handle];
    const std::string ipa = g2p->text_to_ipa(text);
    *out_phonemes_count = 0;
    *out_phonemes = nullptr;
    if (ipa.empty()) {
      return MOONSHINE_ERROR_NONE;
    }
    char *buf = static_cast<char *>(std::malloc(ipa.size() + 1));
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    std::memcpy(buf, ipa.c_str(), ipa.size() + 1);
    *out_phonemes = buf;
    *out_phonemes_count = ipa.size();
  } catch (const std::exception &e) {
    LOGF("Failed to convert text to phonemes: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}