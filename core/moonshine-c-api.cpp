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
#include "clone-clip.h"
#include "debug-utils.h"
#include "moonshine-asset-catalog.h"
#include "moonshine-g2p.h"
#include "moonshine-model-catalog.h"
#include "moonshine-model-file-metadata.h"
#include "moonshine-model.h"
#include "moonshine-ort-allocator.h"
#include "moonshine-tensor-view.h"
#include "moonshine-tts.h"
#include "ort-utils.h"
#include "resampler.h"
#include "speech-clip.h"
#include "string-utils.h"
#include "text-embedder.h"
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
    } else if (option_name == "use_speculative_decoding") {
      out_options.use_speculative_decoding = bool_from_string(option_value);
    } else if (option_name == "identify_speakers") {
      out_options.identify_speakers = bool_from_string(option_value);
    } else if (option_name == "diarization_cluster_cadence") {
      out_options.diarization_cluster_cadence = float_from_string(option_value);
    } else if (option_name == "diarization_analyze_cadence") {
      out_options.diarization_analyze_cadence = float_from_string(option_value);
    } else if (option_name == "diarization_cluster_window_sec") {
      out_options.diarization_cluster_window_sec =
          float_from_string(option_value);
    } else if (option_name == "diarization_model_dir") {
      out_options.diarization_model_dir = option_value;
    } else if (option_name == "return_audio_data") {
      out_options.return_audio_data = bool_from_string(option_value);
    } else if (option_name == "log_output_text") {
      out_options.log_output_text = bool_from_string(option_value);
    } else if (option_name == "word_timestamps") {
      out_options.word_timestamps = bool_from_string(option_value);
    } else if (option_name == "spelling_model_path") {
      out_options.spelling_model_path = option_value;
    } else if (option_name == "ort_providers" ||
               option_name == "ort_provider") {
      out_options.ort_provider_names = ort_parse_provider_names(option_value);
    } else if (option_name == "coreml_cache_dir") {
      out_options.coreml_cache_dir = option_value;
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

/* Frees a caller-owned buffer that was allocated by this library with
   std::malloc (see the "release with free" functions documented in the
   header). Routing the free through the library ensures the allocation and
   deallocation use the same C runtime heap, which matters on Windows where the
   library and its host may link different runtimes. Safe on NULL. */
extern "C" void moonshine_free_buffer(void *ptr) { std::free(ptr); }

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

int32_t moonshine_load_transcriber_from_memory(
    const uint8_t *encoder_model_data, size_t encoder_model_data_size,
    const uint8_t *decoder_model_data, size_t decoder_model_data_size,
    const uint8_t *tokenizer_data, size_t tokenizer_data_size,
    const uint8_t *spelling_model_data, size_t spelling_model_data_size,
    uint32_t model_arch, const moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version) {
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (log_api_calls) {
    LOGF(
        "moonshine_load_transcriber_from_memory(encoder_model_data=%p, "
        "encoder_model_data_size=%zu, decoder_model_data=%p, "
        "decoder_model_data_size=%zu, tokenizer_data=%p, "
        "tokenizer_data_size=%zu, spelling_model_data=%p, "
        "spelling_model_data_size=%zu, model_arch=%d, options=%p, "
        "options_count=%" PRIu64 ", moonshine_version=%d)",
        (void *)(encoder_model_data), encoder_model_data_size,
        (void *)(decoder_model_data), decoder_model_data_size,
        (void *)(tokenizer_data), tokenizer_data_size,
        (void *)(spelling_model_data), spelling_model_data_size, model_arch,
        (void *)(options), options_count, moonshine_version);
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }

  // This entry point only understands a fixed encoder/decoder/tokenizer set,
  // so it can't express the assets newer architectures need. Clients built
  // against a current header have to move to the filename-keyed loader, but
  // ones built against an older header still pass that older version here and
  // keep working.
  if (moonshine_version >= MOONSHINE_FROM_MEMORY_REMOVED_VERSION) {
    LOGF(
        "moonshine_load_transcriber_from_memory() is no longer supported for "
        "clients built against Moonshine %d or newer (this caller passed %d). "
        "Use moonshine_load_transcriber_from_memory_files() instead.",
        MOONSHINE_FROM_MEMORY_REMOVED_VERSION, moonshine_version);
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
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
    transcriber_options.spelling_model_data = spelling_model_data;
    transcriber_options.spelling_model_data_size = spelling_model_data_size;
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

int32_t moonshine_load_transcriber_from_memory_files(
    const char **filenames, const uint8_t **memory,
    const uint64_t *memory_sizes, uint64_t file_count, uint32_t model_arch,
    const moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version) {
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (file_count > 0 &&
      (filenames == nullptr || memory == nullptr || memory_sizes == nullptr)) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (log_api_calls) {
    LOGF(
        "moonshine_load_transcriber_from_memory_files(filenames=%p, "
        "memory=%p, memory_sizes=%p, file_count=%" PRIu64
        ", model_arch=%d, options=%p, options_count=%" PRIu64
        ", moonshine_version=%d)",
        (const void *)(filenames), (const void *)(memory),
        (const void *)(memory_sizes), file_count, model_arch, (void *)(options),
        options_count, moonshine_version);
    for (uint64_t i = 0; i < file_count; i++) {
      LOGF("  file[%" PRIu64 "] = %s (%" PRIu64 " bytes)", i,
           filenames[i] ? filenames[i] : "(null)", memory_sizes[i]);
    }
  }

  Transcriber *transcriber = nullptr;
  try {
    TranscriberOptions transcriber_options;
    transcriber_options.model_source =
        TranscriberOptions::ModelSource::MEMORY_FILES;
    transcriber_options.model_arch = model_arch;
    for (uint64_t i = 0; i < file_count; ++i) {
      if (filenames[i] == nullptr) {
        return MOONSHINE_ERROR_INVALID_ARGUMENT;
      }
      const std::string key(filenames[i]);
      if (memory[i] != nullptr && memory_sizes[i] > 0) {
        transcriber_options.model_files.set_memory(
            key, memory[i], static_cast<size_t>(memory_sizes[i]));
      } else {
        // No buffer supplied: treat the canonical key as a path (relative to
        // the current working directory unless absolute), so callers can mix
        // in-memory and on-disk assets.
        transcriber_options.model_files.set_path(key,
                                                 std::filesystem::path(key));
      }
    }
    parse_transcriber_options(uncommon_options, transcriber_options);
    transcriber = new Transcriber(transcriber_options);
  } catch (const std::exception &e) {
    LOGF("Failed to load transcriber from memory files: %s\n", e.what());
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

int32_t moonshine_free_stream(int32_t transcriber_handle,
                              int32_t stream_handle) {
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

/* ------------------------------ EMBEDDING MODEL --------------------------- */

namespace {

std::mutex embedding_model_map_mutex;
std::map<int32_t, TextEmbedder *> embedding_model_map;
int32_t next_embedding_model_handle = 0;

int32_t allocate_embedding_model_handle(TextEmbedder *embedder) {
  std::lock_guard<std::mutex> lock(embedding_model_map_mutex);
  int32_t handle = next_embedding_model_handle++;
  embedding_model_map[handle] = embedder;
  return handle;
}

void free_embedding_model_handle(int32_t handle) {
  // Note: Caller must hold embedding_model_map_mutex
  delete embedding_model_map[handle];
  embedding_model_map[handle] = nullptr;
  embedding_model_map.erase(handle);
}

#define CHECK_EMBEDDING_MODEL_HANDLE(handle)                                  \
  do {                                                                        \
    if (handle < 0 || !embedding_model_map.contains(handle)) {                \
      LOGF("Moonshine embedding model handle is invalid: handle %d", handle); \
      return MOONSHINE_ERROR_INVALID_HANDLE;                                  \
    }                                                                         \
  } while (0)

}  // namespace

int32_t moonshine_create_embedding_model(const char *model_path,
                                         uint32_t model_arch,
                                         const char *model_variant) {
  if (log_api_calls) {
    LOGF(
        "moonshine_create_embedding_model(model_path=%s, model_arch=%d, "
        "model_variant=%s)",
        model_path, model_arch, model_variant ? model_variant : "q4");
  }

  if (model_path == nullptr) {
    LOGF("%s", "Invalid model_path: nullptr");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }

  TextEmbedder *embedder = nullptr;
  try {
    TextEmbedderOptions options;
    options.model_path = model_path;
    options.model_arch = static_cast<EmbeddingModelArch>(model_arch);
    options.model_variant = model_variant ? model_variant : "q4";

    embedder = new TextEmbedder(options);
  } catch (const std::exception &e) {
    delete embedder;
    LOGF("Failed to create embedding model: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return allocate_embedding_model_handle(embedder);
}

int32_t moonshine_create_embedding_model_from_memory(
    uint32_t model_arch, const char *model_variant, const char **filenames,
    uint64_t filenames_count, const uint8_t **memory,
    const uint64_t *memory_sizes, const struct moonshine_option_t *options,
    uint64_t options_count, int32_t moonshine_version) {
  (void)moonshine_version;
  (void)options;
  (void)options_count;
  if (filenames_count == 0 || filenames == nullptr || memory == nullptr ||
      memory_sizes == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (log_api_calls) {
    LOGF(
        "moonshine_create_embedding_model_from_memory(model_arch=%d, "
        "model_variant=%s, filenames_count=%" PRIu64 ")",
        model_arch, model_variant ? model_variant : "q4", filenames_count);
    for (uint64_t i = 0; i < filenames_count; i++) {
      LOGF("  file[%" PRIu64 "] = %s (%" PRIu64 " bytes)", i,
           filenames[i] ? filenames[i] : "(null)", memory_sizes[i]);
    }
  }

  // Resolve the model and tokenizer buffers from the keyed files. The embedding
  // manifest contains exactly two assets: the all-in-one model (a `.ort`, or a
  // legacy self-contained `.onnx`) and `tokenizer.bin`, so the model is simply
  // the non-tokenizer entry.
  const uint8_t *model_data = nullptr;
  size_t model_data_size = 0;
  const uint8_t *tokenizer_data = nullptr;
  size_t tokenizer_data_size = 0;
  for (uint64_t i = 0; i < filenames_count; ++i) {
    if (filenames[i] == nullptr) {
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const std::string key(filenames[i]);
    if (key == "tokenizer.bin") {
      tokenizer_data = memory[i];
      tokenizer_data_size = static_cast<size_t>(memory_sizes[i]);
    } else if (memory[i] != nullptr && memory_sizes[i] > 0) {
      // First (or matching) non-tokenizer asset is the model.
      model_data = memory[i];
      model_data_size = static_cast<size_t>(memory_sizes[i]);
    }
  }
  if (model_data == nullptr || model_data_size == 0) {
    LOGF("%s", "No embedding model buffer supplied");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }

  TextEmbedder *embedder = nullptr;
  try {
    TextEmbedderOptions embedder_options;
    embedder_options.model_arch = static_cast<EmbeddingModelArch>(model_arch);
    embedder_options.model_variant = model_variant ? model_variant : "q4";
    embedder_options.model_data = model_data;
    embedder_options.model_data_size = model_data_size;
    embedder_options.tokenizer_data = tokenizer_data;
    embedder_options.tokenizer_data_size = tokenizer_data_size;
    embedder = new TextEmbedder(embedder_options);
  } catch (const std::exception &e) {
    delete embedder;
    LOGF("Failed to create embedding model from memory: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return allocate_embedding_model_handle(embedder);
}

void moonshine_free_embedding_model(int32_t embedding_model_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_embedding_model(handle=%d)", embedding_model_handle);
  }
  std::lock_guard<std::mutex> lock(embedding_model_map_mutex);
  if (embedding_model_map.contains(embedding_model_handle)) {
    free_embedding_model_handle(embedding_model_handle);
  }
}

int32_t moonshine_calculate_embedding(int32_t embedding_model_handle,
                                      const char *sentence,
                                      float **out_embedding,
                                      uint64_t *out_embedding_size,
                                      const char *model_name) {
  (void)model_name;
  if (log_api_calls) {
    LOGF(
        "moonshine_calculate_embedding(handle=%d, sentence=%s, "
        "out_embedding=%p, out_embedding_size=%p, model_name=%s)",
        embedding_model_handle, sentence ? sentence : "(null)",
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
  CHECK_EMBEDDING_MODEL_HANDLE(embedding_model_handle);
  try {
    std::vector<float> emb =
        embedding_model_map[embedding_model_handle]->calculate_embedding(
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
    LOGF("Failed to calculate embedding: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

void moonshine_free_embedding(float *embedding) { std::free(embedding); }

int32_t moonshine_calculate_embedding_distance(int32_t embedding_model_handle,
                                               const float *embedding_a,
                                               const float *embedding_b,
                                               uint64_t embedding_size,
                                               float *out_similarity) {
  if (log_api_calls) {
    LOGF(
        "moonshine_calculate_embedding_distance(handle=%d, embedding_a=%p, "
        "embedding_b=%p, embedding_size=%" PRIu64 ", out_similarity=%p)",
        embedding_model_handle, static_cast<const void *>(embedding_a),
        static_cast<const void *>(embedding_b), embedding_size,
        static_cast<void *>(out_similarity));
  }
  if (embedding_a == nullptr || embedding_b == nullptr ||
      out_similarity == nullptr || embedding_size == 0) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  CHECK_EMBEDDING_MODEL_HANDLE(embedding_model_handle);
  try {
    std::vector<float> a(embedding_a, embedding_a + embedding_size);
    std::vector<float> b(embedding_b, embedding_b + embedding_size);
    *out_similarity =
        embedding_model_map[embedding_model_handle]->calculate_similarity(a, b);
  } catch (const std::exception &e) {
    LOGF("Failed to calculate embedding distance: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

/* ------------------------------ TEXT TO SPEECH ------------------------- */

namespace {

std::mutex text_to_speech_synthesizer_map_mutex;
std::map<int32_t, moonshine_tts::MoonshineTTS *> text_to_speech_synthesizer_map;
// Clone ASR owned by a ZipVoice synthesizer (same lifetime as the TTS handle).
std::map<int32_t, int32_t> text_to_speech_clone_asr_map;
int32_t next_text_to_speech_synthesizer_handle = 0;

int32_t allocate_text_to_speech_synthesizer_handle(
    moonshine_tts::MoonshineTTS *synthesizer, int32_t clone_asr_handle = -1) {
  std::lock_guard<std::mutex> lock(text_to_speech_synthesizer_map_mutex);
  int32_t handle = next_text_to_speech_synthesizer_handle++;
  text_to_speech_synthesizer_map[handle] = synthesizer;
  if (clone_asr_handle >= 0) {
    text_to_speech_clone_asr_map[handle] = clone_asr_handle;
  }
  return handle;
}

void parse_tts_options(const OptionVector &options,
                       moonshine_tts::MoonshineTTSOptions &out_options,
                       std::string &cli_language_out,
                       bool &language_was_set_out) {
  language_was_set_out = false;
  out_options.parse_options(options, &cli_language_out, &language_was_set_out);
}

constexpr float kDefaultCloneRequestedDurationSeconds = 4.0f;
constexpr float kDefaultCloneMaxExtensionSeconds = 1.5f;
constexpr int32_t kCloneClipSampleRate = 16000;

std::string transcript_line_text(const transcript_t *transcript) {
  std::string text;
  if (transcript == nullptr) {
    return text;
  }
  for (uint64_t i = 0; i < transcript->line_count; ++i) {
    const char *line = transcript->lines[i].text;
    if (line == nullptr) {
      continue;
    }
    std::string t = trim(std::string(line));
    if (t.empty()) {
      continue;
    }
    if (!text.empty()) {
      text += " ";
    }
    text += t;
  }
  return text;
}

std::vector<CloneClipWord> transcript_clone_words(
    const transcript_t *transcript) {
  std::vector<CloneClipWord> words;
  if (transcript == nullptr) {
    return words;
  }
  for (uint64_t i = 0; i < transcript->line_count; ++i) {
    const transcript_line_t &line = transcript->lines[i];
    for (uint64_t j = 0; j < line.word_count; ++j) {
      const transcript_word_t &w = line.words[j];
      if (w.text == nullptr || w.text[0] == '\0') {
        continue;
      }
      words.push_back({w.text, w.start, w.end});
    }
  }
  return words;
}

char *malloc_c_string(const std::string &text) {
  char *out = static_cast<char *>(std::malloc(text.size() + 1));
  if (out == nullptr) {
    return nullptr;
  }
  std::memcpy(out, text.c_str(), text.size() + 1);
  return out;
}

constexpr const char *kTtsCdnBaseUrl = "https://download.moonshine.ai/tts/";
constexpr const char *kCloneAsrPrefix = "clone_asr/";
constexpr size_t kCloneAsrPrefixLen = 10;  // strlen("clone_asr/")

// TTS language tags are regional (en_us); STT catalog codes are not (en).
std::string stt_language_from_tts_language(const std::string &tts_lang) {
  const std::string trimmed = trim(tts_lang);
  const size_t cut = trimmed.find_first_of("_-");
  if (cut == std::string::npos || cut == 0) {
    return to_lowercase(trimmed);
  }
  return to_lowercase(trimmed.substr(0, cut));
}

uint32_t default_stt_model_arch_for_language(const std::string &stt_lang) {
  for (const moonshine::SttCatalogLanguage &lang :
       moonshine::stt_catalog_listing()) {
    if (lang.code != stt_lang) {
      continue;
    }
    for (const moonshine::SttCatalogModel &model : lang.models) {
      if (model.is_default) {
        return static_cast<uint32_t>(model.model_arch);
      }
    }
    if (!lang.models.empty()) {
      return static_cast<uint32_t>(lang.models.front().model_arch);
    }
  }
  return static_cast<uint32_t>(MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING);
}

moonshine::ModelDependencyGroup tts_cdn_group_from_keys(
    const std::vector<std::string> &keys) {
  moonshine::ModelDependencyGroup group;
  group.base_url = kTtsCdnBaseUrl;
  group.files.reserve(keys.size());
  for (const std::string &key : keys) {
    moonshine::ModelFile file;
    file.name = key;
    // Base already ends with '/'; avoid a double slash.
    file.url = std::string(kTtsCdnBaseUrl) + key;
    const moonshine::ModelFileMetadata meta =
        moonshine::find_model_file_metadata(file.url);
    file.size = meta.size;
    file.checksum = meta.checksum;
    file.checksum_type = meta.checksum_type;
    group.files.push_back(std::move(file));
  }
  return group;
}

void append_clone_asr_groups(moonshine::ModelDependencies &deps,
                             const std::string &tts_language) {
  const std::string stt_lang = stt_language_from_tts_language(tts_language);
  const std::optional<moonshine::ModelDependencies> stt =
      moonshine::stt_model_dependencies(stt_lang, std::nullopt,
                                        /*include_spelling=*/false,
                                        /*include_word_timestamps=*/true);
  if (!stt.has_value()) {
    return;
  }
  for (moonshine::ModelDependencyGroup group : stt->groups) {
    group.role = "clone_asr";
    for (moonshine::ModelFile &file : group.files) {
      file.name = std::string(kCloneAsrPrefix) + file.name;
    }
    deps.groups.push_back(std::move(group));
  }
}

// Create the owned clone ASR from ZipVoice assets under g2p_root/clone_asr/ or
// clone_asr/* memory keys. Returns -1 if assets are missing or load fails.
int32_t try_create_clone_asr_from_assets(
    const moonshine_tts::MoonshineTTSOptions &tts_options,
    const std::string &tts_language) {
  FileInformationMap clone_files;
  for (const auto &entry : tts_options.files.entries) {
    const std::string &key = entry.first;
    if (key.size() <= kCloneAsrPrefixLen ||
        key.compare(0, kCloneAsrPrefixLen, kCloneAsrPrefix) != 0) {
      continue;
    }
    const std::string stt_key = key.substr(kCloneAsrPrefixLen);
    const FileInformation &info = entry.second;
    if (info.memory != nullptr && info.memory_size > 0) {
      clone_files.set_memory(stt_key, info.memory, info.memory_size);
    } else if (!info.path.empty()) {
      clone_files.set_path(stt_key, info.path);
    }
  }

  const std::filesystem::path clone_dir =
      std::filesystem::path(tts_options.g2p_options.g2p_root) / "clone_asr";
  const bool have_memory = !clone_files.entries.empty();
  const bool have_dir = !tts_options.g2p_options.g2p_root.empty() &&
                        std::filesystem::is_directory(clone_dir);

  if (!have_memory && !have_dir) {
    return -1;
  }

  const std::string stt_lang = stt_language_from_tts_language(tts_language);
  const uint32_t model_arch = default_stt_model_arch_for_language(stt_lang);

  try {
    TranscriberOptions transcriber_options;
    transcriber_options.word_timestamps = true;
    transcriber_options.model_arch = model_arch;
    std::string dir_str;
    if (have_memory) {
      transcriber_options.model_source =
          TranscriberOptions::ModelSource::MEMORY_FILES;
      transcriber_options.model_files = std::move(clone_files);
    } else {
      transcriber_options.model_source = TranscriberOptions::ModelSource::FILES;
      dir_str = clone_dir.string();
      transcriber_options.model_path = dir_str.c_str();
    }
    auto *transcriber = new Transcriber(transcriber_options);
    return allocate_transcriber_handle(transcriber);
  } catch (const std::exception &e) {
    LOGF("Failed to create clone ASR from ZipVoice assets: %s", e.what());
    return -1;
  }
}

bool zipvoice_clone_audio_present(
    const moonshine_tts::MoonshineTTSOptions &tts_options) {
  const std::string clone_key{moonshine_tts::kTtsZipVoiceCloneAudioKey};
  const auto pit = tts_options.files.entries.find(clone_key);
  return pit != tts_options.files.entries.end() &&
         pit->second.memory != nullptr &&
         pit->second.memory_size >= sizeof(float);
}

// Load owned clone ASR only when ZipVoice create must auto-transcribe a
// reference clip. Extract is VAD-only, so the ASR is not kept on the TTS
// handle afterward — keeping it loaded made every ZipVoice create (including
// preset voices) pay for a full STT session on the UI thread.
int32_t maybe_create_clone_asr_for_autotranscribe(
    const moonshine_tts::MoonshineTTSOptions &tts_options,
    const std::string &lang) {
  if (tts_options.vocoder_engine != "zipvoice") {
    return -1;
  }
  if (!tts_options.zipvoice_clone_transcript.empty()) {
    return -1;
  }
  if (!zipvoice_clone_audio_present(tts_options)) {
    return -1;
  }
  return try_create_clone_asr_from_assets(tts_options, lang);
}

// Refine ``audio`` (16 kHz) with ``asr_handle``. On success fills out_clip
// (malloc'd audio + optional transcript). Word-timestamp failure falls back to
// the requested window + line text.
int32_t refine_clip_with_asr(const std::vector<float> &audio,
                             int32_t asr_handle, float requested_duration,
                             float max_extension,
                             moonshine_speech_clip_t *out_clip) {
  *out_clip = moonshine_speech_clip_t{};
  if (asr_handle < 0 || audio.empty() || out_clip == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  const float audio_seconds = static_cast<float>(audio.size()) /
                              static_cast<float>(kCloneClipSampleRate);
  max_extension = std::min(max_extension,
                           std::max(0.0f, audio_seconds - requested_duration));

  transcript_t *asr_transcript = nullptr;
  {
    std::lock_guard<std::mutex> lock(transcriber_map_mutex);
    const auto tit = transcriber_map.find(asr_handle);
    if (tit == transcriber_map.end() || tit->second == nullptr) {
      return MOONSHINE_ERROR_INVALID_HANDLE;
    }
    try {
      tit->second->transcribe_without_streaming(
          const_cast<float *>(audio.data()),
          static_cast<uint64_t>(audio.size()), kCloneClipSampleRate, 0,
          &asr_transcript);
    } catch (const std::exception &e) {
      LOGF("clone ASR transcription failed: %s", e.what());
      return MOONSHINE_ERROR_UNKNOWN;
    }
  }

  const std::vector<CloneClipWord> words =
      transcript_clone_words(asr_transcript);
  CloneClipBounds bounds;
  if (!words.empty()) {
    bounds = refine_clone_clip_bounds(0.0f, requested_duration, words,
                                      max_extension, /*end_pad=*/0.05f);
  } else {
    // No word timings: keep the VAD/requested window and use line text.
    bounds.start_seconds = 0.0f;
    bounds.end_seconds = std::min(requested_duration, audio_seconds);
    bounds.transcript = transcript_line_text(asr_transcript);
  }

  bounds.start_seconds = std::max(0.0f, bounds.start_seconds);
  bounds.end_seconds = std::min(bounds.end_seconds, audio_seconds);
  if (!(bounds.end_seconds > bounds.start_seconds)) {
    bounds.start_seconds = 0.0f;
    bounds.end_seconds = audio_seconds;
  }

  const size_t from = static_cast<size_t>(
      std::lround(bounds.start_seconds * kCloneClipSampleRate));
  const size_t to = static_cast<size_t>(
      std::lround(bounds.end_seconds * kCloneClipSampleRate));
  const size_t begin = std::min(from, audio.size());
  const size_t end = std::min(std::max(to, begin + 1), audio.size());
  const size_t count = end - begin;

  float *buffer = static_cast<float *>(std::malloc(count * sizeof(float)));
  if (buffer == nullptr) {
    return MOONSHINE_ERROR_UNKNOWN;
  }
  std::memcpy(buffer, audio.data() + begin, count * sizeof(float));
  out_clip->audio_data = buffer;
  out_clip->audio_length = static_cast<uint64_t>(count);
  out_clip->start_time = bounds.start_seconds;
  out_clip->speech_duration = bounds.end_seconds - bounds.start_seconds;
  out_clip->is_complete = 1;
  out_clip->transcript = nullptr;
  if (!bounds.transcript.empty()) {
    out_clip->transcript = malloc_c_string(bounds.transcript);
    if (out_clip->transcript == nullptr) {
      std::free(buffer);
      *out_clip = moonshine_speech_clip_t{};
      return MOONSHINE_ERROR_UNKNOWN;
    }
  }
  return MOONSHINE_ERROR_NONE;
}

// When ZipVoice has a caller-supplied clone clip but no transcript, refine
// with the owned clone ASR (from g2p_root/clone_asr or clone_asr/... memory
// keys). Keeps the TTS
// library free of an ASR dependency.
void maybe_autotranscribe_zipvoice_clone(
    const OptionVector &options,
    moonshine_tts::MoonshineTTSOptions &tts_options,
    std::vector<uint8_t> *owned_clone_pcm, int32_t clone_asr_handle) {
  if (tts_options.vocoder_engine != "zipvoice") {
    return;
  }
  if (!tts_options.zipvoice_clone_transcript.empty()) {
    return;
  }
  if (clone_asr_handle < 0) {
    return;
  }
  const std::string clone_key{moonshine_tts::kTtsZipVoiceCloneAudioKey};
  const auto pit = tts_options.files.entries.find(clone_key);
  if (pit == tts_options.files.entries.end() || pit->second.memory == nullptr ||
      pit->second.memory_size < sizeof(float)) {
    return;
  }
  float requested_duration = kDefaultCloneRequestedDurationSeconds;
  float max_extension = kDefaultCloneMaxExtensionSeconds;
  for (const auto &kv : options) {
    std::string key = replace_all(to_lowercase(kv.first), "-", "_");
    if (key == "clip_duration_seconds" || key == "requested_duration_seconds") {
      requested_duration = float_from_string(kv.second);
    } else if (key == "max_extension_seconds") {
      max_extension = float_from_string(kv.second);
    }
  }
  const size_t n = pit->second.memory_size / sizeof(float);
  const float *pcm = reinterpret_cast<const float *>(pit->second.memory);
  const std::vector<float> input(pcm, pcm + n);
  const float source_rate =
      tts_options.zipvoice_clone_sample_rate > 0
          ? static_cast<float>(tts_options.zipvoice_clone_sample_rate)
          : static_cast<float>(moonshine_tts::MoonshineTTS::kSampleRateHz);
  const std::vector<float> audio =
      source_rate == static_cast<float>(kCloneClipSampleRate)
          ? input
          : resample_audio(input, source_rate, kCloneClipSampleRate);
  if (audio.empty()) {
    return;
  }

  moonshine_speech_clip_t refined{};
  const int32_t err = refine_clip_with_asr(
      audio, clone_asr_handle, requested_duration, max_extension, &refined);
  if (err != MOONSHINE_ERROR_NONE || refined.audio_data == nullptr ||
      refined.audio_length == 0) {
    if (refined.audio_data != nullptr) {
      moonshine_free_buffer(refined.audio_data);
    }
    if (refined.transcript != nullptr) {
      moonshine_free_buffer(refined.transcript);
    }
    LOGF("ZipVoice clone refine failed: %d", err);
    return;
  }

  if (owned_clone_pcm != nullptr) {
    const size_t byte_count = refined.audio_length * sizeof(float);
    owned_clone_pcm->resize(byte_count);
    std::memcpy(owned_clone_pcm->data(), refined.audio_data, byte_count);
    pit->second.memory = owned_clone_pcm->data();
    pit->second.memory_size = byte_count;
  }
  moonshine_free_buffer(refined.audio_data);

  if (refined.transcript != nullptr && refined.transcript[0] != '\0') {
    tts_options.zipvoice_clone_transcript = refined.transcript;
  }
  if (refined.transcript != nullptr) {
    moonshine_free_buffer(refined.transcript);
  }
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

int32_t moonshine_extract_speech_clip(
    const float *audio_data, uint64_t audio_length, int32_t sample_rate,
    int32_t tts_synthesizer_handle, const moonshine_option_t *options,
    uint64_t options_count, moonshine_speech_clip_t *out_clip) {
  if (out_clip == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_clip = moonshine_speech_clip_t{};
  if (audio_data == nullptr || audio_length == 0 || sample_rate <= 0) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  {
    std::lock_guard<std::mutex> lock(text_to_speech_synthesizer_map_mutex);
    if (tts_synthesizer_handle < 0 ||
        !text_to_speech_synthesizer_map.contains(tts_synthesizer_handle)) {
      LOGF("Moonshine text to speech synthesizer handle is invalid: handle %d",
           (int)tts_synthesizer_handle);
      return MOONSHINE_ERROR_INVALID_HANDLE;
    }
  }

  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (log_api_calls) {
    LOGF("moonshine_extract_speech_clip(audio_length=%" PRIu64
         ", sample_rate=%d, tts_handle=%d, options_count=%" PRIu64 ")",
         audio_length, sample_rate, tts_synthesizer_handle, options_count);
  }

  SpeechClipOptions clip_options;
  for (const auto &[key, value] : uncommon_options) {
    if (key == "clip_duration_seconds") {
      clip_options.clip_duration_seconds = float_from_string(value);
    } else if (key == "minimum_speech_seconds") {
      clip_options.minimum_speech_seconds = float_from_string(value);
    } else if (key == "vad_threshold") {
      clip_options.vad_threshold = float_from_string(value);
    } else if (key == "tail_pad_seconds") {
      clip_options.tail_pad_seconds = float_from_string(value);
    }
  }

  SpeechClip clip;
  try {
    clip = extract_speech_clip(audio_data, static_cast<size_t>(audio_length),
                               sample_rate, clip_options);
  } catch (const std::exception &e) {
    LOGF("moonshine_extract_speech_clip failed: %s", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }

  out_clip->start_time = clip.start_time_seconds;
  out_clip->speech_duration = clip.speech_seconds;
  out_clip->is_complete = clip.is_complete ? 1 : 0;
  out_clip->transcript = nullptr;
  if (!clip.is_complete || clip.audio.empty()) {
    return MOONSHINE_ERROR_NONE;
  }

  // VAD only here. Owned clone ASR refine + transcript run once at ZipVoice
  // create (maybe_autotranscribe_zipvoice_clone), so incremental VoiceClone
  // search stays responsive on the audio / UI thread.
  const size_t byte_count = clip.audio.size() * sizeof(float);
  float *buffer = static_cast<float *>(std::malloc(byte_count));
  if (buffer == nullptr) {
    return MOONSHINE_ERROR_UNKNOWN;
  }
  std::memcpy(buffer, clip.audio.data(), byte_count);
  out_clip->audio_data = buffer;
  out_clip->audio_length = static_cast<uint64_t>(clip.audio.size());
  return MOONSHINE_ERROR_NONE;
}

int32_t moonshine_create_tts_synthesizer_from_files(
    const char *language, const char **filenames, uint64_t filenames_count,
    const struct moonshine_option_t *options, uint64_t options_count,
    int32_t moonshine_version) {
  OptionVector option_vector = parse_option_vector(options, options_count);
  OptionVector uncommon_options = parse_common_options(option_vector);
  if (log_api_calls) {
    LOGF(
        "moonshine_create_tts_synthesizer_from_files(language=%s, "
        "filenames=%p, filenames_count=%" PRIu64
        ", options=%p, options_count=%" PRIu64
        ", "
        "moonshine_version=%d)",
        language, reinterpret_cast<const void *>(filenames), filenames_count,
        static_cast<const void *>(options), options_count, moonshine_version);
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
  const int32_t clone_asr =
      maybe_create_clone_asr_for_autotranscribe(tts_options, lang);
  std::vector<uint8_t> owned_clone_pcm;
  maybe_autotranscribe_zipvoice_clone(uncommon_options, tts_options,
                                      &owned_clone_pcm, clone_asr);
  // ASR was only needed for autotranscribe; do not keep it on the TTS handle.
  if (clone_asr >= 0) {
    free_transcriber_handle(clone_asr);
  }
  try {
    auto *synthesizer = new moonshine_tts::MoonshineTTS(lang, tts_options);
    return allocate_text_to_speech_synthesizer_handle(synthesizer, -1);
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
        "filenames_count=%" PRIu64
        ", memory=%p, memory_sizes=%p, options=%p, "
        "options_count=%" PRIu64 ", moonshine_version=%d)",
        language, reinterpret_cast<const void *>(filenames), filenames_count,
        reinterpret_cast<const void *>(memory),
        reinterpret_cast<const void *>(memory_sizes),
        static_cast<const void *>(options), options_count, moonshine_version);
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
          (key.size() >= 6 && key.compare(0, 6, "piper/") == 0) ||
          (key.size() >= 9 && key.compare(0, 9, "zipvoice/") == 0) ||
          (key.size() >= 10 && key.compare(0, 10, "clone_asr/") == 0);
      FileInformationMap &dest =
          is_tts_only ? tts_options.files : tts_options.g2p_options.files;
      if (memory[i] != nullptr && memory_sizes[i] > 0) {
        dest.set_memory(key, memory[i], static_cast<size_t>(memory_sizes[i]));
      } else {
        dest.set_path(key, std::filesystem::path(key));
      }
    }
    {
      // Kokoro shipped as ONNX before it moved to ORT, so accept the old
      // in-memory key under the canonical one.
      FileInformationMap &tf = tts_options.files;
      const std::string canon_k{moonshine_tts::kTtsKokoroModelKey};
      const auto canon_it = tf.entries.find(canon_k);
      const bool canon_ok = canon_it != tf.entries.end() &&
                            canon_it->second.memory != nullptr &&
                            canon_it->second.memory_size > 0;
      if (!canon_ok) {
        const auto leg = tf.entries.find("kokoro/model.onnx");
        if (leg != tf.entries.end() && leg->second.memory != nullptr &&
            leg->second.memory_size > 0) {
          const FileInformation &src = leg->second;
          tf.entries[canon_k] = FileInformation{std::filesystem::path(canon_k),
                                                src.memory, src.memory_size};
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
    const int32_t clone_asr =
        maybe_create_clone_asr_for_autotranscribe(tts_options, lang);
    try {
      std::vector<uint8_t> owned_clone_pcm;
      maybe_autotranscribe_zipvoice_clone(uncommon_options, tts_options,
                                          &owned_clone_pcm, clone_asr);
      if (clone_asr >= 0) {
        free_transcriber_handle(clone_asr);
      }
      auto *synthesizer = new moonshine_tts::MoonshineTTS(lang, tts_options);
      return allocate_text_to_speech_synthesizer_handle(synthesizer, -1);
    } catch (...) {
      if (clone_asr >= 0) {
        free_transcriber_handle(clone_asr);
      }
      throw;
    }
  } catch (const std::exception &e) {
    LOGF("Failed to create TTS synthesizer from memory: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

/* Releases the resources used by a text to speech synthesizer.
 Returns zero on success, or a non-zero error code on failure.
*/
void moonshine_free_tts_synthesizer(int32_t tts_synthesizer_handle) {
  if (log_api_calls) {
    LOGF("moonshine_free_tts_synthesizer(handle=%d)", tts_synthesizer_handle);
  }
  int32_t clone_asr = -1;
  {
    std::lock_guard<std::mutex> lock(text_to_speech_synthesizer_map_mutex);
    if (text_to_speech_synthesizer_map.contains(tts_synthesizer_handle)) {
      delete text_to_speech_synthesizer_map[tts_synthesizer_handle];
      text_to_speech_synthesizer_map[tts_synthesizer_handle] = nullptr;
      text_to_speech_synthesizer_map.erase(tts_synthesizer_handle);
    }
    const auto asr_it =
        text_to_speech_clone_asr_map.find(tts_synthesizer_handle);
    if (asr_it != text_to_speech_clone_asr_map.end()) {
      clone_asr = asr_it->second;
      text_to_speech_clone_asr_map.erase(asr_it);
    }
  }
  if (clone_asr >= 0) {
    free_transcriber_handle(clone_asr);
  }
}

namespace {

/* Converts a C ``moonshine_option_t`` array into the name/value pair vector the
   ``moonshine_tts::MoonshineTTS`` synthesize overloads expect. Null option
   names/values become empty strings. Shared by moonshine_text_to_speech and
   moonshine_phonemes_to_speech. */
std::vector<std::pair<std::string, std::string>> tts_option_pairs_from_c(
    const struct moonshine_option_t *options, uint64_t options_count) {
  std::vector<std::pair<std::string, std::string>> tts_pairs;
  if (options == nullptr || options_count == 0) {
    return tts_pairs;
  }
  tts_pairs.reserve(static_cast<size_t>(options_count));
  for (uint64_t i = 0; i < options_count; i++) {
    const moonshine_option_t &option = options[i];
    const std::string name =
        option.name != nullptr ? std::string(option.name) : std::string();
    const std::string value =
        option.value != nullptr ? std::string(option.value) : std::string();
    tts_pairs.emplace_back(name, value);
  }
  return tts_pairs;
}

}  // namespace

/* Synthesizes text to speech.
 Returns zero on success, or a non-zero error code on failure.
*/
int32_t moonshine_text_to_speech(int32_t tts_synthesizer_handle,
                                 const char *text,
                                 const struct moonshine_option_t *options,
                                 uint64_t options_count, float **out_audio_data,
                                 uint64_t *out_audio_data_size,
                                 int32_t *out_sample_rate) {
  if (log_api_calls) {
    LOGF(
        "moonshine_text_to_speech(handle=%d, text=%s, options=%p, "
        "options_count=%" PRIu64
        ", out_audio_data=%p, out_audio_data_size=%p, "
        "out_sample_rate=%p)",
        tts_synthesizer_handle, text, static_cast<const void *>(options),
        options_count, static_cast<void *>(out_audio_data),
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
    const std::vector<std::pair<std::string, std::string>> tts_pairs =
        tts_option_pairs_from_c(options, options_count);
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

int32_t moonshine_phonemes_to_speech(int32_t tts_synthesizer_handle,
                                     const char *phonemes,
                                     const struct moonshine_option_t *options,
                                     uint64_t options_count,
                                     float **out_audio_data,
                                     uint64_t *out_audio_data_size,
                                     int32_t *out_sample_rate) {
  if (log_api_calls) {
    LOGF(
        "moonshine_phonemes_to_speech(handle=%d, phonemes=%s, options=%p, "
        "options_count=%" PRIu64
        ", out_audio_data=%p, out_audio_data_size=%p, "
        "out_sample_rate=%p)",
        tts_synthesizer_handle, phonemes, static_cast<const void *>(options),
        options_count, static_cast<void *>(out_audio_data),
        static_cast<void *>(out_audio_data_size),
        static_cast<void *>(out_sample_rate));
    for (uint64_t i = 0; i < options_count; i++) {
      const moonshine_option_t &option = options[i];
      LOGF("  option[%" PRIu64 "] = %s=%s", i, option.name, option.value);
    }
  }
  if (phonemes == nullptr || out_audio_data == nullptr ||
      out_audio_data_size == nullptr || out_sample_rate == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  CHECK_TTS_SYNTHESIZER_HANDLE(tts_synthesizer_handle);
  try {
    moonshine_tts::MoonshineTTS *synth =
        text_to_speech_synthesizer_map[tts_synthesizer_handle];
    const std::vector<std::pair<std::string, std::string>> tts_pairs =
        tts_option_pairs_from_c(options, options_count);
    const std::vector<float> wave =
        tts_pairs.empty()
            ? synth->synthesize_from_phonemes(phonemes)
            : synth->synthesize_from_phonemes(phonemes, tts_pairs);
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
    LOGF("Failed to synthesize phonemes to speech: %s", e.what());
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

// Serializes one model file as a JSON object carrying everything a client needs
// to fetch and verify it:
//   { "name": "encoder.ort", "url": "https://.../encoder.ort",
//     "size": 12345, "checksum": "abc==", "checksum_type": "crc32c" }
// `size` is null and `checksum`/`checksum_type` are "" when the metadata
// registry has no entry for the file (see moonshine-model-file-metadata.*).
std::string json_model_file(const moonshine::ModelFile &file) {
  std::string o = "{\"name\":";
  o += json_utf8_string_literal(file.name);
  o += ",\"url\":";
  o += json_utf8_string_literal(file.url);
  o += ",\"size\":";
  if (file.size >= 0) {
    o += std::to_string(file.size);
  } else {
    o += "null";
  }
  o += ",\"checksum\":";
  o += json_utf8_string_literal(file.checksum);
  o += ",\"checksum_type\":";
  o += json_utf8_string_literal(file.checksum_type);
  o += "}";
  return o;
}

// Serializes a model download manifest as a JSON object with a "groups" array.
// Each group is { "base_url": "...", "files": [<file object>, ...] } where each
// file object is produced by json_model_file (name/url/size/checksum/
// checksum_type). STT models emit a single group (plus an optional spelling
// group); embedding models emit one group.
std::string json_model_dependencies(const moonshine::ModelDependencies &deps) {
  std::string o = "{\"groups\":[";
  for (size_t i = 0; i < deps.groups.size(); ++i) {
    if (i > 0) {
      o.push_back(',');
    }
    const moonshine::ModelDependencyGroup &group = deps.groups[i];
    o += "{\"base_url\":";
    o += json_utf8_string_literal(group.base_url);
    if (!group.role.empty()) {
      o += ",\"role\":";
      o += json_utf8_string_literal(group.role);
    }
    o += ",\"files\":[";
    for (size_t j = 0; j < group.files.size(); ++j) {
      if (j > 0) {
        o.push_back(',');
      }
      o += json_model_file(group.files[j]);
    }
    o += "]}";
  }
  o += "]}";
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
               key == "normalize_audio" || key == "piper_normalize_audio" ||
               key == "output_volume" || key == "piper_output_volume" ||
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

int32_t moonshine_get_g2p_dependencies(const char *languages,
                                       const moonshine_option_t *options,
                                       uint64_t options_count,
                                       char **out_dependencies_json) {
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

int32_t moonshine_get_tts_dependencies(const char *languages,
                                       const moonshine_option_t *options,
                                       uint64_t options_count,
                                       char **out_dependencies_json) {
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
    std::vector<std::string> lang_tags_for_clone;
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
      lang_tags_for_clone = tags;
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
        lang_tags_for_clone = tags;
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
          lang_tags_for_clone.push_back(part);
        }
      }
    }

    moonshine::ModelDependencies deps;
    if (!merged.empty()) {
      deps.groups.push_back(tts_cdn_group_from_keys(merged));
    }

    // ZipVoice owns a catalog-default STT (with word timestamps) for clone
    // refine. Advertise it so bindings download under clone_asr/.
    if (tts_opt.vocoder_engine == "zipvoice") {
      std::unordered_set<std::string> seen_stt_langs;
      for (const std::string &tag : lang_tags_for_clone) {
        const std::string stt_lang = stt_language_from_tts_language(tag);
        if (!seen_stt_langs.insert(stt_lang).second) {
          continue;
        }
        append_clone_asr_groups(deps, tag);
      }
      // Bare zipvoice / all-langs with zipvoice voice still needs English STT
      // when no language tags were collected.
      if (lang_tags_for_clone.empty()) {
        append_clone_asr_groups(deps, "en_us");
      }
    }

    const std::string dumped = json_model_dependencies(deps);
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

int32_t moonshine_get_tts_voices(const char *languages,
                                 const moonshine_option_t *options,
                                 uint64_t options_count,
                                 char **out_voices_json) {
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

/* ------------------------------ MODEL CATALOG --------------------------- */

namespace {

std::string normalize_option_key(const char *name) {
  if (name == nullptr) {
    return std::string();
  }
  return replace_all(to_lowercase(std::string(name)), "-", "_");
}

// Parses an integer option value; returns std::nullopt on empty/invalid input.
std::optional<int32_t> parse_int_option(const std::string &value) {
  const std::string t = trim(value);
  if (t.empty()) {
    return std::nullopt;
  }
  try {
    size_t consumed = 0;
    const long parsed = std::stol(t, &consumed, 10);
    if (consumed != t.size()) {
      return std::nullopt;
    }
    return static_cast<int32_t>(parsed);
  } catch (const std::exception &) {
    return std::nullopt;
  }
}

}  // namespace

int32_t moonshine_get_stt_dependencies(const char *language,
                                       const moonshine_option_t *options,
                                       uint64_t options_count,
                                       char **out_dependencies_json) {
  if (out_dependencies_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_dependencies_json = nullptr;
  const std::string language_str =
      language != nullptr ? std::string(language) : std::string();
  if (trim(language_str).empty()) {
    LOGF("moonshine_get_stt_dependencies: language must not be empty%s\n", "");
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  try {
    std::optional<int32_t> model_arch;
    bool include_spelling = false;
    bool include_word_timestamps = false;
    for (uint64_t i = 0; i < options_count; ++i) {
      const std::string key = normalize_option_key(options[i].name);
      const std::string value = options[i].value != nullptr
                                    ? std::string(options[i].value)
                                    : std::string();
      if (key == "model_arch") {
        const std::optional<int32_t> parsed = parse_int_option(value);
        if (!parsed.has_value()) {
          LOGF("moonshine_get_stt_dependencies: invalid model_arch \"%s\"\n",
               value.c_str());
          return MOONSHINE_ERROR_INVALID_ARGUMENT;
        }
        model_arch = parsed;
      } else if (key == "include_spelling" || key == "spelling") {
        include_spelling = bool_from_string(value.c_str());
      } else if (key == "spelling_model_path") {
        // The loader uses a spelling model when given its path; mirror that by
        // adding the spelling group to the manifest.
        include_spelling = !trim(value).empty();
      } else if (key == "word_timestamps") {
        include_word_timestamps = bool_from_string(value.c_str());
      }
    }

    const std::optional<moonshine::ModelDependencies> deps =
        moonshine::stt_model_dependencies(trim(language_str), model_arch,
                                          include_spelling,
                                          include_word_timestamps);
    if (!deps.has_value()) {
      LOGF(
          "moonshine_get_stt_dependencies: unknown language \"%s\" or "
          "model_arch\n",
          language_str.c_str());
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const std::string dumped = json_model_dependencies(*deps);
    char *buf = malloc_string_copy(dumped);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_dependencies_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_stt_dependencies failed: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int32_t moonshine_get_embedding_dependencies(const char *model_name,
                                             const moonshine_option_t *options,
                                             uint64_t options_count,
                                             char **out_dependencies_json) {
  if (out_dependencies_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  if (options_count > 0 && options == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_dependencies_json = nullptr;
  // Default to the only published embedding model when none is requested.
  std::string resolved_model_name =
      model_name != nullptr ? trim(std::string(model_name)) : std::string();
  if (resolved_model_name.empty()) {
    resolved_model_name = "embeddinggemma-300m";
  }
  try {
    std::string variant;
    for (uint64_t i = 0; i < options_count; ++i) {
      const std::string key = normalize_option_key(options[i].name);
      const std::string value = options[i].value != nullptr
                                    ? std::string(options[i].value)
                                    : std::string();
      if (key == "variant" || key == "model_variant") {
        variant = trim(value);
      }
    }

    const std::optional<moonshine::ModelDependencies> deps =
        moonshine::embedding_model_dependencies(resolved_model_name, variant);
    if (!deps.has_value()) {
      LOGF(
          "moonshine_get_embedding_dependencies: unknown model \"%s\" or "
          "variant \"%s\"\n",
          resolved_model_name.c_str(), variant.c_str());
      return MOONSHINE_ERROR_INVALID_ARGUMENT;
    }
    const std::string dumped = json_model_dependencies(*deps);
    char *buf = malloc_string_copy(dumped);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_dependencies_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_embedding_dependencies failed: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int32_t moonshine_get_diarization_dependencies(char **out_dependencies_json) {
  if (out_dependencies_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_dependencies_json = nullptr;
  try {
    const std::string dumped =
        json_model_dependencies(moonshine::diarization_model_dependencies());
    char *buf = malloc_string_copy(dumped);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_dependencies_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_diarization_dependencies failed: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int32_t moonshine_get_stt_catalog(char **out_catalog_json) {
  if (out_catalog_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_catalog_json = nullptr;
  try {
    std::string o = "{\"languages\":[";
    const std::vector<moonshine::SttCatalogLanguage> languages =
        moonshine::stt_catalog_listing();
    for (size_t i = 0; i < languages.size(); ++i) {
      if (i > 0) {
        o.push_back(',');
      }
      const moonshine::SttCatalogLanguage &lang = languages[i];
      o += "{\"code\":";
      o += json_utf8_string_literal(lang.code);
      o += ",\"english_name\":";
      o += json_utf8_string_literal(lang.english_name);
      o += ",\"models\":[";
      for (size_t j = 0; j < lang.models.size(); ++j) {
        if (j > 0) {
          o.push_back(',');
        }
        const moonshine::SttCatalogModel &model = lang.models[j];
        o += "{\"model_arch\":";
        o += std::to_string(model.model_arch);
        o += ",\"download_url\":";
        o += json_utf8_string_literal(model.download_url);
        o += ",\"is_default\":";
        o += model.is_default ? "true" : "false";
        o += "}";
      }
      o += "]}";
    }
    o += "]}";
    char *buf = malloc_string_copy(o);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_catalog_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_stt_catalog failed: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int32_t moonshine_get_embedding_catalog(char **out_catalog_json) {
  if (out_catalog_json == nullptr) {
    return MOONSHINE_ERROR_INVALID_ARGUMENT;
  }
  *out_catalog_json = nullptr;
  try {
    std::string o = "{\"models\":[";
    const std::vector<moonshine::EmbeddingCatalogModel> models =
        moonshine::embedding_catalog_listing();
    for (size_t i = 0; i < models.size(); ++i) {
      if (i > 0) {
        o.push_back(',');
      }
      const moonshine::EmbeddingCatalogModel &model = models[i];
      o += "{\"name\":";
      o += json_utf8_string_literal(model.name);
      o += ",\"english_name\":";
      o += json_utf8_string_literal(model.english_name);
      o += ",\"download_url\":";
      o += json_utf8_string_literal(model.download_url);
      o += ",\"variants\":";
      o += json_flat_string_array(model.variants);
      o += ",\"default_variant\":";
      o += json_utf8_string_literal(model.default_variant);
      o += "}";
    }
    o += "]}";
    char *buf = malloc_string_copy(o);
    if (buf == nullptr) {
      return MOONSHINE_ERROR_UNKNOWN;
    }
    *out_catalog_json = buf;
    return MOONSHINE_ERROR_NONE;
  } catch (const std::exception &e) {
    LOGF("moonshine_get_embedding_catalog failed: %s\n", e.what());
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
int32_t moonshine_create_grapheme_to_phonemizer_from_files(
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
        "filenames=%p, filenames_count=%" PRIu64
        ", options=%p, options_count=%" PRIu64
        ", "
        "moonshine_version=%d)",
        language != nullptr ? language : "",
        reinterpret_cast<const void *>(filenames), filenames_count,
        static_cast<const void *>(options), options_count, moonshine_version);
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
int32_t moonshine_create_grapheme_to_phonemizer_from_memory(
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
        "filenames_count=%" PRIu64
        ", memory=%p, memory_sizes=%p, options=%p, "
        "options_count=%" PRIu64 ", moonshine_version=%d)",
        language != nullptr ? language : "",
        reinterpret_cast<const void *>(filenames), filenames_count,
        reinterpret_cast<const void *>(memory),
        reinterpret_cast<const void *>(memory_sizes),
        static_cast<const void *>(options), options_count, moonshine_version);
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
void moonshine_free_grapheme_to_phonemizer(
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
int32_t moonshine_text_to_phonemes(int32_t grapheme_to_phonemizer_handle,
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
        static_cast<const void *>(options), options_count,
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