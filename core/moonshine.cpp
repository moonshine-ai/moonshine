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

#include "moonshine.h"

#include <cassert>
#include <cctype>
#include <cerrno>
#include <cerrno> // For errno
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstring> // For strerror

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <map>
#include <numeric>
#include <vector>

#include "bin-tokenizer.h"
#include "moonshine-model.h"
#include "moonshine-ort-allocator.h"
#include "moonshine-tensor-view.h"
#include "string-utils.h"
#include "transcriber.h"

#define DEBUG_ALLOC_ENABLED 1
#include "debug-utils.h"
#include "ort-utils.h"

#define CHECK_MOONSHINE_HANDLE(handle)                                         \
  do {                                                                         \
    if (handle < 0 || handle >= static_cast<int32_t>(models.size())) {         \
      fprintf(stderr, "Moonshine model failed to load\n");                     \
      return -1;                                                               \
    }                                                                          \
    if (models[handle] == nullptr) {                                           \
      fprintf(stderr, "Moonshine model has been freed\n");                     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

std::vector<MoonshineModel *> models;

extern "C" moonshine_handle_t
moonshine_load_model(const char *encoder_model_path,
                     const char *decoder_model_path, const char *tokenizer_path,
                     int32_t model_type) {
  if (models.size() >= INT32_MAX) {
    LOG("Too many models loaded\n");
    return -1;
  }
  int load_error = -1;
  MoonshineModel *model = nullptr;
  try {
    model = new MoonshineModel();
    load_error = model->load(encoder_model_path, decoder_model_path,
                             tokenizer_path, model_type);
  } catch (const std::exception &e) {
    LOGF("Failed to load model: %s\n", e.what());
    return -1;
  }
  if (load_error != 0) {
    LOG("Failed to load model");
    return -1;
  }
  int32_t model_index = -1;
  for (size_t i = 0; i < models.size(); i++) {
    if (models[i] == nullptr) {
      models[i] = model;
      model_index = i;
      break;
    }
  }
  if (model_index == -1) {
    model_index = models.size();
    models.push_back(model);
  }
  return model_index;
}

#if defined(ANDROID)
extern "C" moonshine_handle_t
moonshine_load_model_from_assets(const char *encoder_model_path,
                                 const char *decoder_model_path,
                                 const char *tokenizer_path, int32_t model_type,
                                 AAssetManager *assetManager) {
  if (models.size() >= INT32_MAX) {
    LOG("Too many models loaded\n");
    return -1;
  }
  int load_error = -1;
  MoonshineModel *model = nullptr;
  try {
    model = new MoonshineModel();
    load_error =
        model->load_from_assets(encoder_model_path, decoder_model_path,
                                tokenizer_path, model_type, assetManager);
  } catch (const std::exception &e) {
    LOGF("Failed to load model: %s", e.what());
    return -1;
  }
  if (load_error != 0) {
    LOG("Failed to load model");
    return -1;
  }
  int32_t model_index = -1;
  for (size_t i = 0; i < models.size(); i++) {
    if (models[i] == nullptr) {
      models[i] = model;
      model_index = i;
      break;
    }
  }
  if (model_index == -1) {
    model_index = models.size();
    models.push_back(model);
  }
  return model_index;
}
#endif

extern "C" int moonshine_transcribe(moonshine_handle_t handle,
                                    const float *audio_data,
                                    size_t audio_data_size, char **out_text) {
  CHECK_MOONSHINE_HANDLE(handle);
  try {
    return models[handle]->transcribe(audio_data, audio_data_size, out_text);
  } catch (const std::exception &e) {
    fprintf(stderr, "Failed to transcribe: %s\n", e.what());
    return -1;
  }
}

extern "C" int moonshine_transcribe_wav(moonshine_handle_t handle,
                                        const char *wav_path, char **out_text) {
  CHECK_MOONSHINE_HANDLE(handle);
  try {
    return models[handle]->transcribe_wav(wav_path, out_text);
  } catch (const std::exception &e) {
    LOGF("Failed to transcribe WAV: %s\n", e.what());
    return -1;
  }
}

/* Releases all resources allocated by moonshine_load_model. */
extern "C" void moonshine_free_model(moonshine_handle_t handle) {
  if (handle < 0 || handle >= static_cast<int32_t>(models.size())) {
    LOGF("Moonshine free called with invalid handle %d\n", handle);
    return;
  }
  delete models[handle];
  models[handle] = nullptr;
}

extern "C" void moonshine_free_text(char *text) { DEBUG_FREE(text); }

// v2 of the C API.

#include <mutex>

#define CHECK_TRANSCRIBER_HANDLE(handle)                                       \
  do {                                                                         \
    if (handle < 0 || !transcriber_map.contains(handle)) {                     \
      LOGF("Moonshine transcriber handle is invalid: handle %d", handle);      \
      return MOONSHINE_ERROR_INVALID_HANDLE;                                   \
    }                                                                          \
  } while (0)

#define RETURN_IF_ERROR(error)                                                 \
  do {                                                                         \
    const int32_t error_code = (error);                                        \
    if (error_code != 0) {                                                     \
      return error_code;                                                       \
    }                                                                          \
  } while (0)

namespace {
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
    } else {
      throw std::runtime_error(
          "Unknown transcriber option: '" + std::string(in_option.name) + "'");
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

} // namespace

extern "C" int32_t moonshine_get_version(void) {
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
    uint64_t options_count, int32_t /* moonshine_version */) {
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
    uint64_t options_count, int32_t /* moonshine_version */) {
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
  free_transcriber_handle(transcriber_handle);
}

int32_t moonshine_transcribe_without_streaming(
    int32_t transcriber_handle, float *audio_data, uint64_t audio_length,
    int32_t sample_rate, uint32_t flags, struct transcript_t **out_transcript) {
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

int32_t moonshine_create_stream(int32_t transcriber_handle, uint32_t) {
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    return transcriber_map[transcriber_handle]->create_stream();
  } catch (const std::exception &e) {
    LOGF("Failed to create stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
}

int moonshine_free_stream(int32_t transcriber_handle, int32_t stream_handle) {
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
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->stop_stream(stream_handle);
  } catch (const std::exception &e) {
    LOGF("Failed to stop stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}

const char *
moonshine_transcript_to_string(const struct transcript_t *transcript) {
  static std::string description;
  description = Transcriber::transcript_to_string(transcript);
  return description.c_str();
}

int32_t moonshine_transcribe_add_audio_to_stream(int32_t transcriber_handle,
                                                 int32_t stream_handle,
                                                 const float *new_audio_data,
                                                 uint64_t audio_length,
                                                 int32_t sample_rate,
                                                 uint32_t /*flags*/) {
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
  CHECK_TRANSCRIBER_HANDLE(transcriber_handle);
  try {
    transcriber_map[transcriber_handle]->transcribe_stream(
        stream_handle, flags, out_transcript);
  } catch (const std::exception &e) {
    LOGF("Failed to transcribe stream: %s\n", e.what());
    return MOONSHINE_ERROR_UNKNOWN;
  }
  return MOONSHINE_ERROR_NONE;
}