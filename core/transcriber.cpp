#include "transcriber.h"

#include <cstdio>
#include <filesystem>
#include <string>

#include "debug-utils.h"
#include "resampler.h"
#include "string-utils.h"
#include "utf8.h"

#include <iostream>

namespace {
constexpr int32_t INTERNAL_SAMPLE_RATE = 16000;

const int32_t VALID_MODEL_ARCHS[] = {
    MOONSHINE_MODEL_ARCH_TINY,
    MOONSHINE_MODEL_ARCH_BASE,
    MOONSHINE_MODEL_ARCH_TINY_STREAMING,
    MOONSHINE_MODEL_ARCH_BASE_STREAMING,
};

bool is_streaming_model_arch(uint32_t model_arch) {
  return model_arch == MOONSHINE_MODEL_ARCH_TINY_STREAMING ||
         model_arch == MOONSHINE_MODEL_ARCH_BASE_STREAMING;
}

void validate_model_arch(uint32_t model_arch) {
  for (uint32_t valid_model_arch : VALID_MODEL_ARCHS) {
    if (model_arch == valid_model_arch) {
      return;
    }
  }
  throw std::runtime_error("Invalid model architecture: " +
                           std::to_string(model_arch));
}
} // namespace

Transcriber::Transcriber(const TranscriberOptions &options)
    : stt_model(nullptr), streaming_model(nullptr), next_stream_id(1) {
  this->options = options;
  // Start with a random 64-bit value as a unique identifier. We increment
  // this value to generate each new line ID. These should be safe to use as a
  // persistent identifier for every line, since duplicates are so unlikely as
  // to be impossible, assuming std::random_device is sufficiently random.
  std::random_device rd;
  this->next_line_id = (uint64_t)(rd()) << 32 | (uint64_t)(rd());
  const TranscriberOptions::ModelSource model_source = options.model_source;
  if (model_source == TranscriberOptions::ModelSource::FILES) {
    load_from_files(options.model_path, options.model_arch);
  } else if (model_source == TranscriberOptions::ModelSource::MEMORY) {
    load_from_memory(options.encoder_model_data,
                     options.encoder_model_data_size,
                     options.decoder_model_data,
                     options.decoder_model_data_size, options.tokenizer_data,
                     options.tokenizer_data_size, options.model_arch);
  } else if (model_source == TranscriberOptions::ModelSource::NONE) {
    // Both models stay nullptr
  } else {
    throw std::runtime_error("Invalid model source: " +
                             std::to_string((int)(model_source)));
  }
}

void Transcriber::load_from_files(const char *model_path, uint32_t model_arch) {
  if (model_path == nullptr) {
    throw std::runtime_error("Model path is null");
  }
  validate_model_arch(model_arch);
  if (!std::filesystem::exists(model_path)) {
    throw std::runtime_error("Model directory does not exist at path '" +
                             std::string(model_path) + "'");
  }

  std::string tokenizer_path =
      append_path_component(model_path, "tokenizer.bin");
  if (!std::filesystem::exists(tokenizer_path)) {
    throw std::runtime_error(
        "Required tokenizer file does not exist at path '" + tokenizer_path +
        "'");
  }

  if (is_streaming_model_arch(model_arch)) {
    // Streaming model: expects frontend.onnx, encoder.onnx, adapter.onnx,
    // decoder.onnx, and streaming_config.json
    this->streaming_model = new MoonshineStreamingModel();

    std::string frontend_path =
        append_path_component(model_path, "frontend.onnx");
    std::string encoder_path =
        append_path_component(model_path, "encoder.onnx");
    std::string adapter_path =
        append_path_component(model_path, "adapter.onnx");
    std::string decoder_path =
        append_path_component(model_path, "decoder.onnx");
    std::string config_path =
        append_path_component(model_path, "streaming_config.json");

    if (!std::filesystem::exists(frontend_path)) {
      throw std::runtime_error(
          "Required frontend model file does not exist at path '" +
          frontend_path + "'");
    }
    if (!std::filesystem::exists(encoder_path)) {
      throw std::runtime_error(
          "Required encoder model file does not exist at path '" +
          encoder_path + "'");
    }
    if (!std::filesystem::exists(adapter_path)) {
      throw std::runtime_error(
          "Required adapter model file does not exist at path '" +
          adapter_path + "'");
    }
    if (!std::filesystem::exists(decoder_path)) {
      throw std::runtime_error(
          "Required decoder model file does not exist at path '" +
          decoder_path + "'");
    }
    if (!std::filesystem::exists(config_path)) {
      throw std::runtime_error(
          "Required streaming config file does not exist at path '" +
          config_path + "'");
    }

    int32_t load_error = this->streaming_model->load(
        model_path, tokenizer_path.c_str(), model_arch);
    if (load_error != 0) {
      throw std::runtime_error(
          "Failed to load Moonshine streaming models from " +
          std::string(model_path) +
          ". Error code: " + std::to_string(load_error));
    }
  } else {
    // Non-streaming model: expects encoder_model.ort and
    // decoder_model_merged.ort
    this->stt_model = new MoonshineModel();

    std::string encoder_model_path =
        append_path_component(model_path, "encoder_model.ort");
    std::string decoder_model_path =
        append_path_component(model_path, "decoder_model_merged.ort");

    if (!std::filesystem::exists(encoder_model_path)) {
      throw std::runtime_error(
          "Required encoder model file does not exist at path '" +
          encoder_model_path + "'");
    }
    if (!std::filesystem::exists(decoder_model_path)) {
      throw std::runtime_error(
          "Required decoder model file does not exist at path '" +
          decoder_model_path + "'");
    }

    int32_t load_error = this->stt_model->load(
        encoder_model_path.c_str(), decoder_model_path.c_str(),
        tokenizer_path.c_str(), model_arch);
    if (load_error != 0) {
      throw std::runtime_error("Failed to load Moonshine models from " +
                               encoder_model_path + ", " + decoder_model_path +
                               ", " + tokenizer_path +
                               ". Error code: " + std::to_string(load_error));
    }
  }
}

void Transcriber::load_from_memory(const uint8_t *encoder_model_data,
                                   size_t encoder_model_data_size,
                                   const uint8_t *decoder_model_data,
                                   size_t decoder_model_data_size,
                                   const uint8_t *tokenizer_data,
                                   size_t tokenizer_data_size,
                                   uint32_t model_arch) {
  // Note: load_from_memory currently only supports non-streaming models.
  // Streaming models require additional ONNX files (frontend, adapter) and
  // config.
  if (is_streaming_model_arch(model_arch)) {
    throw std::runtime_error(
        "Streaming models cannot be loaded from memory with the current API. "
        "Use load_from_files instead.");
  }

  this->stt_model = new MoonshineModel();
  int32_t load_error = this->stt_model->load_from_memory(
      encoder_model_data, encoder_model_data_size, decoder_model_data,
      decoder_model_data_size, tokenizer_data, tokenizer_data_size, model_arch);
  if (load_error != 0) {
    throw std::runtime_error(
        "Failed to load Moonshine models from memory. Error code: " +
        std::to_string(load_error));
  }
}

Transcriber::~Transcriber() {
  delete this->stt_model;
  delete this->streaming_model;
  for (auto &stream : this->streams) {
    delete stream.second;
  }
  if (this->batch_stream != nullptr) {
    delete this->batch_stream;
  }
}

void Transcriber::transcribe_without_streaming(
    const float *audio_data, uint64_t audio_length, int32_t sample_rate,
    uint32_t /*flags*/, struct transcript_t **out_transcript) {
  std::lock_guard<std::mutex> lock(this->batch_stream_mutex);
  if (this->batch_stream == nullptr) {
    this->batch_stream = new TranscriberStream(
        new VoiceActivityDetector(this->options.vad_threshold), -1,
        this->options.save_input_wav_path);
  }
  if (!this->batch_stream->save_input_wav_path.empty()) {
    this->batch_stream->save_audio_data_to_wav(audio_data, audio_length,
                                               sample_rate);
    this->batch_stream->save_audio_data_to_wav(nullptr, 0, 0);
  }
  TranscriberStream *stream = this->batch_stream;
  std::vector<VoiceActivitySegment> segments;
  {
    std::lock_guard<std::mutex> lock(stream->vad_mutex);
    stream->start();
    stream->vad->process_audio(audio_data, (int32_t)audio_length, sample_rate);
    stream->stop();
    segments = *(stream->vad->get_segments());
  }

  this->update_transcript_from_segments(segments, stream, out_transcript);
}

int32_t Transcriber::create_stream() {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  int32_t stream_id = this->next_stream_id++;
  TranscriberStream *stream = new TranscriberStream(
      new VoiceActivityDetector(this->options.vad_threshold), stream_id,
      this->options.save_input_wav_path);

  this->streams.insert({stream_id, stream});
  return stream_id;
}

void Transcriber::free_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  TranscriberStream *stream = this->streams[stream_id];
  this->streams.erase(stream_id);
  delete stream;
}

void Transcriber::start_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  TranscriberStream *stream = this->streams[stream_id];
  // Starting a stream invalidates any pointers to stream data (audio, strings)
  // that have been returned to the client during prior sessions.
  stream->transcript_output->internal_lines_map.clear();
  stream->transcript_output->ordered_internal_line_ids.clear();
  stream->start();
}

void Transcriber::stop_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  TranscriberStream *stream = this->streams[stream_id];
  stream->stop();
  stream->save_audio_data_to_wav(nullptr, 0, 0);
}

void Transcriber::add_audio_to_stream(int32_t stream_id,
                                      const float *audio_data,
                                      uint64_t audio_length,
                                      int32_t sample_rate) {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  TranscriberStream *stream = this->streams[stream_id];
  if (!stream->vad->is_active()) {
    std::string error_message =
        "Adding new audio for stream with ID " + std::to_string(stream_id) +
        " but VAD is not active. Did you call start_stream()?";
    throw std::runtime_error(error_message);
  }
  stream->add_to_new_audio_buffer(audio_data, audio_length, sample_rate);
}

void Transcriber::transcribe_stream(int32_t stream_id, uint32_t flags,
                                    struct transcript_t **out_transcript) {
  TranscriberStream *stream = nullptr;
  {
    std::lock_guard<std::mutex> lock(this->streams_mutex);
    if (this->streams.find(stream_id) == this->streams.end()) {
      std::string error_message =
          "Stream with ID " + std::to_string(stream_id) + " not found in " +
          std::to_string(this->streams.size()) + " streams: ";
      for (const auto &stream : this->streams) {
        char addr_str[32];
        snprintf(addr_str, sizeof(addr_str), "%p", (void *)stream.second);
        error_message += "ID: " + std::to_string(stream.first) +
                         ", Address: " + std::string(addr_str) + "\n";
      }
      throw std::runtime_error(error_message);
    }
  }

  stream = this->streams[stream_id];
  if (stream == nullptr) {
    std::string error_message =
        "Stream with ID " + std::to_string(stream_id) + " is null";
    throw std::runtime_error(error_message);
  }

  const float *audio_data = stream->new_audio_buffer.data();
  const uint64_t audio_length = stream->new_audio_buffer.size();
  const bool has_new_audio = (audio_length > 0);
  const float new_audio_duration = audio_length / (float)(INTERNAL_SAMPLE_RATE);
  const bool long_enough_to_analyze =
      new_audio_duration >= this->options.transcription_interval;
  const bool force_update = flags & MOONSHINE_FLAG_FORCE_UPDATE;
  const bool should_update =
      (long_enough_to_analyze || force_update) && has_new_audio;
  const bool is_stopped = !stream->vad->is_active();
  // Return the cached transcript if it's only been a short time since the
  // last transcription.
  if (!should_update) {
    stream->transcript_output->clear_update_flags();
    // Ensure that all lines are marked as complete if the stream is stopped.
    if (is_stopped) {
      stream->transcript_output->mark_all_lines_as_complete();
    }
    *out_transcript = &(stream->transcript_output->transcript);
    return;
  }

  // Use VAD to segment audio
  std::vector<VoiceActivitySegment> segments;
  {
    std::lock_guard<std::mutex> lock(stream->vad_mutex);
    stream->vad->process_audio(audio_data, (int32_t)audio_length,
                               INTERNAL_SAMPLE_RATE);
    segments = *(stream->vad->get_segments());
  }
  stream->clear_new_audio_buffer();
  this->update_transcript_from_segments(segments, stream, out_transcript);
}

std::string
Transcriber::transcript_to_string(const struct transcript_t *transcript) {
  std::string result;
  result += std::to_string(transcript->line_count) + " lines\n";
  for (size_t i = 0; i < transcript->line_count; i++) {
    const struct transcript_line_t &line = transcript->lines[i];
    char time_str[32];
    snprintf(time_str, sizeof(time_str), "%.1fs: ", line.start_time);
    result += time_str;
    if (line.text == nullptr) {
      result += "<null>\n";
    } else {
      result += std::string(line.text);
      result += "\n";
    }
  }
  return result;
}

std::string
Transcriber::transcript_line_to_string(const struct transcript_line_t *line) {
  std::string result;
  result += "text: '" +
            (line->text == nullptr ? std::string("<null>")
                                   : std::string(line->text)) +
            "'";
  result += ", audio_data_count: " + std::to_string(line->audio_data_count);
  char time_str[64];
  snprintf(time_str, sizeof(time_str), ", start_time: %.2fs", line->start_time);
  result += time_str;
  snprintf(time_str, sizeof(time_str), ", duration: %.2fs", line->duration);
  result += time_str;
  result += ", is_complete: " + std::to_string(line->is_complete);
  result += ", is_updated: " + std::to_string(line->is_updated);
  result += ", is_new: " + std::to_string(line->is_new);
  result += ", has_text_changed: " + std::to_string(line->has_text_changed);
  result += ", id: " + std::to_string(line->id);
  return result;
}

void Transcriber::update_transcript_from_segments(
    const std::vector<VoiceActivitySegment> &segments,
    TranscriberStream *stream, struct transcript_t **out_transcript) {
  stream->transcript_output->clear_update_flags();
  for (size_t segment_index = 0; segment_index < segments.size();
       segment_index++) {
    const VoiceActivitySegment &segment = segments[segment_index];
    if (!segment.just_updated) {
      continue;
    }
    TranscriberLine line;
    line.start_time = segment.start_time;
    line.duration = segment.end_time - segment.start_time;
    line.is_complete = segment.is_complete;
    line.just_updated = segment.just_updated;
    if (segment_index >=
        stream->transcript_output->ordered_internal_line_ids.size()) {
      uint64_t new_segment_id = this->next_line_id.fetch_add(1);
      stream->transcript_output->ordered_internal_line_ids.push_back(
          new_segment_id);
    }
    line.id =
        stream->transcript_output->ordered_internal_line_ids.at(segment_index);

    std::chrono::steady_clock::time_point start_time =
        std::chrono::steady_clock::now();
    // Transcribe the segment using the appropriate model
    if (is_streaming_model_arch(this->options.model_arch) &&
        this->streaming_model != nullptr) {
      // Use streaming model for transcription
      line.text = transcribe_segment_with_streaming_model(
          segment.audio_data.data(), segment.audio_data.size());
    } else if (this->stt_model != nullptr) {
      // Use non-streaming model for transcription
      std::lock_guard<std::mutex> lock(this->stt_model_mutex);
      char *out_text = nullptr;
      int transcribe_error = this->stt_model->transcribe(
          segment.audio_data.data(), segment.audio_data.size(), &out_text);
      if (transcribe_error != 0) {
        LOGF("Failed to transcribe: %d", transcribe_error);
        throw std::runtime_error("Failed to transcribe: " +
                                 std::to_string(transcribe_error));
      }
      // Ensure the output text is valid UTF-8.
      line.text = sanitize_text(out_text);
    } else {
      // No model available - return audio data and segment info only
      line.text = nullptr;
    }
    std::chrono::steady_clock::time_point end_time =
        std::chrono::steady_clock::now();
    line.last_transcription_latency_ms =
        (uint32_t)(std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count());
    line.audio_data = segment.audio_data;
    stream->transcript_output->add_or_update_line(line);
  }
  const bool is_stopped = !stream->vad->is_active();
  if (is_stopped) {
    stream->transcript_output->mark_all_lines_as_complete();
  }
  stream->transcript_output->update_transcript_from_lines();
  *out_transcript = &(stream->transcript_output->transcript);
}

std::string *
Transcriber::transcribe_segment_with_streaming_model(const float *audio_data,
                                                     size_t audio_length) {
  if (audio_length == 0 || this->streaming_model == nullptr) {
    return new std::string();
  }

  const MoonshineStreamingConfig &config = this->streaming_model->config;

  // Create a temporary state for this segment transcription
  MoonshineStreamingState state;
  state.reset(config);

  // Process audio in chunks through the streaming model's frontend
  const int chunk_size = 1280; // 80ms at 16kHz
  {
    std::lock_guard<std::mutex> lock(this->streaming_model_mutex);

    for (size_t offset = 0; offset < audio_length; offset += chunk_size) {
      int len = static_cast<int>(
          std::min(static_cast<size_t>(chunk_size), audio_length - offset));
      int err = this->streaming_model->process_audio_chunk(
          &state, audio_data + offset, len, nullptr);
      if (err != 0) {
        LOGF("Failed to process audio chunk: %d", err);
        throw std::runtime_error("Failed to process audio chunk: " +
                                 std::to_string(err));
      }
    }

    // Run encoder (final - this is the complete segment)
    int new_frames = 0;
    int err = this->streaming_model->encode(&state, true, &new_frames);
    if (err != 0) {
      LOGF("Failed to encode: %d", err);
      throw std::runtime_error("Failed to encode: " + std::to_string(err));
    }
  }

  // If no memory accumulated, return empty string
  if (state.memory_len == 0) {
    return new std::string();
  }

  // Decode to get transcription
  const int max_tokens = 256;
  std::vector<int64_t> tokens;
  tokens.push_back(config.bos_id);

  std::vector<float> logits(config.vocab_size);
  int current_token = config.bos_id;

  {
    std::lock_guard<std::mutex> lock(this->streaming_model_mutex);

    for (int step = 0; step < max_tokens; ++step) {
      int err = this->streaming_model->decode_step(&state, current_token,
                                                   logits.data());
      if (err != 0) {
        break;
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

      if (next_token == config.eos_id)
        break;
    }
  }

  // Convert tokens to text
  std::string text = this->streaming_model->tokens_to_text(tokens);
  return sanitize_text(text.c_str());
}

std::string *Transcriber::sanitize_text(const char *text) {
  std::string text_string(text);
  std::string *result = new std::string();
  size_t i = 0;
  while (i < text_string.size()) {
    const size_t remaining = text_string.size() - i;
    const uint8_t c = text_string.at(i);
    if (c < 0x80) {
      // ASCII - always valid, but watch for embedded nulls in Modified UTF-8
      result->push_back(c);
      i++;
    } else if ((c & 0xE0) == 0xC0) {
      // 2-byte sequence
      if (remaining < 2 || (text_string.at(i + 1) & 0xC0) != 0x80) {
        result->push_back('?');
        i++;
      } else {
        result->push_back(text_string.at(i));
        result->push_back(text_string.at(i + 1));
        i += 2;
      }
    } else if ((c & 0xF0) == 0xE0) {
      // 3-byte sequence
      if (remaining < 3 || (text_string.at(i + 1) & 0xC0) != 0x80 ||
          (text_string.at(i + 2) & 0xC0) != 0x80) {
        result->push_back('?');
        i++;
      } else {
        result->push_back(text_string.at(i));
        result->push_back(text_string.at(i + 1));
        result->push_back(text_string.at(i + 2));
        i += 3;
      }
    } else if ((c & 0xF8) == 0xF0) {
      // 4-byte sequence
      if (remaining < 4 || (text_string.at(i + 1) & 0xC0) != 0x80 ||
          (text_string.at(i + 2) & 0xC0) != 0x80 ||
          (text_string.at(i + 3) & 0xC0) != 0x80) {
        result->push_back('?');
        i++;
      } else {
        result->push_back(text_string.at(i));
        result->push_back(text_string.at(i + 1));
        result->push_back(text_string.at(i + 2));
        result->push_back(text_string.at(i + 3));
        i += 4;
      }
    } else {
      // Invalid start byte (0x80-0xBF, 0xF5-0xFF)
      result->push_back('?');
      i++;
    }
  }
  return result;
}

TranscriberLine::TranscriberLine() {
  this->text = nullptr;
  this->audio_data = std::vector<float>();
  this->start_time = 0.0f;
  this->duration = 0.0f;
  this->is_complete = false;
  this->just_updated = false;
  this->is_new = false;
  this->has_text_changed = false;
  this->id = 0;
  this->last_transcription_latency_ms = 0;
}

TranscriberLine::TranscriberLine(const TranscriberLine &other) {
  this->text = other.text == nullptr ? nullptr : new std::string(*other.text);
  this->audio_data = other.audio_data;
  this->start_time = other.start_time;
  this->duration = other.duration;
  this->is_complete = other.is_complete;
  this->just_updated = other.just_updated;
  this->is_new = other.is_new;
  this->has_text_changed = other.has_text_changed;
  this->id = other.id;
  this->last_transcription_latency_ms = other.last_transcription_latency_ms;
}

TranscriberLine &TranscriberLine::operator=(const TranscriberLine &other) {
  this->text = other.text == nullptr ? nullptr : new std::string(*other.text);
  this->audio_data = other.audio_data;
  this->start_time = other.start_time;
  this->duration = other.duration;
  this->is_complete = other.is_complete;
  this->just_updated = other.just_updated;
  this->is_new = other.is_new;
  this->has_text_changed = other.has_text_changed;
  this->id = other.id;
  this->last_transcription_latency_ms = other.last_transcription_latency_ms;
  return *this;
}

TranscriberLine::~TranscriberLine() { delete this->text; }

std::string TranscriberLine::to_string() const {
  return "TranscriberLine(start_time=" + std::to_string(start_time) +
         ", text='" + (text == nullptr ? "<null>" : *text) + "'" +
         ", duration=" + std::to_string(duration) +
         ", is_complete=" + std::to_string(is_complete) +
         ", just_updated=" + std::to_string(just_updated) +
         ", is_new=" + std::to_string(is_new) +
         ", has_text_changed=" + std::to_string(has_text_changed) +
         ", id=" + std::to_string(id) + ", last_transcription_latency_ms=" +
         std::to_string(last_transcription_latency_ms) + ")";
}

void TranscriptStreamOutput::add_or_update_line(TranscriberLine &line) {
  if (internal_lines_map.find(line.id) != internal_lines_map.end()) {
    line.is_new = false;
    TranscriberLine *existing_line = &this->internal_lines_map[line.id];
    line.has_text_changed =
        (existing_line->text == nullptr && line.text != nullptr) ||
        (existing_line->text != nullptr && line.text == nullptr) ||
        (existing_line->text != nullptr && line.text != nullptr &&
         *existing_line->text != *line.text);
  } else {
    line.is_new = true;
    line.has_text_changed = line.text != nullptr;
  }
  this->internal_lines_map[line.id] = line;
}

void TranscriptStreamOutput::update_transcript_from_lines() {
  this->output_lines.clear();
  for (const uint64_t &line_id : this->ordered_internal_line_ids) {
    const TranscriberLine &line = this->internal_lines_map[line_id];
    this->output_lines.push_back({
        .text = line.text == nullptr ? nullptr : line.text->c_str(),
        .audio_data = line.audio_data.data(),
        .audio_data_count = line.audio_data.size(),
        .start_time = line.start_time,
        .duration = line.duration,
        .id = line.id,
        .is_complete = line.is_complete,
        .is_updated = line.just_updated,
        .is_new = line.is_new,
        .has_text_changed = line.has_text_changed,
        .last_transcription_latency_ms = line.last_transcription_latency_ms,
    });
  }
  this->transcript.lines = this->output_lines.data();
  this->transcript.line_count = (uint64_t)(this->output_lines.size());
}

void TranscriptStreamOutput::clear_update_flags() {
  for (const uint64_t &line_id : this->ordered_internal_line_ids) {
    TranscriberLine &line = this->internal_lines_map.at(line_id);
    line.just_updated = false;
    line.is_new = false;
    line.has_text_changed = false;
  }
  for (transcript_line_t &line : this->output_lines) {
    line.is_updated = 0;
    line.has_text_changed = 0;
    line.is_new = 0;
  }
}

void TranscriptStreamOutput::mark_all_lines_as_complete() {
  for (const uint64_t &line_id : this->ordered_internal_line_ids) {
    TranscriberLine &line = this->internal_lines_map[line_id];
    if (!line.is_complete) {
      line.is_complete = 1;
      line.just_updated = 1;
    }
  }
  this->update_transcript_from_lines();
}

TranscriberStream::TranscriberStream(VoiceActivityDetector *vad,
                                     int32_t stream_id,
                                     const std::string &save_input_wav_path)
    : vad(vad), transcript_output(new TranscriptStreamOutput()),
      save_input_wav_path(save_input_wav_path), stream_id(stream_id) {
  if (!this->save_input_wav_path.empty()) {
    std::filesystem::create_directory(this->save_input_wav_path);
    std::string wav_path = append_path_component(this->save_input_wav_path,
                                                 this->get_wav_filename());
    std::filesystem::remove(wav_path);
  }
}

void TranscriberStream::start() {
  this->vad->start();
  this->transcript_output->internal_lines_map.clear();
  this->transcript_output->ordered_internal_line_ids.clear();
}

void TranscriberStream::stop() { this->vad->stop(); }

std::string TranscriberStream::get_wav_filename() {
  if (this->stream_id == -1) {
    return "input_batch.wav";
  } else {
    return std::string("input_") + std::to_string(this->stream_id) +
           std::string(".wav");
  }
}

void TranscriberStream::save_audio_data_to_wav(const float *audio_data,
                                               uint64_t audio_length,
                                               int32_t sample_rate) {
  if (this->save_input_wav_path.empty()) {
    return;
  }
  const size_t previous_second = save_input_data.size() / INTERNAL_SAMPLE_RATE;
  if (audio_data != nullptr) {
    save_input_data.insert(save_input_data.end(), audio_data,
                           audio_data + audio_length);
    last_save_sample_rate = sample_rate;
  }
  const size_t current_second = save_input_data.size() / INTERNAL_SAMPLE_RATE;
  // Only save every second to avoid too much latency overhead.
  if (current_second != previous_second || audio_data == nullptr) {
    std::string wav_path = append_path_component(this->save_input_wav_path,
                                                 this->get_wav_filename());
    save_wav_data(wav_path.c_str(), save_input_data.data(),
                  save_input_data.size(), last_save_sample_rate);
  }
}

void TranscriberStream::add_to_new_audio_buffer(const float *audio_data,
                                                uint64_t audio_length,
                                                int32_t sample_rate) {
  this->save_audio_data_to_wav(audio_data, audio_length, sample_rate);
  std::vector<float> audio_vector(audio_data, audio_data + audio_length);
  std::vector<float> resampled_audio =
      resample_audio(audio_vector, sample_rate, INTERNAL_SAMPLE_RATE);
  this->new_audio_buffer.insert(this->new_audio_buffer.end(),
                                resampled_audio.begin(), resampled_audio.end());
}

void TranscriberStream::clear_new_audio_buffer() {
  this->new_audio_buffer.clear();
}