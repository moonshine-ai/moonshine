#include "transcriber.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <string>

#include "debug-utils.h"
#include "moonshine-c-api.h"
#include "ort-utils.h"
#include "resampler.h"
#include "string-utils.h"
#include "utf8.h"

namespace {
constexpr int32_t INTERNAL_SAMPLE_RATE = 16000;

const int32_t VALID_MODEL_ARCHS[] = {
    MOONSHINE_MODEL_ARCH_TINY,
    MOONSHINE_MODEL_ARCH_BASE,
    MOONSHINE_MODEL_ARCH_TINY_STREAMING,
    MOONSHINE_MODEL_ARCH_BASE_STREAMING,
    MOONSHINE_MODEL_ARCH_SMALL_STREAMING,
    MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING,
};

bool is_streaming_model_arch(uint32_t model_arch) {
  return model_arch == MOONSHINE_MODEL_ARCH_TINY_STREAMING ||
         model_arch == MOONSHINE_MODEL_ARCH_BASE_STREAMING ||
         model_arch == MOONSHINE_MODEL_ARCH_SMALL_STREAMING ||
         model_arch == MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;
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

int32_t vad_window_size_from_duration(float duration, int32_t hop_size) {
  return static_cast<int32_t>(
      std::ceil((duration * INTERNAL_SAMPLE_RATE) / hop_size));
}

size_t vad_sample_count_from_duration(float duration) {
  return static_cast<size_t>(std::round(duration * INTERNAL_SAMPLE_RATE));
}
}  // namespace

Transcriber::Transcriber(const TranscriberOptions &options)
    : stt_model(nullptr),
      streaming_model(nullptr),
      speaker_diarizer(nullptr),
      next_stream_id(1) {
  this->options = options;
  // Speaker-to-text mapping needs per-word timings, so turn on word timestamps
  // whenever diarization is requested (even if the caller did not set the
  // word_timestamps option explicitly).
  if (this->options.identify_speakers) {
    this->options.word_timestamps = true;
  }
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
  if (options.identify_speakers) {
    SpeakerDiarizerOptions diarizer_options;
    diarizer_options.cluster_cadence = this->options.diarization_cluster_cadence;
    diarizer_options.analyze_cadence = this->options.diarization_analyze_cadence;
    diarizer_options.cluster_window_sec =
        this->options.diarization_cluster_window_sec;
    this->speaker_diarizer = new SpeakerDiarizer(diarizer_options);
  }
  // Lazily attach the spelling model when the caller provided one.
  // We deliberately don't fall back to a built-in: the model weights
  // are language-specific, so the C API leaves the choice to the
  // caller (Python downloads it, native callers can ship an .ort).
  const bool has_spelling_buffer = options.spelling_model_data != nullptr &&
                                    options.spelling_model_data_size > 0;
  const bool has_spelling_path = !options.spelling_model_path.empty();
  if (has_spelling_buffer || has_spelling_path) {
    this->spelling_model = new SpellingModel(this->options.log_ort_run,
                                             this->options.ort_provider_names,
                                             this->options.coreml_cache_dir);
    int load_error = 0;
    if (has_spelling_buffer) {
      load_error = this->spelling_model->load_from_memory(
          options.spelling_model_data, options.spelling_model_data_size);
    } else {
      load_error = this->spelling_model->load(options.spelling_model_path.c_str());
    }
    if (load_error != 0) {
      delete this->spelling_model;
      this->spelling_model = nullptr;
      throw std::runtime_error(
          "Failed to load spelling model. Error code: " +
          std::to_string(load_error));
    }
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
    this->streaming_model = new MoonshineStreamingModel(
        this->options.log_ort_run, this->options.ort_provider_names,
        this->options.coreml_cache_dir);

    int32_t load_error = this->streaming_model->load(
        model_path, tokenizer_path.c_str(), model_arch);
    if (load_error != 0) {
      throw std::runtime_error(
          "Failed to load Moonshine streaming models from " +
          std::string(model_path) +
          ". Error code: " + std::to_string(load_error));
    }
    const MoonshineStreamingConfig &config = this->streaming_model->config;
    this->streaming_state.reset(config);

    // Load attention-enabled streaming decoder if word timestamps requested
    if (this->options.word_timestamps) {
      std::string decoder_attn_path =
          append_path_component(model_path, "decoder_kv_with_attention.ort");
      if (std::filesystem::exists(decoder_attn_path)) {
        // Replace the streaming decoder with the attention-enabled version
        if (this->streaming_model->decoder_kv_session) {
          this->streaming_model->ort_api->ReleaseSession(
              this->streaming_model->decoder_kv_session);
        }
        this->streaming_model->decoder_kv_session = nullptr;
        const char *dec_mmapped = nullptr;
        size_t dec_mmapped_size = 0;
        int32_t dec_err = ort_session_from_path(
            this->streaming_model->ort_api, this->streaming_model->ort_env,
            this->streaming_model->ort_session_options,
            decoder_attn_path.c_str(),
            &this->streaming_model->decoder_kv_session, &dec_mmapped,
            &dec_mmapped_size);
        if (dec_err != 0) {
          LOGF("Warning: Failed to load decoder_kv_with_attention from %s\n",
               decoder_attn_path.c_str());
        }
      }
    }
  } else {
    // Non-streaming model: expects encoder_model.ort and
    // decoder_model_merged.ort
    this->stt_model = new MoonshineModel(this->options.log_ort_run,
                                         this->options.max_tokens_per_second,
                                         this->options.ort_provider_names,
                                         this->options.coreml_cache_dir);

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

    // Load word timestamp model if enabled.
    // Try decoder_with_attention.ort first (single-pass, replaces decoder
    // with one that outputs cross-attention weights during decoding).
    // Fall back to alignment_model.ort (two-pass, runs alignment after
    // transcription using a separate teacher-forced decoder pass).
    if (this->options.word_timestamps) {
      std::string decoder_attn_path =
          append_path_component(model_path, "decoder_with_attention.ort");
      std::string alignment_path =
          append_path_component(model_path, "alignment_model.ort");

      if (std::filesystem::exists(decoder_attn_path)) {
        // Single-pass: replace decoder with attention-enabled version
        if (this->stt_model->decoder_session) {
          this->stt_model->ort_api->ReleaseSession(
              this->stt_model->decoder_session);
        }
        this->stt_model->decoder_session = nullptr;
        const char *dec_mmapped = nullptr;
        size_t dec_mmapped_size = 0;
        int32_t dec_err = ort_session_from_path(
            this->stt_model->ort_api, this->stt_model->ort_env,
            this->stt_model->ort_session_options,
            decoder_attn_path.c_str(), &this->stt_model->decoder_session,
            &dec_mmapped, &dec_mmapped_size);
        if (dec_err != 0) {
          LOGF("Warning: Failed to load decoder_with_attention from %s\n",
               decoder_attn_path.c_str());
        }
      } else if (std::filesystem::exists(alignment_path)) {
        // Two-pass fallback: separate alignment model
        int32_t align_err =
            this->stt_model->load_alignment_model(alignment_path.c_str());
        if (align_err != 0) {
          LOGF("Warning: Failed to load alignment model from %s\n",
               alignment_path.c_str());
        }
      } else {
        LOG("Warning: No word timestamp model found, word timestamps "
            "disabled\n");
      }
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

  this->stt_model = new MoonshineModel(this->options.log_ort_run,
                                       this->options.max_tokens_per_second,
                                       this->options.ort_provider_names,
                                       this->options.coreml_cache_dir);
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
  delete this->speaker_diarizer;
  delete this->spelling_model;
  for (auto &stream : this->streams) {
    delete stream.second;
  }
  if (this->batch_stream != nullptr) {
    delete this->batch_stream;
  }
}

void Transcriber::transcribe_without_streaming(
    const float *audio_data, uint64_t audio_length, int32_t sample_rate,
    uint32_t flags, struct transcript_t **out_transcript) {
  std::lock_guard<std::mutex> lock(this->batch_stream_mutex);
  if (this->batch_stream == nullptr) {
    const int32_t vad_window_size = vad_window_size_from_duration(
        this->options.vad_window_duration, this->options.vad_hop_size);
    const size_t vad_max_segment_sample_count =
        vad_sample_count_from_duration(this->options.vad_max_segment_duration);
    this->batch_stream = new TranscriberStream(
        new VoiceActivityDetector(this->options.vad_threshold, vad_window_size,
                                  this->options.vad_hop_size,
                                  this->options.vad_look_behind_sample_count,
                                  vad_max_segment_sample_count),
        -1, this->options.save_input_wav_path);
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

  this->update_transcript_from_segments(segments, stream, flags, out_transcript);

  if (this->speaker_diarizer != nullptr) {
    const std::vector<SpeakerTurn> turns =
        this->speaker_diarizer->diarize(audio_data, audio_length, sample_rate);
    apply_speaker_turns_to_lines(turns, stream->transcript_output);
    stream->transcript_output->update_transcript_from_lines();
  }
}

int32_t Transcriber::create_stream() {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  int32_t stream_id = this->next_stream_id++;
  const int32_t vad_window_size = vad_window_size_from_duration(
      this->options.vad_window_duration, this->options.vad_hop_size);
  const size_t vad_max_segment_sample_count =
      vad_sample_count_from_duration(this->options.vad_max_segment_duration);
  TranscriberStream *stream = new TranscriberStream(
      new VoiceActivityDetector(this->options.vad_threshold, vad_window_size,
                                this->options.vad_hop_size,
                                this->options.vad_look_behind_sample_count,
                                vad_max_segment_sample_count),
      stream_id, this->options.save_input_wav_path);
  if (this->speaker_diarizer != nullptr) {
    stream->diarizer_stream_id = this->speaker_diarizer->create_stream();
  }

  this->streams.insert({stream_id, stream});
  return stream_id;
}

void Transcriber::free_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  TranscriberStream *stream = this->streams[stream_id];
  this->streams.erase(stream_id);
  if (this->speaker_diarizer != nullptr && stream->diarizer_stream_id >= 0) {
    this->speaker_diarizer->free_stream(stream->diarizer_stream_id);
  }
  delete stream;
}

void Transcriber::start_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  TranscriberStream *stream = this->streams[stream_id];
  // Starting a stream invalidates any pointers to stream data (audio, strings)
  // that have been returned to the client during prior sessions.
  {
    std::lock_guard<std::mutex> output_lock(stream->transcript_output->mutex);
    stream->transcript_output->internal_lines_map.clear();
    stream->transcript_output->ordered_internal_line_ids.clear();
    // Reset to an empty transcript.
    stream->transcript_output->transcript.lines = nullptr;
    stream->transcript_output->transcript.line_count = 0;
  }
  stream->start();
  if (this->speaker_diarizer != nullptr && stream->diarizer_stream_id >= 0) {
    this->speaker_diarizer->start_stream(stream->diarizer_stream_id);
  }
}

void Transcriber::stop_stream(int32_t stream_id) {
  std::lock_guard<std::mutex> lock(this->streams_mutex);
  TranscriberStream *stream = this->streams[stream_id];
  stream->stop();
  stream->save_audio_data_to_wav(nullptr, 0, 0);
  if (this->speaker_diarizer != nullptr && stream->diarizer_stream_id >= 0) {
    // Run a final clustering pass so the next transcribe_stream call picks up
    // the finalized speaker spans.
    this->speaker_diarizer->finish_stream(stream->diarizer_stream_id);
  }
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
  const bool diarization_enabled =
      (this->speaker_diarizer != nullptr && stream->diarizer_stream_id >= 0);
  // Return the cached transcript if it's only been a short time since the
  // last transcription.
  if (!should_update) {
    stream->transcript_output->clear_update_flags();
    // Speaker spans may still have been revised since the last call (for
    // example by the final clustering pass in stop_stream), so pick up the
    // latest turns even when the transcription itself is unchanged.
    bool speakers_changed = false;
    if (diarization_enabled) {
      const std::vector<SpeakerTurn> turns =
          this->speaker_diarizer->get_turns(stream->diarizer_stream_id);
      speakers_changed =
          apply_speaker_turns_to_lines(turns, stream->transcript_output);
    }
    // Ensure that all lines are marked as complete if the stream is stopped.
    if (is_stopped) {
      stream->transcript_output->mark_all_lines_as_complete();
    }
    if (speakers_changed) {
      stream->transcript_output->update_transcript_from_lines();
    }
    *out_transcript = &(stream->transcript_output->transcript);
    return;
  }

  // Feed the new audio to the diarizer before it's consumed. This runs the
  // segmentation/embedding models on new analysis chunks and re-clusters on
  // the configured cadence, which is the main cost of identify_speakers.
  if (diarization_enabled) {
    this->speaker_diarizer->add_audio_to_stream(
        stream->diarizer_stream_id, audio_data, audio_length,
        INTERNAL_SAMPLE_RATE);
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
  this->update_transcript_from_segments(segments, stream, flags, out_transcript);

  if (diarization_enabled) {
    const std::vector<SpeakerTurn> turns =
        this->speaker_diarizer->get_turns(stream->diarizer_stream_id);
    if (apply_speaker_turns_to_lines(turns, stream->transcript_output)) {
      stream->transcript_output->update_transcript_from_lines();
    }
  }
}

std::string Transcriber::transcript_to_string(
    const struct transcript_t *transcript) {
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

std::string Transcriber::transcript_line_to_string(
    const struct transcript_line_t *line) {
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

bool Transcriber::apply_spelling_fusion(TranscriberLine &line) {
  // Always run the matcher: command words like "stop" / "clear" /
  // "delete" rely on the matcher even when the .ort model isn't
  // loaded. The matcher's STOPPED / CLEAR / UNDO results are surfaced
  // through ``line.text`` only when fusion produces an actual
  // character; non-character matches leave the original ASR text
  // intact so higher-level Python code can still classify them.
  if (line.text == nullptr) return false;
  const std::string raw_text = *line.text;
  SpellingMatch match = this->spelling_matcher.classify(raw_text);

  // Only run the spelling-CNN when we have audio (it's a 1-second
  // 16kHz waveform model, so empty / too-short clips just produce
  // garbage). The fuser handles a null prediction gracefully.
  SpellingPrediction prediction;
  bool have_prediction = false;
  if (this->spelling_model != nullptr && !line.audio_data.empty()) {
    std::lock_guard<std::mutex> lock(this->spelling_model_mutex);
    int err = this->spelling_model->predict(
        line.audio_data.data(), line.audio_data.size(), INTERNAL_SAMPLE_RATE,
        &prediction);
    have_prediction = (err == 0);
  }

  FusedResult result =
      fuse_default(raw_text, match,
                   have_prediction ? &prediction : nullptr,
                   this->spelling_matcher);
  if (!result.is_character()) return false;

  delete line.text;
  line.text = new std::string(result.character);
  return true;
}

void Transcriber::update_transcript_from_segments(
    const std::vector<VoiceActivitySegment> &segments,
    TranscriberStream *stream, uint32_t flags,
    struct transcript_t **out_transcript) {
  const bool spelling_mode_enabled =
      (flags & MOONSHINE_FLAG_SPELLING_MODE) != 0;
  stream->transcript_output->clear_update_flags();

  for (size_t segment_index = 0; segment_index < segments.size();
       segment_index++) {
    std::lock_guard<std::mutex> output_lock(stream->transcript_output->mutex);
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
      // Use streaming model for transcription (incremental processing)
      line.text = transcribe_segment_with_streaming_model(
          segment.audio_data.data(), segment.audio_data.size(), line.id,
          segment.is_complete);

      // Compute word timestamps from streaming model's collected attention
      if (this->options.word_timestamps &&
          !this->streaming_model->cross_attention_buffer.empty() &&
          !this->last_streaming_tokens.empty()) {
        float seg_duration =
            segment.audio_data.size() / (float)INTERNAL_SAMPLE_RATE;
        int L = this->streaming_model->config.depth;
        int total_steps = this->streaming_model->cross_attn_steps / L;
        int H = this->streaming_model->cross_attn_heads;
        int E = this->streaming_model->cross_attn_enc_len;

        if (total_steps > 0 && H > 0 && E > 0) {
          size_t per_layer_step = H * E;
          std::vector<float> rearranged(L * H * total_steps * E);
          for (int s = 0; s < total_steps; s++) {
            for (int l = 0; l < L; l++) {
              const float *src =
                  this->streaming_model->cross_attention_buffer.data() +
                  (s * L + l) * per_layer_step;
              for (int h = 0; h < H; h++) {
                float *dst =
                    rearranged.data() + ((l * H + h) * total_steps + s) * E;
                memcpy(dst, src + h * E, E * sizeof(float));
              }
            }
          }

          float time_per_frame = seg_duration / static_cast<float>(E);
          std::vector<TranscriberWord> words =
              align_words(rearranged.data(), L, H, total_steps, E,
                          this->last_streaming_tokens, time_per_frame,
                          this->streaming_model->tokenizer);

          if (!words.empty()) {
            for (auto &w : words) {
              w.start += segment.start_time;
              w.end += segment.start_time;
            }
            line.words = std::move(words);
          }
        }

        this->streaming_model->cross_attention_buffer.clear();
        this->streaming_model->cross_attn_steps = 0;
      }
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
      if (this->options.log_output_text) {
        LOGF("Transcribed text: '%s'", out_text);
      }
      // Ensure the output text is valid UTF-8.
      line.text = sanitize_text(out_text);

      // Compute word timestamps (single-pass or alignment model)
      if (this->options.word_timestamps) {
        float seg_duration =
            segment.audio_data.size() / (float)INTERNAL_SAMPLE_RATE;
        std::vector<TranscriberWord> words;
        int align_err =
            this->stt_model->compute_word_timestamps(seg_duration, words);
        if (align_err == 0 && !words.empty()) {
          // Offset word times by the segment's start time
          for (auto &w : words) {
            w.start += segment.start_time;
            w.end += segment.start_time;
          }
          line.words = std::move(words);
        }
      }
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
    if (this->options.return_audio_data || spelling_mode_enabled) {
      // Spelling fusion needs the segment audio for the .ort model.
      // We store it on the line either way; the line is reset before
      // we hand the transcript back so this doesn't leak per-segment.
      line.audio_data = segment.audio_data;
    }
    if (spelling_mode_enabled && line.is_complete) {
      apply_spelling_fusion(line);
    }
    stream->transcript_output->add_or_update_line(line);
  }
  const bool is_stopped = !stream->vad->is_active();
  if (is_stopped) {
    stream->transcript_output->mark_all_lines_as_complete();
  }
  stream->transcript_output->update_transcript_from_lines();
  *out_transcript = &(stream->transcript_output->transcript);
}

namespace {

// Map each aligned word to a UTF-8 byte range in the line text by walking the
// words in order through the transcription string.
std::vector<std::pair<uint64_t, uint64_t>> map_words_to_char_ranges(
    const std::string &line_text, const std::vector<TranscriberWord> &words) {
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  ranges.reserve(words.size());
  size_t search_from = 0;
  for (const TranscriberWord &word : words) {
    if (word.text.empty()) {
      ranges.push_back({0, 0});
      continue;
    }
    size_t pos = line_text.find(word.text, search_from);
    if (pos == std::string::npos) {
      ranges.push_back({0, 0});
      continue;
    }
    ranges.push_back({static_cast<uint64_t>(pos),
                      static_cast<uint64_t>(pos + word.text.size())});
    search_from = pos + word.text.size();
  }
  return ranges;
}

// Fill UTF-8 byte offsets [start_char, end_char) for a speaker span by
// selecting words whose timings overlap the span's absolute time range.
void fill_speaker_span_char_indices(const std::string *line_text,
                                    const std::vector<TranscriberWord> &words,
                                    float span_start_time, float span_duration,
                                    uint64_t *out_start_char,
                                    uint64_t *out_end_char) {
  *out_start_char = 0;
  *out_end_char = 0;
  if (line_text == nullptr || line_text->empty() || words.empty()) {
    return;
  }

  const float span_end_time = span_start_time + span_duration;
  const auto char_ranges = map_words_to_char_ranges(*line_text, words);

  bool found = false;
  uint64_t start_char = 0;
  uint64_t end_char = 0;
  for (size_t i = 0; i < words.size(); i++) {
    const TranscriberWord &word = words[i];
    const auto &range = char_ranges[i];
    if (range.second <= range.first) {
      continue;
    }
    if (word.start < span_end_time && word.end > span_start_time) {
      if (!found) {
        start_char = range.first;
        end_char = range.second;
        found = true;
      } else {
        start_char = std::min(start_char, range.first);
        end_char = std::max(end_char, range.second);
      }
    }
  }

  if (found) {
    *out_start_char = start_char;
    *out_end_char = end_char;
  }
}

}  // namespace

bool Transcriber::apply_speaker_turns_to_lines(
    const std::vector<SpeakerTurn> &turns, TranscriptStreamOutput *output) {
  // Boundary jitter below this size doesn't count as a change, so that small
  // frame-level wobbles between clustering passes don't spam clients with
  // have_speakers_changed notifications.
  constexpr float kTimeTolerance = 0.1f;

  std::lock_guard<std::mutex> lock(output->mutex);
  bool any_changed = false;
  for (const uint64_t &line_id : output->ordered_internal_line_ids) {
    TranscriberLine &line = output->internal_lines_map.at(line_id);
    const float line_start = line.start_time;
    const float line_end = line.start_time + line.duration;

    // Clip each diarization turn to the line's time range. Turns are already
    // sorted by start time.
    std::vector<SpeakerTurn> spans;
    for (const SpeakerTurn &turn : turns) {
      const float span_start = std::max(turn.start_time, line_start);
      const float span_end =
          std::min(turn.start_time + turn.duration, line_end);
      if ((span_end - span_start) <= 0.0f) {
        continue;
      }
      SpeakerTurn span = turn;
      span.start_time = span_start;
      span.duration = span_end - span_start;
      spans.push_back(span);
    }

    bool changed = (spans.size() != line.speaker_spans.size());
    if (!changed) {
      for (size_t i = 0; i < spans.size(); i++) {
        const SpeakerTurn &a = spans[i];
        const SpeakerTurn &b = line.speaker_spans[i];
        if (a.speaker_id != b.speaker_id ||
            std::abs(a.start_time - b.start_time) > kTimeTolerance ||
            std::abs(a.duration - b.duration) > kTimeTolerance) {
          changed = true;
          break;
        }
      }
    }
    if (changed) {
      line.speaker_spans = std::move(spans);
      line.have_speakers_changed = true;
      any_changed = true;
    }
  }
  return any_changed;
}

std::string *Transcriber::transcribe_segment_with_streaming_model(
    const float *audio_data, size_t audio_length, uint64_t segment_id,
    bool is_final) {
  if (audio_length == 0 || this->streaming_model == nullptr) {
    return new std::string();
  }

  const MoonshineStreamingConfig &config = this->streaming_model->config;

  // Check if this is a new segment - if so, reset state
  bool is_new_segment = (segment_id != this->current_streaming_segment_id);
  if (is_new_segment) {
    this->streaming_state.reset(config);
    this->current_streaming_segment_id = segment_id;
    this->streaming_samples_processed = 0;
  }

  // Calculate how many new samples we need to process
  size_t new_samples_start = this->streaming_samples_processed;

  if (new_samples_start >= audio_length) {
    // No new audio to process, but we may still need to decode
    // (e.g., if is_final changed from false to true)
  } else {
    // Process only the NEW audio samples
    const float *new_audio_data = audio_data + new_samples_start;
    size_t new_audio_length = audio_length - new_samples_start;

    const int chunk_size = 1280;  // 80ms at 16kHz
    const size_t chunk_count = new_audio_length / chunk_size;
    {
      std::lock_guard<std::mutex> lock(this->streaming_model_mutex);

      for (size_t chunk_index = 0; chunk_index < chunk_count; chunk_index++) {
        size_t offset = chunk_index * chunk_size;
        int err = this->streaming_model->process_audio_chunk(
            &this->streaming_state, new_audio_data + offset, chunk_size,
            nullptr);
        if (err != 0) {
          LOGF("Failed to process audio chunk: %d", err);
          throw std::runtime_error("Failed to process audio chunk: " +
                                   std::to_string(err));
        }
      }

      // Run encoder - is_final determines if we emit all frames or keep
      // lookahead
      int new_frames = 0;
      int err = this->streaming_model->encode(&this->streaming_state, is_final,
                                              &new_frames);
      if (err != 0) {
        LOGF("Failed to encode: %d", err);
        throw std::runtime_error("Failed to encode: " + std::to_string(err));
      }
    }

    // Update the count of processed samples with the chunks we've actually
    // processed.
    this->streaming_samples_processed += chunk_count * chunk_size;
  }

  // If no memory accumulated, return empty string
  if (this->streaming_state.memory_len == 0) {
    return new std::string();
  }

  // Reset decoder state before decoding (we decode from scratch each time
  // since memory may have changed)
  this->streaming_model->decoder_reset(&this->streaming_state);

  // Decode to get transcription
  const float duration_sec = audio_length / (float)INTERNAL_SAMPLE_RATE;
  const int max_tokens =
      std::min(static_cast<int>(std::ceil(duration_sec *
                                          this->options.max_tokens_per_second)),
               256);
  std::vector<int64_t> tokens;
  tokens.push_back(config.bos_id);

  std::vector<float> logits(config.vocab_size);
  int current_token = config.bos_id;

  {
    std::lock_guard<std::mutex> lock(this->streaming_model_mutex);

    for (int step = 0; step < max_tokens; ++step) {
      int err = this->streaming_model->decode_step(
          &this->streaming_state, current_token, logits.data());
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

      if (next_token == config.eos_id) break;
    }
  }

  // Save tokens for word timestamp alignment
  this->last_streaming_tokens.clear();
  for (auto t : tokens) {
    this->last_streaming_tokens.push_back(static_cast<int>(t));
  }

  // Convert tokens to text
  std::string text = this->streaming_model->tokens_to_text(tokens);
  if (this->options.log_output_text) {
    LOGF("Streaming model transcribed text: '%s'", text.c_str());
  }
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
  this->have_speakers_changed = false;
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
  this->have_speakers_changed = other.have_speakers_changed;
  this->id = other.id;
  this->last_transcription_latency_ms = other.last_transcription_latency_ms;
  this->speaker_spans = other.speaker_spans;
  this->words = other.words;
}

TranscriberLine &TranscriberLine::operator=(const TranscriberLine &other) {
  if (this == &other) {
    return *this;
  }
  std::string *new_text =
      other.text == nullptr ? nullptr : new std::string(*other.text);
  delete this->text;
  this->text = new_text;
  this->audio_data = other.audio_data;
  this->start_time = other.start_time;
  this->duration = other.duration;
  this->is_complete = other.is_complete;
  this->just_updated = other.just_updated;
  this->is_new = other.is_new;
  this->has_text_changed = other.has_text_changed;
  this->have_speakers_changed = other.have_speakers_changed;
  this->id = other.id;
  this->last_transcription_latency_ms = other.last_transcription_latency_ms;
  this->speaker_spans = other.speaker_spans;
  this->words = other.words;
  return *this;
}

TranscriberLine::~TranscriberLine() { delete this->text; }

std::string TranscriberLine::to_string() const {
  std::string spans_string = "[";
  for (size_t i = 0; i < speaker_spans.size(); i++) {
    const SpeakerTurn &span = speaker_spans[i];
    if (i > 0) {
      spans_string += ", ";
    }
    spans_string += "(start_time=" + std::to_string(span.start_time) +
                    ", duration=" + std::to_string(span.duration) +
                    ", speaker_id=" + std::to_string(span.speaker_id) +
                    ", speaker_index=" + std::to_string(span.speaker_index) +
                    ")";
  }
  spans_string += "]";
  return "TranscriberLine(start_time=" + std::to_string(start_time) +
         ", text='" + (text == nullptr ? "<null>" : *text) + "'" +
         ", duration=" + std::to_string(duration) +
         ", is_complete=" + std::to_string(is_complete) +
         ", just_updated=" + std::to_string(just_updated) +
         ", is_new=" + std::to_string(is_new) +
         ", has_text_changed=" + std::to_string(has_text_changed) +
         ", have_speakers_changed=" + std::to_string(have_speakers_changed) +
         ", id=" + std::to_string(id) + ", last_transcription_latency_ms=" +
         std::to_string(last_transcription_latency_ms) +
         ", speaker_spans=" + spans_string + ")";
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
    // Speaker spans are maintained separately by
    // apply_speaker_turns_to_lines(), so carry them over from the existing
    // line rather than dropping them on every transcription update.
    line.speaker_spans = existing_line->speaker_spans;
    line.have_speakers_changed = existing_line->have_speakers_changed;
  } else {
    line.is_new = true;
    line.has_text_changed = line.text != nullptr;
  }
  this->internal_lines_map[line.id] = line;
}

void TranscriptStreamOutput::update_transcript_from_lines() {
  std::lock_guard<std::mutex> lock(this->mutex);
  this->output_lines.clear();
  this->output_words.clear();
  this->output_word_texts.clear();
  this->output_speaker_spans.clear();

  size_t num_lines = this->ordered_internal_line_ids.size();
  this->output_words.resize(num_lines);
  this->output_word_texts.resize(num_lines);
  this->output_speaker_spans.resize(num_lines);

  size_t line_index = 0;
  for (const uint64_t &line_id : this->ordered_internal_line_ids) {
    const TranscriberLine &line = this->internal_lines_map[line_id];
    const bool has_audio_data = line.audio_data.size() > 0;
    const float *audio_data = has_audio_data ? line.audio_data.data() : nullptr;
    const size_t audio_data_count = has_audio_data ? line.audio_data.size() : 0;

    // Build word C structs for this line
    auto &word_texts = this->output_word_texts[line_index];
    auto &word_structs = this->output_words[line_index];
    word_texts.clear();
    word_structs.clear();

    for (const auto &w : line.words) {
      word_texts.push_back(w.text);
    }
    for (size_t i = 0; i < line.words.size(); i++) {
      word_structs.push_back({
          .text = word_texts[i].c_str(),
          .start = line.words[i].start,
          .end = line.words[i].end,
          .confidence = line.words[i].confidence,
      });
    }

    // Build speaker span C structs for this line
    auto &span_structs = this->output_speaker_spans[line_index];
    span_structs.clear();
    for (const SpeakerTurn &span : line.speaker_spans) {
      uint64_t start_char = 0;
      uint64_t end_char = 0;
      fill_speaker_span_char_indices(line.text, line.words, span.start_time,
                                     span.duration, &start_char, &end_char);
      span_structs.push_back({
          .start_time = span.start_time,
          .duration = span.duration,
          .speaker_id = span.speaker_id,
          .speaker_index = span.speaker_index,
          .start_char = start_char,
          .end_char = end_char,
      });
    }

    this->output_lines.push_back({
        .text = line.text == nullptr ? nullptr : line.text->c_str(),
        .audio_data = audio_data,
        .audio_data_count = audio_data_count,
        .start_time = line.start_time,
        .duration = line.duration,
        .id = line.id,
        .is_complete = line.is_complete,
        .is_updated = line.just_updated,
        .is_new = line.is_new,
        .has_text_changed = line.has_text_changed,
        .have_speakers_changed = line.have_speakers_changed,
        .speaker_spans = span_structs.empty() ? nullptr : span_structs.data(),
        .speaker_span_count = (uint64_t)span_structs.size(),
        .last_transcription_latency_ms = line.last_transcription_latency_ms,
        .words = word_structs.empty() ? nullptr : word_structs.data(),
        .word_count = (uint64_t)word_structs.size(),
    });

    line_index++;
  }
  this->transcript.lines = this->output_lines.data();
  this->transcript.line_count = (uint64_t)(this->output_lines.size());
}

void TranscriptStreamOutput::clear_update_flags() {
  std::lock_guard<std::mutex> lock(this->mutex);
  for (const uint64_t &line_id : this->ordered_internal_line_ids) {
    TranscriberLine &line = this->internal_lines_map.at(line_id);
    line.just_updated = false;
    line.is_new = false;
    line.has_text_changed = false;
    line.have_speakers_changed = false;
  }
  for (transcript_line_t &line : this->output_lines) {
    line.is_updated = 0;
    line.has_text_changed = 0;
    line.is_new = 0;
    line.have_speakers_changed = 0;
  }
}

void TranscriptStreamOutput::mark_all_lines_as_complete() {
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    for (const uint64_t &line_id : this->ordered_internal_line_ids) {
      TranscriberLine &line = this->internal_lines_map[line_id];
      if (!line.is_complete) {
        line.is_complete = 1;
        line.just_updated = 1;
      }
    }
  }
  this->update_transcript_from_lines();
}

TranscriberStream::TranscriberStream(VoiceActivityDetector *vad,
                                     int32_t stream_id,
                                     const std::string &save_input_wav_path)
    : vad(vad),
      transcript_output(new TranscriptStreamOutput()),
      save_input_wav_path(save_input_wav_path),
      stream_id(stream_id) {
  if (!this->save_input_wav_path.empty()) {
    std::filesystem::create_directory(this->save_input_wav_path);
    std::string wav_path = append_path_component(this->save_input_wav_path,
                                                 this->get_wav_filename());
    std::filesystem::remove(wav_path);
  }
}

void TranscriberStream::start() {
  this->vad->start();
  std::lock_guard<std::mutex> lock(this->transcript_output->mutex);
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
    // Only log the first time we save a WAV file for a given stream.
    static std::map<std::string, bool> *saved_wav_paths = nullptr;
    if (saved_wav_paths == nullptr) {
      saved_wav_paths = new std::map<std::string, bool>();
    }
    if (saved_wav_paths->find(wav_path) == saved_wav_paths->end()) {
      saved_wav_paths->insert({wav_path, true});
      LOGF("Saving audio data to WAV file: '%s'", wav_path.c_str());
    }
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
