#include "transcriber.h"

#include <cstdio>
#include <filesystem>
#include <string>

#include "utf8.h"

#include "debug-utils.h"
#include "string-utils.h"
#include "test-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("transcriber-test") {
  if (!std::filesystem::exists("output")) {
    std::filesystem::create_directory("output");
  }
  SUBCASE("transcribe-without-streaming-from-memory") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    std::vector<uint8_t> encoder_model_data =
        load_file_into_memory(root_model_path + "/encoder_model.ort");
    std::vector<uint8_t> decoder_model_data =
        load_file_into_memory(root_model_path + "/decoder_model_merged.ort");
    std::vector<uint8_t> tokenizer_data =
        load_file_into_memory(root_model_path + "/tokenizer.bin");
    REQUIRE(encoder_model_data.size() > 0);
    REQUIRE(decoder_model_data.size() > 0);
    REQUIRE(tokenizer_data.size() > 0);
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::MEMORY;
    options.encoder_model_data = encoder_model_data.data();
    options.encoder_model_data_size = encoder_model_data.size();
    options.decoder_model_data = decoder_model_data.data();
    options.decoder_model_data_size = decoder_model_data.size();
    options.tokenizer_data = tokenizer_data.data();
    options.tokenizer_data_size = tokenizer_data.size();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    Transcriber transcriber(options);
    struct transcript_t *transcript = nullptr;
    transcriber.transcribe_without_streaming(wav_data, wav_data_size,
                                             wav_sample_rate, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    std::set<uint64_t> found_ids;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.text != nullptr);
      REQUIRE(line.audio_data != nullptr);
      REQUIRE(line.audio_data_count > 0);
      REQUIRE(line.start_time >= 0.0f);
      REQUIRE(line.duration > 0.0f);
      REQUIRE(line.is_complete == 1);
      REQUIRE(line.is_updated == 1);
      LOG_UINT64(line.id);
      REQUIRE(found_ids.find(line.id) == found_ids.end());
      found_ids.insert(line.id);
    }
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      char filename_buf[64];
      snprintf(filename_buf, sizeof(filename_buf), "output/line_%02zu.wav", i);
      std::string filename = filename_buf;
      save_wav_data(filename.c_str(), line.audio_data, line.audio_data_count,
                    16000);
      LOGF("Saved %s", filename.c_str());
    }
    LOGF("Transcript: %s",
         Transcriber::transcript_to_string(transcript).c_str());
  }
  SUBCASE("transcribe-vad-threshold-0") {
    std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    options.vad_threshold = 0.0f;
    Transcriber transcriber(options);
    struct transcript_t *transcript = nullptr;
    transcriber.transcribe_without_streaming(wav_data, wav_data_size,
                                             wav_sample_rate, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count == 1);
    const struct transcript_line_t &line = transcript->lines[0];
    REQUIRE(line.text != nullptr);
    REQUIRE(line.audio_data != nullptr);
    REQUIRE(line.audio_data_count > 0);
    REQUIRE(line.start_time >= 0.0f);
    const int32_t hop_size = 256;
    const float epsilon = hop_size * (1.0f / 16000);
    REQUIRE(line.start_time < epsilon);
    const float expected_duration = (float)wav_data_size / wav_sample_rate;
    const float expected_duration_min = expected_duration - epsilon;
    const float expected_duration_max = expected_duration + epsilon;
    REQUIRE(line.duration >= expected_duration_min);
    REQUIRE(line.duration <= expected_duration_max);
    REQUIRE(line.duration > 0.0f);
    REQUIRE(line.is_complete == 1);
    REQUIRE(line.is_updated == 1);
    LOGF("Transcript: %s",
         Transcriber::transcript_to_string(transcript).c_str());
  }
  SUBCASE("transcribe-without-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    Transcriber transcriber(options);
    struct transcript_t *transcript = nullptr;
    transcriber.transcribe_without_streaming(wav_data, wav_data_size,
                                             wav_sample_rate, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    std::set<uint64_t> found_ids;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.text != nullptr);
      REQUIRE(line.audio_data != nullptr);
      REQUIRE(line.audio_data_count > 0);
      REQUIRE(line.start_time >= 0.0f);
      REQUIRE(line.duration > 0.0f);
      REQUIRE(line.is_complete == 1);
      REQUIRE(line.is_updated == 1);
      REQUIRE(found_ids.find(line.id) == found_ids.end());
      found_ids.insert(line.id);
    }
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      char filename_buf[64];
      snprintf(filename_buf, sizeof(filename_buf), "output/line_%02zu.wav", i);
      std::string filename = filename_buf;
      save_wav_data(filename.c_str(), line.audio_data, line.audio_data_count,
                    16000);
      LOGF("Saved %s", filename.c_str());
    }
    LOGF("Transcript: %s",
         Transcriber::transcript_to_string(transcript).c_str());
  }
  SUBCASE("transcribe-with-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "slinkier-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY_STREAMING;
    Transcriber transcriber(options);
    int32_t stream_id = transcriber.create_stream();
    transcriber.start_stream(stream_id);
    REQUIRE(stream_id >= 0);
    struct transcript_t *transcript = nullptr;
    const float chunk_duration_seconds = 0.01f;
    const size_t chunk_size =
        (size_t)(chunk_duration_seconds * wav_sample_rate);
    size_t samples_since_last_transcription = 0;
    const size_t samples_between_transcriptions =
        (size_t)(wav_sample_rate * 0.5f);
    std::vector<std::string> previous_line_texts;
    for (size_t i = 0; i < wav_data_size; i += chunk_size) {
      const float *chunk_data = wav_data + i;
      const size_t chunk_data_size = std::min(chunk_size, wav_data_size - i);
      transcriber.add_audio_to_stream(stream_id, chunk_data, chunk_data_size,
                                      wav_sample_rate);
      samples_since_last_transcription += chunk_data_size;
      if (samples_since_last_transcription < samples_between_transcriptions) {
        continue;
      }
      samples_since_last_transcription = 0;
      transcriber.transcribe_stream(stream_id, 0, &transcript);
      REQUIRE(transcript != nullptr);
      bool any_updated_lines = false;
      bool any_new_lines = false;
      for (size_t j = 0; j < transcript->line_count; j++) {
        const struct transcript_line_t &line = transcript->lines[j];
        REQUIRE(line.text != nullptr);
        REQUIRE(line.audio_data != nullptr);
        REQUIRE(line.audio_data_count > 0);
        REQUIRE(line.start_time >= 0.0f);
        REQUIRE(line.duration > 0.0f);
        // There should be at most one incomplete line at the end of the
        // transcript.
        if (line.is_complete == 0) {
          const bool is_last_line = (j == (transcript->line_count - 1));
          if (!is_last_line) {
            LOGF("Incomplete line %zu ('%s', %.2fs) is not the last line "
                 "%" PRId64,
                 j, line.text, line.start_time, transcript->line_count - 1);
          }
          REQUIRE(is_last_line);
        }
        if (line.is_updated == 1) {
          any_updated_lines = true;
        } else {
          // If an earlier line has been updated, then all later lines should
          // have been updated as well.
          REQUIRE(!any_updated_lines);
        }
        if (line.is_new) {
          any_new_lines = true;
          REQUIRE(line.is_updated == 1);
        } else {
          // If an earlier line has been marked as newly-added, then all
          // later lines must have been marked as newly-added as well.
          REQUIRE(!any_new_lines);
        }
        if (line.has_text_changed) {
          REQUIRE(line.is_updated == 1);
          if (line.is_new == 1) {
            REQUIRE(j >= previous_line_texts.size());
          } else {
            REQUIRE(j < previous_line_texts.size());
            REQUIRE(previous_line_texts.at(j) != line.text);
          }
        } else {
          REQUIRE(j < previous_line_texts.size());
          REQUIRE(previous_line_texts.at(j) == line.text);
        }
        if (!line.is_updated) {
          continue;
        }
        LOGF("%.2f (%" PRId64 "): %s", line.start_time, line.id, line.text);
      }
      previous_line_texts.resize(transcript->line_count);
      for (size_t j = 0; j < transcript->line_count; j++) {
        previous_line_texts[j] = transcript->lines[j].text;
      }
      // Check that state is correctly cleared when a new transcription is
      // requested, but nothing has changed.
      transcript_t *unchanged_transcript = nullptr;
      transcriber.transcribe_stream(stream_id, 0, &unchanged_transcript);
      REQUIRE(unchanged_transcript != nullptr);
      REQUIRE(unchanged_transcript->line_count == transcript->line_count);
      for (size_t j = 0; j < unchanged_transcript->line_count; j++) {
        const struct transcript_line_t &previous_line = transcript->lines[j];
        const struct transcript_line_t &unchanged_line =
            unchanged_transcript->lines[j];
        REQUIRE(unchanged_line.text == previous_line_texts.at(j));
        REQUIRE(unchanged_line.audio_data == previous_line.audio_data);
        REQUIRE(unchanged_line.audio_data_count ==
                previous_line.audio_data_count);
        REQUIRE(unchanged_line.start_time == previous_line.start_time);
        REQUIRE(unchanged_line.duration == previous_line.duration);
        REQUIRE(unchanged_line.id == previous_line.id);
        REQUIRE(unchanged_line.is_complete == previous_line.is_complete);
        REQUIRE(unchanged_line.is_updated == false);
        REQUIRE(unchanged_line.is_new == false);
        REQUIRE(unchanged_line.has_text_changed == false);
      }
    }
    transcriber.stop_stream(stream_id);
    REQUIRE(transcript->line_count > 0);
    LOGF("Transcript: %s",
         Transcriber::transcript_to_string(transcript).c_str());
    transcriber.free_stream(stream_id);
  }
  SUBCASE("no-transcription") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::NONE;
    Transcriber transcriber(options);
    int32_t stream_id = transcriber.create_stream();
    transcriber.start_stream(stream_id);
    REQUIRE(stream_id >= 0);
    struct transcript_t *transcript = nullptr;
    const float chunk_duration_seconds = 0.01f;
    const size_t chunk_size =
        (size_t)(chunk_duration_seconds * wav_sample_rate);
    size_t samples_since_last_transcription = 0;
    const size_t samples_between_transcriptions =
        (size_t)(wav_sample_rate * 0.5f);
    std::vector<std::string> previous_line_texts;
    for (size_t i = 0; i < wav_data_size; i += chunk_size) {
      const float *chunk_data = wav_data + i;
      const size_t chunk_data_size = std::min(chunk_size, wav_data_size - i);
      transcriber.add_audio_to_stream(stream_id, chunk_data, chunk_data_size,
                                      wav_sample_rate);
      samples_since_last_transcription += chunk_data_size;
      if (samples_since_last_transcription < samples_between_transcriptions) {
        continue;
      }
      samples_since_last_transcription = 0;
      transcriber.transcribe_stream(stream_id, 0, &transcript);
      REQUIRE(transcript != nullptr);
      bool any_updated_lines = false;
      bool any_new_lines = false;
      for (size_t j = 0; j < transcript->line_count; j++) {
        const struct transcript_line_t &line = transcript->lines[j];
        REQUIRE(line.text == nullptr);
        REQUIRE(line.audio_data != nullptr);
        REQUIRE(line.audio_data_count > 0);
        REQUIRE(line.start_time >= 0.0f);
        REQUIRE(line.duration > 0.0f);
        // There should be at most one incomplete line at the end of the
        // transcript.
        if (line.is_complete == 0) {
          const bool is_last_line = (j == (transcript->line_count - 1));
          if (!is_last_line) {
            LOGF("Incomplete line %zu ('%s', %.2fs) is not the last line "
                 "%" PRId64,
                 j, line.text, line.start_time, transcript->line_count - 1);
          }
          REQUIRE(is_last_line);
        }
        if (line.is_updated == 1) {
          any_updated_lines = true;
        } else {
          // If an earlier line has been updated, then all later lines should
          // have been updated as well.
          REQUIRE(!any_updated_lines);
        }
        if (line.is_new) {
          any_new_lines = true;
          REQUIRE(line.is_updated == 1);
        } else {
          // If an earlier line has been marked as newly-added, then all
          // later lines must have been marked as newly-added as well.
          REQUIRE(!any_new_lines);
        }
        REQUIRE(line.has_text_changed == false);
        if (!line.is_updated) {
          continue;
        }
        LOGF("%.2f (%" PRId64 "): %s", line.start_time, line.id,
             line.text == nullptr ? "<null>" : line.text);
      }
      // Check that state is correctly cleared when a new transcription is
      // requested, but nothing has changed.
      transcript_t *unchanged_transcript = nullptr;
      transcriber.transcribe_stream(stream_id, 0, &unchanged_transcript);
      REQUIRE(unchanged_transcript != nullptr);
      REQUIRE(unchanged_transcript->line_count == transcript->line_count);
      for (size_t j = 0; j < unchanged_transcript->line_count; j++) {
        const struct transcript_line_t &previous_line = transcript->lines[j];
        const struct transcript_line_t &unchanged_line =
            unchanged_transcript->lines[j];
        REQUIRE(unchanged_line.text == nullptr);
        REQUIRE(unchanged_line.audio_data == previous_line.audio_data);
        REQUIRE(unchanged_line.audio_data_count ==
                previous_line.audio_data_count);
        REQUIRE(unchanged_line.start_time == previous_line.start_time);
        REQUIRE(unchanged_line.duration == previous_line.duration);
        REQUIRE(unchanged_line.id == previous_line.id);
        REQUIRE(unchanged_line.is_complete == transcript->lines[j].is_complete);
        REQUIRE(unchanged_line.is_updated == false);
        REQUIRE(unchanged_line.is_new == false);
        REQUIRE(unchanged_line.has_text_changed == false);
      }
    }
    transcriber.stop_stream(stream_id);
    REQUIRE(transcript->line_count > 0);
    LOGF("Transcript: %s",
         Transcriber::transcript_to_string(transcript).c_str());
    transcriber.free_stream(stream_id);
  }
  SUBCASE("test-invalid-utf8") {
    const uint8_t invalid_utf8_data[] = {0xa3, 0x0a, 0xf5, 0x78};
    const size_t invalid_utf8_data_size = sizeof(invalid_utf8_data);
    std::string invalid_utf8_string((const char *)(invalid_utf8_data),
                                    invalid_utf8_data_size);
    std::string *sanitized_utf8_string =
        Transcriber::sanitize_text(invalid_utf8_string.c_str());
    const uint8_t first_byte = (uint8_t)(sanitized_utf8_string->c_str()[0]);
    REQUIRE(first_byte < 0x80);
    delete sanitized_utf8_string;
  }
  SUBCASE("test-valid-utf8") {
    char valid_utf8_data[] = "Hello, world!";
    const size_t valid_utf8_data_size = sizeof(valid_utf8_data) - 1;
    std::string valid_utf8_string((const char *)(valid_utf8_data),
                                  valid_utf8_data_size);
    std::string *sanitized_utf8_string =
        Transcriber::sanitize_text(valid_utf8_string.c_str());
    LOG_BYTES(sanitized_utf8_string->data(), sanitized_utf8_string->size());
    LOG_BYTES(valid_utf8_string.data(), valid_utf8_string.size());
    REQUIRE(*sanitized_utf8_string == valid_utf8_string);
    delete sanitized_utf8_string;
  }
  SUBCASE("test-save-input-wav-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    options.save_input_wav_path = "output";
    Transcriber transcriber(options);
    int32_t stream_id = transcriber.create_stream();
    transcriber.start_stream(stream_id);
    REQUIRE(stream_id >= 0);
    struct transcript_t *transcript = nullptr;
    const float chunk_duration_seconds = 0.0143f;
    const size_t chunk_size =
        (size_t)(chunk_duration_seconds * wav_sample_rate);
    size_t samples_since_last_transcription = 0;
    const size_t samples_between_transcriptions =
        (size_t)(wav_sample_rate * 5.0f);
    std::vector<std::string> previous_line_texts;
    for (size_t i = 0; i < wav_data_size; i += chunk_size) {
      const float *chunk_data = wav_data + i;
      const size_t chunk_data_size = std::min(chunk_size, wav_data_size - i);
      transcriber.add_audio_to_stream(stream_id, chunk_data, chunk_data_size,
                                      wav_sample_rate);
      samples_since_last_transcription += chunk_data_size;
      if (samples_since_last_transcription < samples_between_transcriptions) {
        continue;
      }
      samples_since_last_transcription = 0;
      transcriber.transcribe_stream(stream_id, 0, &transcript);
    }
    transcriber.stop_stream(stream_id);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    transcriber.free_stream(stream_id);
    REQUIRE(std::filesystem::is_directory(options.save_input_wav_path));
    std::string expected_debug_filename =
        std::string("input_") + std::to_string(stream_id) + std::string(".wav");
    std::string debug_wav_path = append_path_component(
        options.save_input_wav_path, expected_debug_filename);
    REQUIRE(std::filesystem::exists(debug_wav_path));
    float *debug_wav_data = nullptr;
    size_t debug_wav_data_size = 0;
    int32_t debug_wav_sample_rate = 0;
    REQUIRE(load_wav_data(debug_wav_path.c_str(), &debug_wav_data,
                          &debug_wav_data_size, &debug_wav_sample_rate));
    REQUIRE(debug_wav_data != nullptr);
    REQUIRE(wav_data_size == debug_wav_data_size);
    LOG_SIZET(wav_data_size);
    LOG_SIZET(debug_wav_data_size);
    REQUIRE(wav_sample_rate == debug_wav_sample_rate);
    for (size_t i = 0; i < wav_data_size; i++) {
      const float delta = std::abs(wav_data[i] - debug_wav_data[i]);
      const float epsilon = 0.0001f;
      if (delta > epsilon) {
        LOGF("wav_data[%zu] = %f, debug_wav_data[%zu] = %f", i, wav_data[i], i,
             debug_wav_data[i]);
        CHECK(false);
      }
    }
    free(debug_wav_data);
    free(wav_data);
  }
  SUBCASE("test-save-input-wav-without-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    options.save_input_wav_path = "output";
    Transcriber transcriber(options);
    transcript_t *transcript = nullptr;
    transcriber.transcribe_without_streaming(wav_data, wav_data_size,
                                             wav_sample_rate, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    REQUIRE(std::filesystem::is_directory(options.save_input_wav_path));
    std::string expected_debug_filename = std::string("input_batch.wav");
    std::string debug_wav_path = append_path_component(
        options.save_input_wav_path, expected_debug_filename);
    REQUIRE_FILE_EXISTS(debug_wav_path);
    float *debug_wav_data = nullptr;
    size_t debug_wav_data_size = 0;
    int32_t debug_wav_sample_rate = 0;
    REQUIRE(load_wav_data(debug_wav_path.c_str(), &debug_wav_data,
                          &debug_wav_data_size, &debug_wav_sample_rate));
    REQUIRE(debug_wav_data != nullptr);
    REQUIRE(wav_data_size == debug_wav_data_size);
    LOG_SIZET(wav_data_size);
    LOG_SIZET(debug_wav_data_size);
    REQUIRE(wav_sample_rate == debug_wav_sample_rate);
    for (size_t i = 0; i < wav_data_size; i++) {
      const float delta = std::abs(wav_data[i] - debug_wav_data[i]);
      const float epsilon = 0.0001f;
      if (delta > epsilon) {
        LOGF("wav_data[%zu] = %f, debug_wav_data[%zu] = %f", i, wav_data[i], i,
             debug_wav_data[i]);
        CHECK(false);
      }
    }
    free(debug_wav_data);
    free(wav_data);
  }
}
