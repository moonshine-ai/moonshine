#include "transcriber.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "debug-utils.h"
#include "speaker-diarizer.h"
#include "string-utils.h"
#include "test-utils.h"
#include "utf8.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {
// The diarization models ship as a download rather than compiled-in data
// (docs/diarization-models.md), so the tests point at the copies under
// test-assets, which is where they run from.
constexpr const char *kDiarizationModelDir = "diarization";
}  // namespace

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
    std::string wav_path = "two_cities_librivox_48k.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-streaming-en";
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
            LOGF(
                "Incomplete line %zu ('%s', %.2fs) is not the last line "
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
    float transcript_duration = 0.0f;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      transcript_duration += line.duration;
    }
    const float wav_duration = (float)wav_data_size / wav_sample_rate;
    // We expect that talking will take up at least 80% of the audio
    // for this audio file.
    const float expected_duration_min = (wav_duration * 0.8f);
    const float expected_duration_max = (wav_duration * 1.01f);
    REQUIRE(transcript_duration >= expected_duration_min);
    REQUIRE(transcript_duration <= expected_duration_max);

    LOGF("Original transcript: %s",
         Transcriber::transcript_to_string(transcript).c_str());

    // Store here because it will be overwritten when we restart the stream.
    const size_t original_line_count = transcript->line_count;

    // Ensure that the transcript is cleared after restarting the stream.
    transcriber.start_stream(stream_id);
    transcript_t *restarted_transcript = nullptr;
    transcriber.transcribe_stream(stream_id, 0, &restarted_transcript);
    REQUIRE(restarted_transcript != nullptr);
    REQUIRE(restarted_transcript->lines == nullptr);
    REQUIRE(restarted_transcript->line_count == 0);

    std::map<uint64_t, std::string> transcript_line_map;

    // Ensure that a valid transcript is returned after restarting the stream.
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
      transcriber.transcribe_stream(stream_id, 0, &restarted_transcript);
      REQUIRE(restarted_transcript != nullptr);

      // Make sure that all the flags are set correctly for the transcript
      // lines.
      for (size_t j = 0; j < restarted_transcript->line_count; j++) {
        const struct transcript_line_t &line = restarted_transcript->lines[j];
        if (transcript_line_map.find(line.id) == transcript_line_map.end()) {
          REQUIRE(line.is_new == 1);
          REQUIRE(line.is_updated == 1);
          transcript_line_map[line.id] = std::string(line.text);
        } else {
          REQUIRE(line.is_new == 0);
          if (line.has_text_changed) {
            REQUIRE(line.is_updated == 1);
            REQUIRE(transcript_line_map[line.id] != std::string(line.text));
          } else {
            // FIXME: The internal transcription update triggered by
            // `transcription_interval` may change the text and then the
            // explicit client update will only set the text changed flag if the
            // text is different from the one produced by the internal
            // transcription update. Clients should be able to rely on the text
            // changed flag to know that the text has changed *since their last
            // update*, and not have that be affected by the internal
            // transcription update. REQUIRE(transcript_line_map[line.id] ==
            // std::string(line.text));
          }
        }
      }
    }
    transcriber.stop_stream(stream_id);
    REQUIRE(restarted_transcript->line_count > 0);

    LOGF("Restarted transcript: %s",
         Transcriber::transcript_to_string(restarted_transcript).c_str());

    // Ensure that the two transcripts have roughly the same number of lines.
    const size_t line_delta = std::abs(
        (int64_t)(restarted_transcript->line_count - original_line_count));
    REQUIRE(line_delta <= 4);

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
            LOGF(
                "Incomplete line %zu ('%s', %.2fs) is not the last line "
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
  SUBCASE("test-transcriberline-assignment-operator") {
    TranscriberLine first;
    first.text = new std::string("first");
    first.id = 1;
    first.start_time = 1.25f;
    first.duration = 3.5f;
    first.audio_data = {0.25f, 0.5f, 0.75f};
    first.speaker_spans = {{.start_time = 1.25f,
                            .duration = 2.0f,
                            .speaker_id = 55,
                            .speaker_index = 3}};
    first.have_speakers_changed = true;

    TranscriberLine second;
    second.text = new std::string("second");
    second.id = 2;
    second = first;
    REQUIRE(second.text != nullptr);
    REQUIRE(*second.text == "first");
    REQUIRE(second.text != first.text);
    REQUIRE(second.id == first.id);
    REQUIRE(second.start_time == first.start_time);
    REQUIRE(second.duration == first.duration);
    REQUIRE(second.audio_data == first.audio_data);
    REQUIRE(second.speaker_spans.size() == first.speaker_spans.size());
    REQUIRE(second.speaker_spans[0].speaker_id ==
            first.speaker_spans[0].speaker_id);
    REQUIRE(second.speaker_spans[0].speaker_index ==
            first.speaker_spans[0].speaker_index);
    REQUIRE(second.have_speakers_changed == first.have_speakers_changed);

    TranscriberLine repeated_assignment_target;
    repeated_assignment_target.text = new std::string("seed");
    for (int i = 0; i < 200; i++) {
      TranscriberLine source;
      source.text = new std::string("value_" + std::to_string(i));
      source.id = (uint64_t)(1000 + i);
      source.duration = i * 0.01f;
      source.speaker_spans = {{.start_time = 0.0f,
                               .duration = source.duration,
                               .speaker_id = (uint64_t)(i),
                               .speaker_index = (uint32_t)(i % 4)}};
      repeated_assignment_target = source;
      REQUIRE(repeated_assignment_target.text != nullptr);
      REQUIRE(*repeated_assignment_target.text == *source.text);
      REQUIRE(repeated_assignment_target.text != source.text);
      REQUIRE(repeated_assignment_target.id == source.id);
      REQUIRE(repeated_assignment_target.duration == source.duration);
      REQUIRE(repeated_assignment_target.speaker_spans[0].speaker_index ==
              source.speaker_spans[0].speaker_index);
    }

    const std::string before_self_assignment = *repeated_assignment_target.text;
    TranscriberLine &self_alias = repeated_assignment_target;
    repeated_assignment_target = self_alias;
    REQUIRE(repeated_assignment_target.text != nullptr);
    REQUIRE(*repeated_assignment_target.text == before_self_assignment);

    TranscriberLine null_text_source;
    null_text_source.id = 999;
    null_text_source.text = nullptr;
    repeated_assignment_target = null_text_source;
    REQUIRE(repeated_assignment_target.text == nullptr);
    REQUIRE(repeated_assignment_target.id == 999);

    std::map<uint64_t, TranscriberLine> lines;
    TranscriberLine first_map_line;
    first_map_line.id = 42;
    first_map_line.text = new std::string("map_first");
    lines[first_map_line.id] = first_map_line;
    TranscriberLine second_map_line;
    second_map_line.id = 42;
    second_map_line.text = new std::string("map_second");
    second_map_line.duration = 4.2f;
    lines[second_map_line.id] = second_map_line;
    REQUIRE(lines.size() == 1);
    REQUIRE(lines.at(42).text != nullptr);
    REQUIRE(*lines.at(42).text == "map_second");
    REQUIRE(lines.at(42).duration == second_map_line.duration);
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
  SUBCASE("test-mark-all-lines-as-complete-when-stream-is-stopped") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    // Truncate the audio data so we're in the middle of a sentence.
    REQUIRE(wav_data_size >= (wav_sample_rate * 35));
    wav_data_size = (wav_sample_rate * 35);
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
        (size_t)(wav_sample_rate * 0.45f);
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
    transcriber.transcribe_stream(stream_id, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.is_complete == 1);
    }
    transcriber.free_stream(stream_id);
    free(wav_data);
  }
  SUBCASE("test-speaker-spans-absent-by-default") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
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
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.speaker_spans == nullptr);
      REQUIRE(line.speaker_span_count == 0);
      REQUIRE(line.have_speakers_changed == 0);
    }
    free(wav_data);
  }
  SUBCASE("test-speaker-span-clipping") {
    auto add_line = [](TranscriptStreamOutput &output, uint64_t id, float start,
                       float duration) {
      TranscriberLine line;
      line.id = id;
      line.start_time = start;
      line.duration = duration;
      output.internal_lines_map[id] = line;
      output.ordered_internal_line_ids.push_back(id);
    };
    auto make_turn = [](float start, float duration, uint64_t speaker) {
      SpeakerTurn out;
      out.start_time = start;
      out.duration = duration;
      out.speaker_id = speaker;
      out.speaker_index = (uint32_t)(speaker);
      return out;
    };
    auto spans_of = [](const TranscriptStreamOutput &output, uint64_t id) {
      return output.internal_lines_map.at(id).speaker_spans;
    };

    TranscriptStreamOutput output;
    add_line(output, 1, 0.0f, 10.0f);
    add_line(output, 2, 10.0f, 10.0f);
    // One turn per line plus one straddling the boundary, so that both the
    // turns that touch a line's edges exactly and the one that crosses it are
    // covered.
    const std::vector<SpeakerTurn> turns = {make_turn(0.0f, 10.0f, 7),
                                            make_turn(5.0f, 10.0f, 8),
                                            make_turn(10.0f, 10.0f, 9)};

    REQUIRE(Transcriber::apply_speaker_turns_to_lines(turns, &output));

    // A turn starting exactly where the line ends contributes nothing to it.
    const std::vector<SpeakerTurn> first = spans_of(output, 1);
    REQUIRE(first.size() == 2);
    REQUIRE(first[0].speaker_id == 7);
    REQUIRE(first[0].start_time == doctest::Approx(0.0f));
    REQUIRE(first[0].duration == doctest::Approx(10.0f));
    REQUIRE(first[1].speaker_id == 8);
    REQUIRE(first[1].start_time == doctest::Approx(5.0f));
    REQUIRE(first[1].duration == doctest::Approx(5.0f));

    // Nor does one ending exactly where the line starts.
    const std::vector<SpeakerTurn> second = spans_of(output, 2);
    REQUIRE(second.size() == 2);
    REQUIRE(second[0].speaker_id == 8);
    REQUIRE(second[0].start_time == doctest::Approx(10.0f));
    REQUIRE(second[0].duration == doctest::Approx(5.0f));
    REQUIRE(second[1].speaker_id == 9);
    REQUIRE(second[1].start_time == doctest::Approx(10.0f));
    REQUIRE(second[1].duration == doctest::Approx(10.0f));

    // Nothing has moved, so nothing is reported as having changed.
    REQUIRE(!Transcriber::apply_speaker_turns_to_lines(turns, &output));

    // Turns arriving out of order have to produce the same spans: their being
    // sorted is only ever an opportunity to do less work.
    TranscriptStreamOutput shuffled;
    add_line(shuffled, 1, 0.0f, 10.0f);
    add_line(shuffled, 2, 10.0f, 10.0f);
    const std::vector<SpeakerTurn> out_of_order = {turns[2], turns[0],
                                                   turns[1]};
    REQUIRE(Transcriber::apply_speaker_turns_to_lines(out_of_order, &shuffled));
    for (const uint64_t id : {(uint64_t)(1), (uint64_t)(2)}) {
      auto by_speaker = [](const SpeakerTurn &a, const SpeakerTurn &b) {
        return a.speaker_id < b.speaker_id;
      };
      std::vector<SpeakerTurn> got = spans_of(shuffled, id);
      std::vector<SpeakerTurn> want = spans_of(output, id);
      std::sort(got.begin(), got.end(), by_speaker);
      std::sort(want.begin(), want.end(), by_speaker);
      REQUIRE(got.size() == want.size());
      for (size_t i = 0; i < got.size(); i++) {
        REQUIRE(got[i].speaker_id == want[i].speaker_id);
        REQUIRE(got[i].start_time == doctest::Approx(want[i].start_time));
        REQUIRE(got[i].duration == doctest::Approx(want[i].duration));
      }
    }
  }
  SUBCASE("test-speaker-span-clipping-at-scale") {
    // Lines and turns both accumulate for the whole of a session, and this runs
    // over them on every streaming transcription call, so the answer has to
    // hold at the size an hours-long meeting reaches and not just at the size a
    // hand-written case does. Each line here abuts the next exactly, which is
    // the arrangement that decides whether the neighbouring turns are correctly
    // treated as out of reach: a turn ending where a line starts, or starting
    // where it ends, belongs to the neighbour and not to this line.
    constexpr size_t kLines = 20000;
    TranscriptStreamOutput output;
    std::vector<SpeakerTurn> turns;
    turns.reserve(kLines);
    for (size_t i = 0; i < kLines; i++) {
      const float start = (float)(i) * 4.0f;
      TranscriberLine line;
      line.id = (uint64_t)(i + 1);
      line.start_time = start;
      line.duration = 4.0f;
      output.internal_lines_map[line.id] = line;
      output.ordered_internal_line_ids.push_back(line.id);

      SpeakerTurn turn;
      turn.start_time = start;
      turn.duration = 4.0f;
      turn.speaker_id = (uint64_t)(i % 3);
      turn.speaker_index = (uint32_t)(i % 3);
      turns.push_back(turn);
    }

    REQUIRE(Transcriber::apply_speaker_turns_to_lines(turns, &output));

    // Counted rather than asserted line by line, so that a failure reports how
    // much of the transcript was wrong instead of stopping at the first.
    size_t exact = 0;
    for (size_t i = 0; i < kLines; i++) {
      const std::vector<SpeakerTurn> &spans =
          output.internal_lines_map.at((uint64_t)(i + 1)).speaker_spans;
      if (spans.size() == 1 && spans[0].speaker_id == (uint64_t)(i % 3) &&
          std::abs(spans[0].start_time - (float)(i) * 4.0f) < 0.001f &&
          std::abs(spans[0].duration - 4.0f) < 0.001f) {
        exact += 1;
      }
    }
    REQUIRE(exact == kLines);

    // A second pass over the same turns has nothing to report, which is the
    // common case in a session and the one that has to stay cheap.
    REQUIRE(!Transcriber::apply_speaker_turns_to_lines(turns, &output));
  }
  SUBCASE("test-identify-speakers") {
    std::string first_pete_wav_path = "two_cities.wav";
    std::string other_speaker_wav_path = "two_cities_librivox_48k.wav";
    REQUIRE(std::filesystem::exists(first_pete_wav_path));
    REQUIRE(std::filesystem::exists(other_speaker_wav_path));
    float *first_pete_wav_data = nullptr;
    size_t first_pete_wav_data_size = 0;
    int32_t first_pete_wav_sample_rate = 0;
    REQUIRE(load_wav_data(first_pete_wav_path.c_str(), &first_pete_wav_data,
                          &first_pete_wav_data_size,
                          &first_pete_wav_sample_rate));
    REQUIRE(first_pete_wav_data != nullptr);
    REQUIRE(first_pete_wav_data_size > 0);
    float *other_speaker_wav_data = nullptr;
    size_t other_speaker_wav_data_size = 0;
    int32_t other_speaker_wav_sample_rate = 0;
    REQUIRE(load_wav_data(other_speaker_wav_path.c_str(),
                          &other_speaker_wav_data, &other_speaker_wav_data_size,
                          &other_speaker_wav_sample_rate));
    REQUIRE(other_speaker_wav_data != nullptr);
    REQUIRE(other_speaker_wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    options.identify_speakers = true;
    REQUIRE(std::filesystem::exists(kDiarizationModelDir));
    options.diarization_model_dir = kDiarizationModelDir;
    Transcriber transcriber(options);
    int32_t stream_id = transcriber.create_stream();
    transcriber.start_stream(stream_id);
    REQUIRE(stream_id >= 0);
    struct transcript_t *transcript = nullptr;
    transcriber.add_audio_to_stream(stream_id, first_pete_wav_data,
                                    first_pete_wav_data_size,
                                    first_pete_wav_sample_rate);
    transcriber.transcribe_stream(stream_id, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    // The whole first file is one speaker, so every attributed span should
    // carry the same stable speaker ID, and the first speaker should have
    // index zero.
    REQUIRE(transcript->lines[0].speaker_span_count > 0);
    REQUIRE(transcript->lines[0].speaker_spans[0].speaker_index == 0);
    // identify_speakers turns on word timestamps so spans can be mapped to
    // text.
    REQUIRE(transcript->lines[0].word_count > 0);
    const uint64_t first_pete_speaker_id =
        transcript->lines[0].speaker_spans[0].speaker_id;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      for (uint64_t j = 0; j < line.speaker_span_count; j++) {
        const struct speaker_span_t &span = line.speaker_spans[j];
        REQUIRE(span.speaker_id == first_pete_speaker_id);
        // Spans are clipped to the line's time range (small tolerance for
        // float rounding).
        REQUIRE(span.start_time >= line.start_time - 0.01f);
        REQUIRE(span.start_time + span.duration <=
                line.start_time + line.duration + 0.01f);
        if (line.text != nullptr && line.word_count > 0 &&
            line.speaker_span_count == 1) {
          const size_t text_len = strlen(line.text);
          REQUIRE(span.end_char > span.start_char);
          REQUIRE(span.start_char < text_len);
          REQUIRE(span.end_char <= text_len);
        }
      }
    }
    transcriber.add_audio_to_stream(stream_id, other_speaker_wav_data,
                                    other_speaker_wav_data_size,
                                    other_speaker_wav_sample_rate);
    transcriber.transcribe_stream(stream_id, 0, &transcript);
    REQUIRE(transcript != nullptr);
    // Force a final clustering pass over the whole session.
    transcriber.stop_stream(stream_id);
    transcriber.transcribe_stream(stream_id, 0, &transcript);
    REQUIRE(transcript != nullptr);
    // The second file is a different speaker, so the final clustering should
    // have found more than one stable speaker ID across the transcript.
    std::set<uint64_t> speaker_ids;
    size_t lines_with_spans = 0;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.is_complete == 1);
      if (line.speaker_span_count > 0) {
        lines_with_spans += 1;
      }
      for (uint64_t j = 0; j < line.speaker_span_count; j++) {
        speaker_ids.insert(line.speaker_spans[j].speaker_id);
      }
    }
    REQUIRE(lines_with_spans > 0);
    REQUIRE(speaker_ids.size() > 1);
    transcriber.free_stream(stream_id);
    free(first_pete_wav_data);
    free(other_speaker_wav_data);
  }
  SUBCASE("test-identify-speakers-batch") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    options.identify_speakers = true;
    REQUIRE(std::filesystem::exists(kDiarizationModelDir));
    options.diarization_model_dir = kDiarizationModelDir;
    Transcriber transcriber(options);
    struct transcript_t *transcript = nullptr;
    transcriber.transcribe_without_streaming(wav_data, wav_data_size,
                                             wav_sample_rate, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    size_t lines_with_spans = 0;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      if (line.speaker_span_count > 0) {
        lines_with_spans += 1;
        REQUIRE(line.speaker_spans != nullptr);
      }
    }
    REQUIRE(lines_with_spans > 0);
    free(wav_data);
  }
  SUBCASE("test-identify-speakers-endgame") {
    std::string wav_path = "endgame_nagg_nell.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    const float wav_duration =
        wav_data_size / static_cast<float>(wav_sample_rate);
    REQUIRE(wav_duration >= 20.0f);
    REQUIRE(wav_duration <= 35.0f);

    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = root_model_path.c_str();
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    options.identify_speakers = true;
    REQUIRE(std::filesystem::exists(kDiarizationModelDir));
    options.diarization_model_dir = kDiarizationModelDir;
    Transcriber transcriber(options);
    struct transcript_t *transcript = nullptr;
    transcriber.transcribe_without_streaming(wav_data, wav_data_size,
                                             wav_sample_rate, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);

    std::set<uint64_t> speaker_ids;
    float total_span_duration = 0.0f;
    size_t lines_with_spans = 0;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.word_count > 0);
      if (line.speaker_span_count == 0) {
        continue;
      }
      lines_with_spans += 1;
      for (uint64_t j = 0; j < line.speaker_span_count; j++) {
        const struct speaker_span_t &span = line.speaker_spans[j];
        speaker_ids.insert(span.speaker_id);
        total_span_duration += span.duration;
        REQUIRE(span.duration > 0.0f);
        if (line.text != nullptr && span.end_char > span.start_char) {
          REQUIRE(span.end_char <= strlen(line.text));
        }
      }
    }
    REQUIRE(lines_with_spans > 0);
    // Synthetic ZipVoice dialogue alternates male/female voices; expect at
    // least two stable speaker IDs, but do not assert exact boundaries.
    REQUIRE(speaker_ids.size() >= 2);
    REQUIRE(total_span_duration >= wav_duration * 0.35f);
    REQUIRE(total_span_duration <= wav_duration * 1.25f);
    free(wav_data);
  }
  SUBCASE("test-word-timestamps-wait-for-line-end") {
    // Aligning words is a second pass over the segment, and an unfinished
    // segment gets re-transcribed on every streaming update, so a line is only
    // aligned once its speaker has stopped. This pins both halves of that:
    // nothing is aligned early, and nothing is left unaligned at the end.
    std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    REQUIRE(std::filesystem::exists("tiny-en"));

    TranscriberOptions options;
    options.model_source = TranscriberOptions::ModelSource::FILES;
    options.model_path = "tiny-en";
    options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
    options.word_timestamps = true;
    Transcriber transcriber(options);
    const int32_t stream_id = transcriber.create_stream();
    REQUIRE(stream_id >= 0);
    transcriber.start_stream(stream_id);

    // Half a second at a time, so most updates land mid-utterance.
    const size_t chunk = (size_t)(wav_sample_rate / 2);
    size_t unfinished_seen = 0;
    struct transcript_t *transcript = nullptr;
    for (size_t offset = 0; offset < wav_data_size; offset += chunk) {
      const size_t count = std::min(chunk, wav_data_size - offset);
      transcriber.add_audio_to_stream(stream_id, wav_data + offset, count,
                                      wav_sample_rate);
      transcriber.transcribe_stream(stream_id, 0, &transcript);
      REQUIRE(transcript != nullptr);
      for (size_t i = 0; i < transcript->line_count; i++) {
        const struct transcript_line_t &line = transcript->lines[i];
        if (line.is_complete != 0) {
          continue;
        }
        unfinished_seen += 1;
        REQUIRE(line.word_count == 0);
        REQUIRE(line.words == nullptr);
      }
    }
    // Without this the loop above could pass by never seeing a partial line.
    REQUIRE(unfinished_seen > 0);

    transcriber.stop_stream(stream_id);
    transcriber.transcribe_stream(stream_id, 0, &transcript);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    size_t aligned_lines = 0;
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.is_complete != 0);
      if (line.text == nullptr || strlen(line.text) == 0) {
        continue;
      }
      // Every line that ended up with text carries the words for it, and each
      // word sits inside the line it belongs to.
      REQUIRE(line.word_count > 0);
      aligned_lines += 1;
      for (uint64_t w = 0; w < line.word_count; w++) {
        const struct transcript_word_t &word = line.words[w];
        REQUIRE(word.end >= word.start);
        REQUIRE(word.start >= line.start_time - 0.5f);
        REQUIRE(word.end <= line.start_time + line.duration + 0.5f);
      }
    }
    REQUIRE(aligned_lines > 0);
    free(wav_data);
  }
}
