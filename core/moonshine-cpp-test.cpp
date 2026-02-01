#include "moonshine-cpp.h"

#include <cinttypes>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {
// Duplicate of load_wav_data in debug-utils.cpp to avoid depending on
// internal library code.
bool load_wav_data(const char *path, float **out_float_data,
                   size_t *out_num_samples, int32_t *out_sample_rate) {
  *out_float_data = nullptr;
  *out_num_samples = 0;

  // Open the file in binary mode
  FILE *file = std::fopen(path, "rb");
  if (!file) {
    std::perror("Failed to open WAV file");
    return false;
  }

  // Read the RIFF header
  char riff_header[4];
  if (std::fread(riff_header, 1, 4, file) != 4 ||
      std::strncmp(riff_header, "RIFF", 4) != 0) {
    std::fclose(file);
    std::fprintf(stderr, "Not a RIFF file\n");
    return false;
  }

  // Skip chunk size and check WAVE
  std::fseek(file, 4, SEEK_CUR);
  char wave_header[4];
  if (std::fread(wave_header, 1, 4, file) != 4 ||
      std::strncmp(wave_header, "WAVE", 4) != 0) {
    std::fclose(file);
    std::fprintf(stderr, "Not a WAVE file\n");
    return false;
  }

  // Find the "fmt " chunk
  char chunk_id[4];
  uint32_t chunk_size = 0;
  bool found_fmt = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
      found_fmt = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_fmt) {
    std::fclose(file);
    std::fprintf(stderr, "No fmt chunk found\n");
    return false;
  }

  // Read fmt chunk
  uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0, byte_rate = 0;
  uint16_t block_align = 0;
  if (chunk_size < 16) {
    std::fclose(file);
    std::fprintf(stderr, "fmt chunk too small\n");
    return false;
  }
  std::fread(&audio_format, sizeof(uint16_t), 1, file);
  std::fread(&num_channels, sizeof(uint16_t), 1, file);
  std::fread(&sample_rate, sizeof(uint32_t), 1, file);
  std::fread(&byte_rate, sizeof(uint32_t), 1, file);
  std::fread(&block_align, sizeof(uint16_t), 1, file);
  std::fread(&bits_per_sample, sizeof(uint16_t), 1, file);
  // Skip any extra fmt bytes
  if (chunk_size > 16) std::fseek(file, chunk_size - 16, SEEK_CUR);

  if (audio_format != 1 || bits_per_sample != 16) {
    std::fclose(file);
    std::fprintf(stderr, "Only 16-bit PCM WAV files are supported\n");
    return false;
  }

  // Find the "data" chunk
  bool found_data = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "data", 4) == 0) {
      found_data = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_data) {
    std::fclose(file);
    std::fprintf(stderr, "No data chunk found\n");
    return false;
  }

  // Read PCM data
  size_t num_samples = chunk_size / (bits_per_sample / 8);
  if (num_samples == 0) {
    std::fclose(file);
    std::fprintf(stderr, "No samples found\n");
    return false;
  }
  float *result_data = (float *)malloc(num_samples * sizeof(float));
  for (size_t i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    if (std::fread(&sample, sizeof(int16_t), 1, file) != 1) {
      num_samples = i;
      break;
    }
    result_data[i] = static_cast<float>(sample) / 32768.0f;
  }
  std::fclose(file);
  *out_float_data = result_data;
  *out_num_samples = num_samples;
  if (out_sample_rate != nullptr) {
    *out_sample_rate = sample_rate;
  }
  return true;
}

// Would use std::filesystem::exists, but it's not available in C++11.
bool file_exists(const std::string &path) {
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) {
    return false;
  }
  std::fclose(file);
  return true;
}

class TestListener : public moonshine::TranscriptEventListener {
 public:
  int started_count = 0;
  int updated_count = 0;
  int text_changed_count = 0;
  int completed_count = 0;
  void onLineStarted(const moonshine::LineStarted &) override {
    started_count++;
  }
  void onLineUpdated(const moonshine::LineUpdated &) override {
    updated_count++;
  }
  void onLineTextChanged(const moonshine::LineTextChanged &) override {
    text_changed_count++;
  }
  void onLineCompleted(const moonshine::LineCompleted &) override {
    completed_count++;
  }
};

}  // namespace

TEST_CASE("moonshine-cpp-test") {
  SUBCASE("transcribe-without-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(file_exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    moonshine::Transcriber transcriber(root_model_path,
                                       moonshine::ModelArch::TINY);
    moonshine::Transcript transcript = transcriber.transcribeWithoutStreaming(
        std::vector<float>(wav_data, wav_data + wav_data_size), wav_sample_rate,
        0);
    REQUIRE(transcript.lines.size() > 0);
    for (const auto &line : transcript.lines) {
      REQUIRE(line.audioData.size() > 0);
      REQUIRE(line.startTime >= 0.0f);
      REQUIRE(line.duration > 0.0f);
      REQUIRE(line.isComplete);
      REQUIRE(line.isUpdated);
      REQUIRE(line.isNew);
      REQUIRE(line.hasTextChanged);
      REQUIRE(line.hasSpeakerId);
    }
  }
  SUBCASE("transcribe-with-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(file_exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    moonshine::Transcriber transcriber(root_model_path,
                                       moonshine::ModelArch::TINY);

    TestListener listener;
    transcriber.addListener(&listener);

    transcriber.start();

    const float chunk_duration_seconds = 0.0451f;
    const size_t chunk_size =
        (size_t)(chunk_duration_seconds * wav_sample_rate);
    size_t samples_since_last_transcription = 0;
    const size_t samples_between_transcriptions =
        (size_t)(wav_sample_rate * 0.481f);
    size_t line_count = 0;
    std::set<uint64_t> existing_line_ids;
    for (size_t i = 0; i < wav_data_size; i += chunk_size) {
      const float *chunk_data = wav_data + i;
      const size_t chunk_data_size = std::min(chunk_size, wav_data_size - i);
      transcriber.addAudio(
          std::vector<float>(chunk_data, chunk_data + chunk_data_size),
          wav_sample_rate);
      samples_since_last_transcription += chunk_data_size;
      if (samples_since_last_transcription < samples_between_transcriptions) {
        continue;
      }
      samples_since_last_transcription = 0;
      moonshine::Transcript transcript = transcriber.updateTranscription(0);
      line_count = std::max(line_count, transcript.lines.size());
      bool any_updated_lines = false;
      size_t line_index = 0;
      size_t lines_size = transcript.lines.size();
      for (const auto &line : transcript.lines) {
        REQUIRE(line.audioData.size() > 0);
        REQUIRE(line.startTime >= 0.0f);
        REQUIRE(line.duration > 0.0f);

        // Make sure the line ID is unique and stable.
        const bool seen_id_before =
            existing_line_ids.find(line.lineId) != existing_line_ids.end();
        if (!seen_id_before) {
          existing_line_ids.insert(line.lineId);
        }
        REQUIRE(existing_line_ids.size() <= lines_size);

        // There should be at most one incomplete line at the end of the
        // transcript.
        if (!line.isComplete) {
          const bool is_last_line = (line_index == (lines_size - 1));
          if (!is_last_line) {
            fprintf(stderr,
                    "Incomplete line %" PRIu64
                    " ('%s', %.2fs) is not the last line "
                    "%zu\n",
                    line.lineId, line.text.c_str(), line.startTime,
                    lines_size - 1);
          }
          REQUIRE(is_last_line);
        }
        line_index++;

        if (line.isUpdated) {
          any_updated_lines = true;
        } else {
          // If an earlier line has been updated, then all later lines should
          // have been updated as well.
          REQUIRE(!any_updated_lines);
        }
        if (!line.isUpdated) {
          continue;
        }
        fprintf(stderr, "%.1f (#%" PRId64 "): %s\n", line.startTime,
                line.lineId, line.text.c_str());
      }
    }
    transcriber.stop();
    REQUIRE(line_count > 0);
    REQUIRE(listener.started_count > 0);
    REQUIRE(listener.updated_count > 0);
    REQUIRE(listener.text_changed_count > 0);
    REQUIRE(listener.completed_count > 0);
    REQUIRE(listener.started_count == listener.completed_count);
    REQUIRE(listener.updated_count >= listener.started_count);
  }
}