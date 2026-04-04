#include "moonshine-c-api.h"

#include <array>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "debug-utils.h"
#include "string-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

std::filesystem::path find_de_piper_voices_dir() {
  const std::filesystem::path candidates[] = {
      std::filesystem::path("core") / "moonshine-tts" / "data" / "de" / "piper-voices",
      std::filesystem::path("..") / "core" / "moonshine-tts" / "data" / "de" / "piper-voices",
  };
  for (const auto& c : candidates) {
    if (std::filesystem::is_directory(c)) {
      return c;
    }
  }
  return {};
}

std::vector<uint8_t> read_binary_file(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    return {};
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

}  // namespace

TEST_CASE("moonshine-test-v2") {
  SUBCASE("transcribe-complete") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    int32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;

    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    int32_t transcriber_handle = moonshine_load_transcriber_from_files(
        root_model_path.c_str(), model_arch, nullptr, 0,
        MOONSHINE_HEADER_VERSION);
    REQUIRE(transcriber_handle >= 0);

    struct transcript_t *transcript = nullptr;
    int32_t transcribe_error = moonshine_transcribe_without_streaming(
        transcriber_handle, wav_data, wav_data_size, wav_sample_rate, 0,
        &transcript);
    REQUIRE(transcribe_error == MOONSHINE_ERROR_NONE);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.text != nullptr);
      REQUIRE(line.audio_data != nullptr);
      REQUIRE(line.audio_data_count > 0);
      REQUIRE(line.start_time >= 0.0f);
      REQUIRE(line.duration > 0.0f);
      REQUIRE(line.is_complete == 1);
      REQUIRE(line.is_updated == 1);
      REQUIRE(line.is_new == 1);
      REQUIRE(line.has_text_changed == 1);
      REQUIRE(line.has_speaker_id == 1);
    }
  }
  SUBCASE("transcribe-stream") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    int32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;

    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    const moonshine_option_t options[] = {
        {"identify_speakers", "false"},
    };
    const uint64_t options_count = sizeof(options) / sizeof(options[0]);

    int32_t transcriber_handle = moonshine_load_transcriber_from_files(
        root_model_path.c_str(), model_arch, options, options_count,
        MOONSHINE_HEADER_VERSION);
    REQUIRE(transcriber_handle >= 0);

    int32_t stream_id = moonshine_create_stream(transcriber_handle, 0);
    REQUIRE(stream_id >= 0);
    int32_t start_error = moonshine_start_stream(transcriber_handle, stream_id);
    REQUIRE(start_error == MOONSHINE_ERROR_NONE);

    struct transcript_t *transcript = nullptr;
    const float chunk_duration_seconds = 0.0723f;
    const size_t chunk_size =
        (size_t)(chunk_duration_seconds * wav_sample_rate);
    size_t samples_since_last_transcription = 0;
    const size_t samples_between_transcriptions =
        (size_t)(wav_sample_rate * 0.481f);
    for (size_t i = 0; i < wav_data_size; i += chunk_size) {
      const float *chunk_data = wav_data + i;
      const size_t chunk_data_size = std::min(chunk_size, wav_data_size - i);
      moonshine_transcribe_add_audio_to_stream(transcriber_handle, stream_id,
                                               chunk_data, chunk_data_size,
                                               wav_sample_rate, 0);
      samples_since_last_transcription += chunk_data_size;
      if (samples_since_last_transcription < samples_between_transcriptions) {
        continue;
      }
      samples_since_last_transcription = 0;
      int32_t transcribe_error = moonshine_transcribe_stream(
          transcriber_handle, stream_id, 0, &transcript);
      REQUIRE(transcribe_error == MOONSHINE_ERROR_NONE);
      REQUIRE(transcript != nullptr);
      bool any_updated_lines = false;
      for (size_t j = 0; j < transcript->line_count; j++) {
        const struct transcript_line_t &line = transcript->lines[j];
        REQUIRE(line.text != nullptr);
        REQUIRE(line.audio_data != nullptr);
        REQUIRE(line.audio_data_count > 0);
        REQUIRE(line.start_time >= 0.0f);
        REQUIRE(line.duration > 0.0f);
        REQUIRE(line.has_speaker_id == 0);
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
        if (line.is_updated) {
          any_updated_lines = true;
        } else {
          // If an earlier line has been updated, then all later lines should
          // have been updated as well.
          REQUIRE(!any_updated_lines);
        }
        if (!line.is_updated) {
          continue;
        }
        LOGF("%.1f (#%" PRId64 "): %s", line.start_time, line.id, line.text);
      }
    }
    int32_t stop_error = moonshine_stop_stream(transcriber_handle, stream_id);
    REQUIRE(stop_error == MOONSHINE_ERROR_NONE);
    REQUIRE(transcript->line_count > 0);
    LOGF("Transcript: %s", moonshine_transcript_to_string(transcript));
    moonshine_free_stream(transcriber_handle, stream_id);
  }
  SUBCASE("transcribe-complete-from-memory") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    int32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;

    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    std::string encoder_model_path =
        append_path_component(root_model_path, "encoder_model.ort");
    std::string decoder_model_path =
        append_path_component(root_model_path, "decoder_model_merged.ort");
    std::string tokenizer_path =
        append_path_component(root_model_path, "tokenizer.bin");
    REQUIRE(std::filesystem::exists(encoder_model_path));
    REQUIRE(std::filesystem::exists(decoder_model_path));
    REQUIRE(std::filesystem::exists(tokenizer_path));
    std::vector<uint8_t> encoder_model_data =
        load_file_into_memory(encoder_model_path);
    std::vector<uint8_t> decoder_model_data =
        load_file_into_memory(decoder_model_path);
    std::vector<uint8_t> tokenizer_data = load_file_into_memory(tokenizer_path);
    REQUIRE(encoder_model_data.size() > 0);
    REQUIRE(decoder_model_data.size() > 0);
    REQUIRE(tokenizer_data.size() > 0);

    const moonshine_option_t options[] = {
        {"return_audio_data", "false"},
    };
    const uint64_t options_count = sizeof(options) / sizeof(options[0]);

    int32_t transcriber_handle = moonshine_load_transcriber_from_memory(
        encoder_model_data.data(), encoder_model_data.size(),
        decoder_model_data.data(), decoder_model_data.size(),
        tokenizer_data.data(), tokenizer_data.size(), model_arch, options,
        options_count, MOONSHINE_HEADER_VERSION);
    REQUIRE(transcriber_handle >= 0);

    struct transcript_t *transcript = nullptr;
    int32_t transcribe_error = moonshine_transcribe_without_streaming(
        transcriber_handle, wav_data, wav_data_size, wav_sample_rate, 0,
        &transcript);
    REQUIRE(transcribe_error == MOONSHINE_ERROR_NONE);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.text != nullptr);
      REQUIRE(line.audio_data == nullptr);
      REQUIRE(line.audio_data_count == 0);
      REQUIRE(line.start_time >= 0.0f);
      REQUIRE(line.duration > 0.0f);
      REQUIRE(line.is_complete == 1);
      REQUIRE(line.is_updated == 1);
      REQUIRE(line.has_speaker_id == 1);
    }
  }
  SUBCASE("transcribe-without-streaming-skip-transcription") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    int32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;

    const moonshine_option_t options[] = {
        {"skip_transcription", "true"},
    };
    const uint64_t options_count = sizeof(options) / sizeof(options[0]);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    int32_t transcriber_handle = moonshine_load_transcriber_from_files(
        root_model_path.c_str(), model_arch, options, options_count,
        MOONSHINE_HEADER_VERSION);
    REQUIRE(transcriber_handle >= 0);

    struct transcript_t *transcript = nullptr;
    int32_t transcribe_error = moonshine_transcribe_without_streaming(
        transcriber_handle, wav_data, wav_data_size, wav_sample_rate, 0,
        &transcript);
    REQUIRE(transcribe_error == MOONSHINE_ERROR_NONE);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);
    for (size_t i = 0; i < transcript->line_count; i++) {
      const struct transcript_line_t &line = transcript->lines[i];
      REQUIRE(line.text == nullptr);
      REQUIRE(line.audio_data != nullptr);
      REQUIRE(line.audio_data_count > 0);
      REQUIRE(line.start_time >= 0.0f);
      REQUIRE(line.duration > 0.0f);
      REQUIRE(line.is_complete == 1);
      REQUIRE(line.is_updated == 1);
      REQUIRE(line.is_new == 1);
      REQUIRE(line.has_text_changed == 0);
    }
  }
  SUBCASE("transcribe-without-streaming-vad-threshold-0") {
    std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    int32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;

    const moonshine_option_t options[] = {
        {"vad_threshold", "0.0"},
    };
    const uint64_t options_count = sizeof(options) / sizeof(options[0]);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    int32_t transcriber_handle = moonshine_load_transcriber_from_files(
        root_model_path.c_str(), model_arch, options, options_count,
        MOONSHINE_HEADER_VERSION);
    REQUIRE(transcriber_handle >= 0);

    struct transcript_t *transcript = nullptr;
    int32_t transcribe_error = moonshine_transcribe_without_streaming(
        transcriber_handle, wav_data, wav_data_size, wav_sample_rate, 0,
        &transcript);
    REQUIRE(transcribe_error == MOONSHINE_ERROR_NONE);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count == 1);
    const struct transcript_line_t &line = transcript->lines[0];
    REQUIRE(line.text != nullptr);
    REQUIRE(line.audio_data != nullptr);
    const int32_t hop_size = 256;
    const size_t expected_audio_data_size =
        (size_t)(wav_data_size * wav_sample_rate / 16000.0f);
    const size_t expected_audio_data_size_min =
        expected_audio_data_size - hop_size;
    const size_t expected_audio_data_size_max =
        expected_audio_data_size + hop_size;
    REQUIRE(line.audio_data_count >= expected_audio_data_size_min);
    REQUIRE(line.audio_data_count <= expected_audio_data_size_max);
    REQUIRE(line.start_time < 0.001f);
    REQUIRE(line.duration > 0.0f);
    REQUIRE(line.is_complete == 1);
    REQUIRE(line.is_updated == 1);
    REQUIRE(line.is_new == 1);
    REQUIRE(line.has_text_changed == 1);
  }
  SUBCASE("transcribe-valid-options") {
    int32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;
    const moonshine_option_t options[] = {
        {"skip_transcription", "true"},
        {"transcription_interval", "0.5"},
        {"vad_threshold", "0.5"},
        {"save_input_wav_path", "test.wav"},
        {"log_api_calls", "true"},
        {"log_ort_run", "true"},
        {"vad_window_duration", "0.5"},
        {"vad_hop_size", "512"},
        {"vad_look_behind_sample_count", "8192"},
        {"vad_max_segment_duration", "15.0"},
        {"max_tokens_per_second", "6.5"},
        {"identify_speakers", "true"},
        {"speaker_id_cluster_threshold", "0.6"},
        {"return_audio_data", "false"},
        {"log_output_text", "true"},
    };
    const uint64_t options_count = sizeof(options) / sizeof(options[0]);
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    int32_t transcriber_handle = moonshine_load_transcriber_from_files(
        root_model_path.c_str(), model_arch, options, options_count,
        MOONSHINE_HEADER_VERSION);
    REQUIRE(transcriber_handle >= 0);
  }
  SUBCASE("transcribe-invalid-option") {
    const moonshine_option_t options[] = {
        {"invalid_option", "true"},
    };
    const uint64_t options_count = sizeof(options) / sizeof(options[0]);
    int32_t model_arch = MOONSHINE_MODEL_ARCH_TINY;
    std::string root_model_path = "tiny-en";
    REQUIRE(std::filesystem::exists(root_model_path));
    int32_t transcriber_handle = moonshine_load_transcriber_from_files(
        root_model_path.c_str(), model_arch, options, options_count,
        MOONSHINE_HEADER_VERSION);
    REQUIRE(transcriber_handle < 0);
  }
  SUBCASE("tts-synthesizer-valid-options") {
    const moonshine_option_t options[] = {
        {"engine", "auto"},
        {"model_root", ""},
        {"kokoro_dir", ""},
        {"piper_voices_dir", ""},
        {"lang", "en_us"},
        {"voice", "af_heart"},
        {"speed", "1.0"},
        {"output", "out.wav"},
        {"bundle_g2p_data", "true"},  // corresponds to use_bundled_cpp_g2p_data
    };
    const uint64_t options_count = sizeof(options) / sizeof(options[0]);
    int32_t tts_synthesizer_handle =
        moonshine_create_tts_synthesizer_from_files(
            "en_us", nullptr, 0, options, options_count,
            MOONSHINE_HEADER_VERSION);
    REQUIRE(tts_synthesizer_handle >= 0);
  }
  SUBCASE("tts-piper-german-from-memory") {
    const std::filesystem::path voices_dir = find_de_piper_voices_dir();
    if (voices_dir.empty()) {
      return;
    }
    std::filesystem::path onnx_path;
    for (const auto& ent : std::filesystem::directory_iterator(voices_dir)) {
      if (!ent.is_regular_file()) {
        continue;
      }
      const auto& p = ent.path();
      if (p.extension() == ".onnx") {
        onnx_path = p;
        break;
      }
    }
    if (onnx_path.empty()) {
      return;
    }
    std::filesystem::path json_path = onnx_path;
    json_path.replace_extension(".onnx.json");
    std::vector<uint8_t> onnx_data = read_binary_file(onnx_path);
    std::vector<uint8_t> json_data = read_binary_file(json_path);
    if (onnx_data.empty() || json_data.empty()) {
      return;
    }
    std::array<std::string, 2> keys = {std::string("piper/onnx"), std::string("piper/onnx.json")};
    const char* filenames[2] = {keys[0].c_str(), keys[1].c_str()};
    const uint8_t* mem_ptrs[2] = {onnx_data.data(), json_data.data()};
    const uint64_t mem_sizes[2] = {static_cast<uint64_t>(onnx_data.size()),
                                   static_cast<uint64_t>(json_data.size())};
    const std::string voice_stem = onnx_path.filename().string();
    const moonshine_option_t opts[] = {
        {"engine", "piper"},
        {"voice", voice_stem.c_str()},
        {"speed", "1.0"},
        {"bundle_g2p_data", "true"},
    };
    int32_t h = moonshine_create_tts_synthesizer_from_memory(
        "de", filenames, 2, mem_ptrs, mem_sizes, opts, sizeof(opts) / sizeof(opts[0]),
        MOONSHINE_HEADER_VERSION);
    REQUIRE(h >= 0);
    float* audio = nullptr;
    uint64_t audio_n = 0;
    int32_t sr = 0;
    REQUIRE(moonshine_text_to_speech(h, "Hallo", nullptr, 0, &audio, &audio_n, &sr) ==
            MOONSHINE_ERROR_NONE);
    REQUIRE(audio_n > 0);
    REQUIRE(audio != nullptr);
    std::free(audio);
    moonshine_free_tts_synthesizer(h);
  }
}
