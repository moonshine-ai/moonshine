#include "voice-activity-detector.h"

#include <filesystem>
#include <string>

#include "debug-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("voice-activity-detector-test") {
  if (!std::filesystem::exists("output")) {
    std::filesystem::create_directory("output");
  }
  SUBCASE("vad-block") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    VoiceActivityDetector vad;
    vad.start();
    vad.process_audio(wav_data, wav_data_size, wav_sample_rate);
    vad.stop();

    save_wav_data("output/vad_block_original.wav", wav_data, wav_data_size,
                  wav_sample_rate);
    const std::vector<VoiceActivitySegment> *segments = vad.get_segments();
    REQUIRE(segments->size() >= 1);
    LOGF("Segments count: %zu", segments->size());
    int32_t segment_index = 0;
    for (const VoiceActivitySegment &segment : *segments) {
      LOGF("Segment: start_time=%f, end_time=%f", segment.start_time,
           segment.end_time);
      REQUIRE(segment.audio_data.size() > 0);
      REQUIRE(segment.start_time >= 0);
      REQUIRE(segment.end_time > segment.start_time);
      REQUIRE(segment.is_complete);
      std::string output_wav_path =
          "output/vad_block_" + std::to_string(segment_index) + ".wav";
      REQUIRE(save_wav_data(output_wav_path.c_str(), segment.audio_data.data(),
                            segment.audio_data.size(), 16000));
      segment_index++;
    }
    vad.start();
    vad.stop();
    REQUIRE(vad.get_segments()->empty());
  }
  SUBCASE("vad-stream") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    const float chunk_duration_seconds = 0.1f;
    const size_t chunk_size =
        (size_t)(chunk_duration_seconds * wav_sample_rate);

    VoiceActivityDetector vad;

    vad.start();
    for (size_t i = 0; i < wav_data_size; i += chunk_size) {
      const float *chunk_data = wav_data + i;
      const size_t chunk_data_size = std::min(chunk_size, wav_data_size - i);
      vad.process_audio(chunk_data, chunk_data_size, wav_sample_rate);
      const std::vector<VoiceActivitySegment> *segments = vad.get_segments();
      bool any_updated_segments = false;
      for (size_t j = 0; j < segments->size(); j++) {
        const VoiceActivitySegment &segment = segments->at(j);
        if (!segment.is_complete) {
          const bool is_last_segment = (j == (segments->size() - 1));
          if (!is_last_segment) {
            LOGF("Incomplete segment %zu is not the last segment %zu", j,
                 segments->size() - 1);
          }
          REQUIRE(is_last_segment);
        }
        if (segment.just_updated) {
          any_updated_segments = true;
        } else {
          // If an earlier line has been updated, then all later lines should
          // have been updated as well.
          REQUIRE(!any_updated_segments);
        }
      }
    }
    vad.stop();

    save_wav_data("output/vad_stream_original.wav", wav_data, wav_data_size,
                  wav_sample_rate);
    const std::vector<VoiceActivitySegment> *segments = vad.get_segments();
    REQUIRE(segments->size() >= 1);
    LOGF("Segments count: %zu", segments->size());
    int32_t segment_index = 0;
    for (const VoiceActivitySegment &segment : *segments) {
      LOGF("Segment: start_time=%f, end_time=%f", segment.start_time,
           segment.end_time);
      REQUIRE(segment.audio_data.size() > 0);
      REQUIRE(segment.start_time >= 0);
      REQUIRE(segment.end_time > segment.start_time);
      REQUIRE(segment.is_complete);
      std::string output_wav_path =
          "output/vad_stream_" + std::to_string(segment_index) + ".wav";
      REQUIRE(save_wav_data(output_wav_path.c_str(), segment.audio_data.data(),
                            segment.audio_data.size(), 16000));
      segment_index++;
    }
    vad.start();
    vad.stop();
    REQUIRE(vad.get_segments()->empty());
  }
  SUBCASE("vad-threshold-0") {
    VoiceActivityDetector vad(0.0f);
    std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    vad.start();
    vad.process_audio(wav_data, wav_data_size, wav_sample_rate);
    vad.stop();
    const std::vector<VoiceActivitySegment> *segments = vad.get_segments();
    REQUIRE(segments->size() == 1);
    VoiceActivitySegment segment = segments->at(0);
    REQUIRE(segment.is_complete);
    // The VAD can cut off up to (hop_size - 1) samples from the end of the
    // audio, even with the threshold set to 0.0f.
    const int32_t hop_size = 256;
    const size_t expected_audio_data_size = wav_data_size;
    const size_t expected_audio_data_size_min =
        expected_audio_data_size - hop_size;
    REQUIRE(segment.audio_data.size() >= expected_audio_data_size_min);
    REQUIRE(segment.audio_data.size() <= expected_audio_data_size);
    REQUIRE(segment.start_time >= 0);
    const float epsilon = hop_size * (1.0f / wav_sample_rate);
    REQUIRE(segment.start_time < epsilon);
    const float expected_duration = (float)wav_data_size / wav_sample_rate;
    const float expected_end_time = segment.start_time + expected_duration;
    const float expected_end_time_min = expected_end_time - epsilon;
    const float expected_end_time_max = expected_end_time + epsilon;
    REQUIRE(segment.end_time >= expected_end_time_min);
    REQUIRE(segment.end_time <= expected_end_time_max);
    std::string output_wav_path = "output/vad_threshold_0.wav";
    REQUIRE(save_wav_data(output_wav_path.c_str(), segment.audio_data.data(),
                          segment.audio_data.size(), 16000));
    LOGF("Segment: start_time=%f, end_time=%f", segment.start_time,
         segment.end_time);
    LOGF("Segments count: %zu", segments->size());
  }
}
