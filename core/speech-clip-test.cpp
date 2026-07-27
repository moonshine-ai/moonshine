#include "speech-clip.h"

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "debug-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

constexpr int32_t kClipSampleRate = 16000;

std::vector<float> silence(float seconds, int32_t sample_rate) {
  return std::vector<float>(
      static_cast<size_t>(std::lround(seconds * sample_rate)), 0.0f);
}

}  // namespace

TEST_CASE("speech-clip-test") {
  SUBCASE("extracts a clip from speech surrounded by silence") {
    const std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data_size > 0);

    // Two seconds of silence in front, so the extractor has to find the
    // speech rather than just taking the start of the recording.
    std::vector<float> padded = silence(2.0f, wav_sample_rate);
    const float lead_in_seconds = 2.0f;
    padded.insert(padded.end(), wav_data, wav_data + wav_data_size);

    const SpeechClip clip = extract_speech_clip(padded.data(), padded.size(),
                                                wav_sample_rate);
    REQUIRE(clip.is_complete);
    REQUIRE(clip.audio.size() == 4 * kClipSampleRate);
    REQUIRE(clip.speech_seconds >= 2.0f);
    // The window should land on the speech, not the leading silence. Allow it
    // to start slightly early since the VAD keeps a little look-behind.
    REQUIRE(clip.start_time_seconds > lead_in_seconds - 0.5f);
  }

  SUBCASE("reports incomplete for pure silence") {
    const std::vector<float> quiet = silence(6.0f, kClipSampleRate);
    const SpeechClip clip =
        extract_speech_clip(quiet.data(), quiet.size(), kClipSampleRate);
    REQUIRE(!clip.is_complete);
    REQUIRE(clip.audio.empty());
    REQUIRE(clip.speech_seconds < 2.0f);
  }

  SUBCASE("reports incomplete when the recording is shorter than the clip") {
    const std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));

    const size_t one_second = static_cast<size_t>(wav_sample_rate);
    REQUIRE(wav_data_size > one_second);
    const SpeechClip clip =
        extract_speech_clip(wav_data, one_second, wav_sample_rate);
    REQUIRE(!clip.is_complete);
    REQUIRE(clip.audio.empty());
  }

  SUBCASE("honours a shorter requested duration") {
    const std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));

    SpeechClipOptions options;
    options.clip_duration_seconds = 2.0f;
    options.minimum_speech_seconds = 1.0f;
    const SpeechClip clip = extract_speech_clip(wav_data, wav_data_size,
                                                wav_sample_rate, options);
    REQUIRE(clip.is_complete);
    REQUIRE(clip.audio.size() == 2 * kClipSampleRate);
  }

  SUBCASE("rejects bad arguments") {
    const std::vector<float> quiet = silence(6.0f, kClipSampleRate);
    REQUIRE(!extract_speech_clip(nullptr, 16, kClipSampleRate).is_complete);
    REQUIRE(!extract_speech_clip(quiet.data(), 0, kClipSampleRate).is_complete);
    REQUIRE(!extract_speech_clip(quiet.data(), quiet.size(), 0).is_complete);
  }
}
