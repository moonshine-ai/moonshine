#include "resampler.h"

#include <filesystem>
#include <numeric>
#include <string>

#include "debug-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {
void test_resample_audio(const std::vector<float> &input_audio,
                         int32_t input_sample_rate,
                         int32_t output_sample_rate) {
  const std::vector<float> &resampled_audio =
      resample_audio(input_audio, input_sample_rate, output_sample_rate);

  const float original_max =
      *std::max_element(input_audio.begin(), input_audio.end());
  const float resampled_max =
      *std::max_element(resampled_audio.begin(), resampled_audio.end());
  LOGF("Original max: %f, Resampled max: %f", original_max, resampled_max);
  REQUIRE(original_max == doctest::Approx(resampled_max).epsilon(0.005f));

  const float original_min =
      *std::min_element(input_audio.begin(), input_audio.end());
  const float resampled_min =
      *std::min_element(resampled_audio.begin(), resampled_audio.end());
  LOGF("Original min: %f, Resampled min: %f", original_min, resampled_min);
  REQUIRE(original_min == doctest::Approx(resampled_min).epsilon(0.005f));

  const float original_mean =
      std::accumulate(input_audio.begin(), input_audio.end(), 0.0f) /
      input_audio.size();
  const float resampled_mean =
      std::accumulate(resampled_audio.begin(), resampled_audio.end(), 0.0f) /
      resampled_audio.size();
  LOGF("Original mean: %f, Resampled mean: %f", original_mean, resampled_mean);
  REQUIRE(original_mean == doctest::Approx(resampled_mean).epsilon(0.001f));
}
}  // namespace

TEST_CASE("resampler-test") {
  SUBCASE("resample-audio") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    const std::vector<float> wav_data_vector(wav_data,
                                             wav_data + wav_data_size);
    LOG("Downsampling to 16000 Hz");
    test_resample_audio(wav_data_vector, wav_sample_rate, 16000);
    LOG("Upsampling to 96000 Hz");
    test_resample_audio(wav_data_vector, wav_sample_rate, 96000);
  }
}
