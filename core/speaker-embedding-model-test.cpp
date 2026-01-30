#include "speaker-embedding-model.h"

#include <filesystem>
#include <numeric>
#include <string>

#include "debug-utils.h"
#include "resampler.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("speaker-embedding-model") {
  SUBCASE("load-model") {
    std::string model_path = "speaker-embedding-model.ort";
    REQUIRE(std::filesystem::exists(model_path));
    SpeakerEmbeddingModel model;
    REQUIRE(model.load(model_path.c_str()) == 0);
  }
  SUBCASE("calculate-embedding") {
    std::string wav_path = "two_cities_16k.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate) == true);
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > SpeakerEmbeddingModel::ideal_input_size);
    REQUIRE(wav_sample_rate == SpeakerEmbeddingModel::input_sample_rate);
    std::string model_path = "speaker-embedding-model.ort";
    REQUIRE(std::filesystem::exists(model_path));
    SpeakerEmbeddingModel model;
    REQUIRE(model.load(model_path.c_str()) == 0);
    std::vector<float> embedding;
    REQUIRE(model.calculate_embedding(wav_data,
                                      SpeakerEmbeddingModel::ideal_input_size,
                                      &embedding) == 0);
    REQUIRE(embedding.size() == SpeakerEmbeddingModel::embedding_size);
  }
}
