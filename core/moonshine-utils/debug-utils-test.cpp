#include "debug-utils.h"

#include <cstdio>
#include <filesystem>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {
int return_on_error_test() {
  RETURN_ON_ERROR(1);
  return 0;
}

int return_on_false_test() {
  RETURN_ON_FALSE(false);
  return 0;
}

int return_on_null_test() {
  RETURN_ON_NULL(nullptr);
  return 0;
}

int return_on_not_equal_test() {
  RETURN_ON_NOT_EQUAL(1, 2);
  return 0;
}
}  // namespace

TEST_CASE("debug-utils") {
  SUBCASE("LOG") {
    LOG("Hello, world!");
    CHECK(true);
  }
  SUBCASE("RETURN_ON_ERROR") { CHECK(return_on_error_test() == -1); }
  SUBCASE("RETURN_ON_FALSE") { CHECK(return_on_false_test() == -1); }
  SUBCASE("RETURN_ON_NULL") { CHECK(return_on_null_test() == -1); }
  SUBCASE("RETURN_ON_NOT_EQUAL") { CHECK(return_on_not_equal_test() == -1); }
  SUBCASE("TIMER") {
    TIMER_START(my_timer);
    TIMER_END(my_timer);
    CHECK(true);
  }
  SUBCASE("DEBUG_CALLOC") {
    void *ptr = DEBUG_CALLOC(1, 1);
    CHECK(ptr != nullptr);
    DEBUG_FREE(ptr);
    CHECK(true);
  }
  SUBCASE("TRACE") {
    TRACE();
    CHECK(true);
  }
  SUBCASE("LOG_VARS") {
    LOG_INT(1);
    LOG_INT64((int64_t)(1));
    LOG_LONG((long)(1));
    LOG_SIZET((size_t)(1));
    LOG_PTR(nullptr);
    std::vector<int64_t> vector = {1, 2, 3, 4, 5};
    LOG_VECTOR(vector);
    LOG_FLOAT(1.0f);
    std::vector<float> float_vector = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    LOG_VECTOR(float_vector);
    LOG_STRING(std::string("Hello, world!"));
    std::vector<std::string> string_vector = {"Hello", "world"};
    LOG_VECTOR(string_vector);
    LOG_BOOL(true);
    std::string bytes =
        "Lorem ipsum dolor sit amet, consectetur adipiscing "
        "elit, sed do eiusmod tempor";
    LOG_BYTES(bytes.c_str(), bytes.size());
    CHECK(true);
  }
  SUBCASE("load_file_into_memory") {
    std::string file_contents = "Hello, world!";
    FILE *file = std::fopen("test.txt", "w");
    std::fwrite(file_contents.c_str(), 1, file_contents.size(), file);
    std::fclose(file);

    std::vector<uint8_t> data = load_file_into_memory("test.txt");
    CHECK(data.size() == file_contents.size());
    CHECK(std::string(data.begin(), data.end()) == file_contents);
    std::remove("test.txt");
  }
  SUBCASE("save_memory_to_file") {
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    save_memory_to_file("test.bin", data);
    REQUIRE(std::filesystem::exists("test.bin"));
    REQUIRE(std::filesystem::file_size("test.bin") == data.size());
    FILE *file = std::fopen("test.bin", "rb");
    std::vector<uint8_t> read_data(data.size());
    size_t bytes_read = std::fread(read_data.data(), 1, data.size(), file);
    REQUIRE(bytes_read == data.size());
    std::fclose(file);
    CHECK(read_data == data);
    std::remove("test.bin");
  }
  SUBCASE("load_wav_data_beckett") {
    std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *audio_data = nullptr;
    size_t num_samples = 0;
    int32_t sample_rate = 0;
    bool success = load_wav_data(wav_path.c_str(), &audio_data, &num_samples,
                                 &sample_rate);
    CHECK(success);
    CHECK(audio_data != nullptr);
    CHECK(num_samples == 159414);
    CHECK(sample_rate == 16000);
    free(audio_data);
  }
  SUBCASE("load_wav_data_two_cities") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(std::filesystem::exists(wav_path));
    float *audio_data = nullptr;
    size_t num_samples = 0;
    int32_t sample_rate = 0;
    bool success = load_wav_data(wav_path.c_str(), &audio_data, &num_samples,
                                 &sample_rate);
    CHECK(success);
    CHECK(audio_data != nullptr);
    CHECK(num_samples == 2129958);
    CHECK(sample_rate == 48000);
    free(audio_data);
  }
  SUBCASE("save_wav_data") {
    std::string wav_path = "output/test.wav";
    std::filesystem::create_directory("output");
    std::filesystem::remove(wav_path);
    REQUIRE(!std::filesystem::exists(wav_path));
    std::vector<float> audio_data = {-0.1f, 0.0f, 0.3f, 0.4f, 0.5f};
    size_t num_samples = audio_data.size();
    int32_t sample_rate = 16000;
    bool success = save_wav_data(wav_path.c_str(), audio_data.data(),
                                 num_samples, sample_rate);
    CHECK(success);
    REQUIRE(std::filesystem::exists(wav_path));
    float *read_audio_data = nullptr;
    size_t read_num_samples = 0;
    int32_t read_sample_rate = 0;
    bool read_success = load_wav_data(wav_path.c_str(), &read_audio_data,
                                      &read_num_samples, &read_sample_rate);
    CHECK(read_success);
    CHECK(read_audio_data != nullptr);
    CHECK(read_num_samples == num_samples);
    CHECK(read_sample_rate == sample_rate);
    for (size_t i = 0; i < num_samples; i++) {
      const float delta = std::abs(audio_data[i] - read_audio_data[i]);
      const float epsilon = 0.0001f;
      if (delta > epsilon) {
        LOGF("audio_data[%zu] = %f, read_audio_data[%zu] = %f", i,
             audio_data[i], i, read_audio_data[i]);
        CHECK(false);
      }
    }
    free(read_audio_data);
    std::filesystem::remove(wav_path);
  }
}