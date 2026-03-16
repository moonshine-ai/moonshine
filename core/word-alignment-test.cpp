#include "moonshine-c-api.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "debug-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("word-timestamps") {
  SUBCASE("non-streaming-transcribe-with-word-timestamps") {
    std::string model_path = "tiny-en";
    std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(wav_path));

    // Load with word_timestamps enabled
    struct transcriber_option_t options[] = {
        {.name = "word_timestamps", .value = "true"},
        {.name = "identify_speakers", .value = "false"},
    };

    int32_t handle = moonshine_load_transcriber_from_files(
        model_path.c_str(), MOONSHINE_MODEL_ARCH_TINY, options, 2,
        moonshine_get_version());
    REQUIRE(handle >= 0);

    // Load WAV file
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size, &sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);

    // Transcribe
    struct transcript_t *transcript = nullptr;
    int32_t err = moonshine_transcribe_without_streaming(
        handle, wav_data, wav_data_size, sample_rate, 0, &transcript);
    REQUIRE(err == 0);
    REQUIRE(transcript != nullptr);
    REQUIRE(transcript->line_count > 0);

    // Verify word timestamps
    int total_words = 0;
    float prev_end = -1.0f;

    for (uint64_t i = 0; i < transcript->line_count; i++) {
      struct transcript_line_t *line = &transcript->lines[i];
      REQUIRE(line->word_count > 0);
      REQUIRE(line->words != nullptr);

      for (uint64_t j = 0; j < line->word_count; j++) {
        const struct transcript_word_t *word = &line->words[j];
        REQUIRE(word->text != nullptr);
        REQUIRE(word->end > word->start);
        REQUIRE(word->start >= prev_end);  // monotonic
        REQUIRE(word->confidence >= 0.0f);
        REQUIRE(word->confidence <= 1.0f);
        prev_end = word->start;
        total_words++;
      }
    }

    REQUIRE(total_words > 0);
    moonshine_free_transcriber(handle);
  }
}
