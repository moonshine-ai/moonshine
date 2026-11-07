#include "word-alignment.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "bin-tokenizer/bin-tokenizer.h"
#include "debug-utils.h"
#include "moonshine-c-api.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("word-timestamps") {
  SUBCASE("non-streaming-transcribe-with-word-timestamps") {
    std::string model_path = "tiny-en";
    std::string wav_path = "beckett.wav";
    REQUIRE(std::filesystem::exists(model_path));
    REQUIRE(std::filesystem::exists(wav_path));

    // Load with word_timestamps enabled
    struct moonshine_option_t options[] = {
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
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &sample_rate));
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
    float min_confidence = 1.0f;

    for (uint64_t i = 0; i < transcript->line_count; i++) {
      struct transcript_line_t *line = &transcript->lines[i];
      REQUIRE(line->word_count > 0);
      REQUIRE(line->words != nullptr);

      for (uint64_t j = 0; j < line->word_count; j++) {
        const struct transcript_word_t *word = &line->words[j];
        REQUIRE(word->text != nullptr);
        REQUIRE(word->end > word->start);
        REQUIRE(word->start >= prev_end);  // monotonic
        // Every word was chosen by the decoder, so none can have probability
        // zero, and a probability is never above one.
        REQUIRE(word->confidence > 0.0f);
        REQUIRE(word->confidence <= 1.0f);
        min_confidence = std::min(min_confidence, word->confidence);
        prev_end = word->start;
        total_words++;
      }
    }

    REQUIRE(total_words > 0);
    // Confidence used to be a constant 1.0 (GitHub issue #230). A clip this
    // long always has some word the tiny model is unsure of.
    CHECK(min_confidence < 0.99f);
    moonshine_free_transcriber(handle);
  }
}

TEST_CASE("token-probability") {
  SUBCASE("uniform-row") {
    const std::vector<float> logits(8, 0.25f);
    for (int i = 0; i < 8; i++) {
      CHECK(token_probability(logits.data(), 8, i) == doctest::Approx(0.125f));
    }
  }
  SUBCASE("matches-softmax") {
    // exp(ln 3) / (exp(0) + exp(ln 3)) = 3 / 4.
    const std::vector<float> logits = {0.0f, std::log(3.0f)};
    CHECK(token_probability(logits.data(), 2, 0) == doctest::Approx(0.25f));
    CHECK(token_probability(logits.data(), 2, 1) == doctest::Approx(0.75f));
  }
  SUBCASE("large-logits-do-not-overflow") {
    const std::vector<float> logits = {1000.0f, 1000.0f, -1000.0f};
    CHECK(token_probability(logits.data(), 3, 0) == doctest::Approx(0.5f));
    CHECK(token_probability(logits.data(), 3, 2) == doctest::Approx(0.0f));
  }
  SUBCASE("out-of-range-token") {
    const std::vector<float> logits = {1.0f, 2.0f};
    CHECK(token_probability(logits.data(), 2, -1) == 0.0f);
    CHECK(token_probability(logits.data(), 2, 2) == 0.0f);
    CHECK(token_probability(nullptr, 2, 0) == 0.0f);
  }
}

TEST_CASE("word-confidence") {
  // The streaming vocabulary marks word starts with U+2581, splits
  // "Kubernetes" into two subwords (|K + ubernetes), and gives punctuation
  // tokens of its own, which attach to the word before them.
  const std::string tokenizer_path = "tiny-streaming-en/tokenizer.bin";
  REQUIRE(std::filesystem::exists(tokenizer_path));
  BinTokenizer tokenizer(tokenizer_path.c_str(), "\u2581",
                         BinTokenizerEncoding::kBpe);

  // Aligns ``text`` as if the decoder had produced it with the given
  // probability per token, BOS and EOS included. The attention is a plain
  // diagonal, so the alignment itself is unremarkable. A list of the wrong
  // length is refused unless ``any_length``, since it would otherwise pass for
  // "no probabilities" and every confidence would quietly read 1.0.
  const auto align = [&](const std::string &text,
                         const std::vector<float> &probabilities,
                         bool any_length = false) {
    const std::vector<int> text_tokens = tokenizer.text_to_tokens<int>(text);
    // The ID values of BOS and EOS do not matter here: both are dropped
    // before the words are grouped.
    std::vector<int> tokens = {1};
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());
    tokens.push_back(2);
    const int steps = static_cast<int>(tokens.size()) - 1;
    const int frames = 32;
    std::vector<float> attention(steps * frames, 0.0f);
    for (int s = 0; s < steps; s++) {
      attention.at(s * frames + (s * frames) / steps) = 1.0f;
    }
    CAPTURE(text);
    if (!any_length) {
      REQUIRE(probabilities.size() == tokens.size());
    }
    return align_words(attention.data(), 1, 1, steps, frames, tokens,
                       probabilities, 0.02f, &tokenizer);
  };

  SUBCASE("minimum-over-the-word's-tokens") {
    const std::vector<TranscriberWord> words =
        align(" Kubernetes is here", {
                                         1.0f,   // BOS
                                         0.9f,   // |K
                                         0.4f,   // ubernetes
                                         0.8f,   // |is
                                         0.95f,  // |here
                                         0.6f,   // EOS, which is no word's
                                     });
    REQUIRE(words.size() == 3);
    CHECK(words.at(0).text == "Kubernetes");
    CHECK(words.at(0).confidence == doctest::Approx(0.4f));
    CHECK(words.at(1).text == "is");
    CHECK(words.at(1).confidence == doctest::Approx(0.8f));
    CHECK(words.at(2).text == "here");
    CHECK(words.at(2).confidence == doctest::Approx(0.95f));
  }
  SUBCASE("punctuation-does-not-count-against-the-word") {
    const std::vector<TranscriberWord> words =
        align(" Ever tried? Ever failed.", {
                                               1.0f,   // BOS
                                               0.99f,  // |Ever
                                               0.9f,   // |tried
                                               0.3f,   // ?
                                               0.95f,  // |Ever
                                               0.2f,   // |failed
                                               0.8f,   // .
                                               1.0f,   // EOS
                                           });
    REQUIRE(words.size() == 4);
    CHECK(words.at(1).text == "tried?");
    CHECK(words.at(1).confidence == doctest::Approx(0.9f));
    // Leaving the punctuation out never hides a doubtful word.
    CHECK(words.at(3).text == "failed.");
    CHECK(words.at(3).confidence == doctest::Approx(0.2f));
  }
  SUBCASE("non-ascii-punctuation") {
    const std::vector<TranscriberWord> words =
        align(" \u00bfQu\u00e9?", {
                                      1.0f,  // BOS
                                      0.1f,  // |¿
                                      0.8f,  // Qu
                                      0.7f,  // é, a letter, not punctuation
                                      0.2f,  // ?
                                      1.0f,  // EOS
                                  });
    REQUIRE(words.size() == 1);
    CHECK(words.at(0).text == "\u00bfQu\u00e9?");
    CHECK(words.at(0).confidence == doctest::Approx(0.7f));
  }
  SUBCASE("a-word-of-only-punctuation-is-scored-on-it") {
    const std::vector<TranscriberWord> words =
        align(" ... no", {
                             1.0f,  // BOS
                             0.4f,  // |...
                             0.9f,  // |no
                             1.0f,  // EOS
                         });
    REQUIRE(words.size() == 2);
    CHECK(words.at(0).text == "...");
    CHECK(words.at(0).confidence == doctest::Approx(0.4f));
    CHECK(words.at(1).confidence == doctest::Approx(0.9f));
  }
  SUBCASE("no-probabilities-means-full-confidence") {
    for (const std::vector<float> &probabilities :
         {std::vector<float>{}, std::vector<float>{0.5f, 0.5f}}) {
      const std::vector<TranscriberWord> words =
          align(" Kubernetes is here", probabilities, true);
      REQUIRE(words.size() == 3);
      for (const TranscriberWord &word : words) {
        CHECK(word.confidence == 1.0f);
      }
    }
  }
}
