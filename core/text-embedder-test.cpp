#include "text-embedder.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "gemma-embedding-model.h"
#include "moonshine-c-api.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

// Path to the Gemma embedding model
static const std::string EMBEDDING_MODEL_DIR = "embeddinggemma-300m-ONNX";

TextEmbedderOptions make_options() {
  TextEmbedderOptions options;
  options.model_path = EMBEDDING_MODEL_DIR;
  options.model_arch = EmbeddingModelArch::GEMMA_300M;
  options.model_variant = "q4";
  return options;
}

bool embedding_model_available() {
  return std::filesystem::exists(EMBEDDING_MODEL_DIR);
}

// Reads an entire file into a byte vector. Returns an empty vector on failure.
std::vector<uint8_t> read_file_bytes(const std::string &path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    return {};
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  // Read into a char buffer so istream::read gets the char* it wants directly.
  // Reading into a uint8_t vector instead would need one of the pointer casts
  // this file is not cleared to introduce (core/.banned-constructs-allowlist).
  std::string chars(static_cast<size_t>(size), '\0');
  if (size > 0 && !file.read(chars.data(), size)) {
    return {};
  }
  return std::vector<uint8_t>(chars.begin(), chars.end());
}

// True when the all-in-one .ort model + tokenizer are present for the memory
// tests (older .onnx + .onnx_data directories can't be loaded from memory).
bool memory_model_available() {
  return std::filesystem::exists(EMBEDDING_MODEL_DIR + "/model_q4.ort") &&
         std::filesystem::exists(EMBEDDING_MODEL_DIR + "/tokenizer.bin");
}

namespace {

// Phrase matching the way the language bindings do it: embed every candidate
// once, then score an utterance against each candidate with the embedder's
// cosine similarity.
class PhraseMatcher {
 public:
  PhraseMatcher(const TextEmbedder &embedder,
                const std::vector<std::string> &phrases)
      : embedder_(embedder) {
    for (const std::string &phrase : phrases) {
      embeddings_[phrase] = embedder.calculate_embedding(phrase);
    }
  }

  // Returns the best-scoring phrase at or above `threshold` with its score, or
  // an empty phrase when nothing clears it.
  std::pair<std::string, float> best_match(const std::string &utterance,
                                           float threshold) const {
    if (utterance.empty() || embeddings_.empty()) {
      return {"", 0.0f};
    }
    const std::vector<float> utterance_embedding =
        embedder_.calculate_embedding(utterance);
    std::string best_phrase;
    float best_score = -1.0f;
    for (const auto &[phrase, embedding] : embeddings_) {
      const float score =
          embedder_.calculate_similarity(utterance_embedding, embedding);
      if (score > best_score) {
        best_score = score;
        best_phrase = phrase;
      }
    }
    if (best_score < threshold) {
      return {"", best_score};
    }
    return {best_phrase, best_score};
  }

 private:
  const TextEmbedder &embedder_;
  std::map<std::string, std::vector<float>> embeddings_;
};

}  // namespace

TEST_CASE("text-embedder unit tests") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  TextEmbedder embedder(make_options());

  SUBCASE("returns non-empty embedding") {
    auto emb = embedder.calculate_embedding("hello world");
    CHECK(!emb.empty());
  }

  SUBCASE("get_embedding_size returns correct dimension") {
    size_t dim = embedder.get_embedding_size();
    CHECK(dim > 0);
    auto emb = embedder.calculate_embedding("test");
    CHECK(emb.size() == dim);
  }

  SUBCASE("same text produces same embedding") {
    auto emb1 = embedder.calculate_embedding("hello");
    auto emb2 = embedder.calculate_embedding("hello");
    REQUIRE(emb1.size() == emb2.size());
    for (size_t i = 0; i < emb1.size(); ++i) {
      CHECK(emb1[i] == doctest::Approx(emb2[i]));
    }
  }

  SUBCASE("identical embeddings are maximally similar") {
    auto emb = embedder.calculate_embedding("turn on the lights");
    CHECK(embedder.calculate_similarity(emb, emb) > 0.99f);
  }

  SUBCASE("paraphrases score above unrelated sentences") {
    auto lights = embedder.calculate_embedding("turn on the lights");
    auto lamps = embedder.calculate_embedding("switch on the lamps");
    auto market = embedder.calculate_embedding("the stock market crashed");
    CHECK(embedder.calculate_similarity(lights, lamps) >
          embedder.calculate_similarity(lights, market));
  }

  SUBCASE("mismatched embedding sizes score zero") {
    std::vector<float> a{1.0f, 0.0f};
    std::vector<float> b{1.0f, 0.0f, 0.0f};
    CHECK(embedder.calculate_similarity(a, b) == 0.0f);
  }
}

// ============================================================================
// Precision/Recall tests with real GemmaEmbeddingModel
// ============================================================================

struct PhraseTestCase {
  std::string utterance;
  std::string expected_key;  // empty = no match expected
};

struct PrecisionRecallResult {
  int true_positives = 0;
  int false_positives = 0;
  int false_negatives = 0;
  int true_negatives = 0;

  float precision() const {
    int denom = true_positives + false_positives;
    return denom > 0 ? static_cast<float>(true_positives) / denom : 1.0f;
  }

  float recall() const {
    int denom = true_positives + false_negatives;
    return denom > 0 ? static_cast<float>(true_positives) / denom : 1.0f;
  }

  float f1_score() const {
    float p = precision();
    float r = recall();
    return (p + r) > 0 ? 2.0f * p * r / (p + r) : 0.0f;
  }

  float accuracy() const {
    int total =
        true_positives + false_positives + false_negatives + true_negatives;
    return total > 0
               ? static_cast<float>(true_positives + true_negatives) / total
               : 0.0f;
  }
};

TEST_CASE("text-embedder precision/recall with GemmaEmbeddingModel") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping Gemma embedding tests - model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  float threshold = 0.6f;
  TextEmbedder embedder(make_options());

  std::map<std::string, std::string> keys_by_phrase = {
      {"turn on the lights", "lights_on"},
      {"turn off the lights", "lights_off"},
      {"what is the weather", "weather"},
      {"set a timer", "timer"},
      {"play some music", "music_play"},
      {"stop the music", "music_stop"},
      {"turn up the volume", "volume_up"},
      {"turn down the volume", "volume_down"},
  };

  std::vector<std::string> phrases;
  for (const auto &[phrase, key] : keys_by_phrase) {
    phrases.push_back(phrase);
  }
  PhraseMatcher matcher(embedder, phrases);

  SUBCASE("basic phrase matching") {
    auto [phrase, score] = matcher.best_match("turn on the lights", threshold);
    CHECK(keys_by_phrase[phrase] == "lights_on");

    std::tie(phrase, score) =
        matcher.best_match("what is the weather", threshold);
    CHECK(keys_by_phrase[phrase] == "weather");

    std::tie(phrase, score) = matcher.best_match("play some music", threshold);
    CHECK(keys_by_phrase[phrase] == "music_play");
  }

  SUBCASE("precision/recall evaluation") {
    std::vector<PhraseTestCase> test_cases = {
        {"turn on the lights", "lights_on"},
        {"switch on the lights", "lights_on"},
        {"lights on please", "lights_on"},
        {"can you turn the lights on", "lights_on"},
        {"illuminate the room", "lights_on"},
        {"turn off the lights", "lights_off"},
        {"switch off the lights", "lights_off"},
        {"lights off", "lights_off"},
        {"kill the lights", "lights_off"},
        {"what is the weather", "weather"},
        {"how is the weather today", "weather"},
        {"what's the forecast", "weather"},
        {"is it going to rain", "weather"},
        {"weather report please", "weather"},
        {"set a timer", "timer"},
        {"start a timer for 5 minutes", "timer"},
        {"timer for 10 minutes", "timer"},
        {"set an alarm", "timer"},
        {"play some music", "music_play"},
        {"play a song", "music_play"},
        {"start playing music", "music_play"},
        {"put on some tunes", "music_play"},
        {"stop the music", "music_stop"},
        {"pause the music", "music_stop"},
        {"stop playing", "music_stop"},
        {"turn up the volume", "volume_up"},
        {"louder please", "volume_up"},
        {"increase the volume", "volume_up"},
        {"volume up", "volume_up"},
        {"turn down the volume", "volume_down"},
        {"quieter please", "volume_down"},
        {"decrease the volume", "volume_down"},
        {"volume down", "volume_down"},
        {"hello how are you", ""},
        {"tell me a joke", ""},
        {"what time is it", ""},
        {"open the door", ""},
        {"call mom", ""},
        {"send a message", ""},
        {"navigate to the store", ""},
        {"what's the capital of France", ""},
    };

    PrecisionRecallResult results;

    for (const auto &test_case : test_cases) {
      auto [phrase, score] = matcher.best_match(test_case.utterance, threshold);

      bool expected_match = !test_case.expected_key.empty();
      bool matched = !phrase.empty();
      std::string matched_key = matched ? keys_by_phrase[phrase] : "";

      if (expected_match) {
        if (matched && matched_key == test_case.expected_key) {
          results.true_positives++;
        } else if (matched) {
          results.false_positives++;
          MESSAGE("WRONG PHRASE: '", test_case.utterance, "' -> got '",
                  matched_key, "', expected '", test_case.expected_key,
                  "' (similarity: ", score, ")");
        } else {
          results.false_negatives++;
          MESSAGE("MISSED: '", test_case.utterance, "' -> expected '",
                  test_case.expected_key, "'");
        }
      } else {
        if (!matched) {
          results.true_negatives++;
        } else {
          results.false_positives++;
          MESSAGE("FALSE POSITIVE: '", test_case.utterance, "' -> matched '",
                  matched_key, "' (similarity: ", score,
                  "), expected no match");
        }
      }
    }

    MESSAGE("=== Phrase Matching Results (threshold=", threshold, ") ===");
    MESSAGE("True Positives:  ", results.true_positives);
    MESSAGE("False Positives: ", results.false_positives);
    MESSAGE("False Negatives: ", results.false_negatives);
    MESSAGE("True Negatives:  ", results.true_negatives);
    MESSAGE("Precision: ", results.precision());
    MESSAGE("Recall:    ", results.recall());
    MESSAGE("F1 Score:  ", results.f1_score());
    MESSAGE("Accuracy:  ", results.accuracy());

    CHECK(results.precision() >= 0.7f);
    CHECK(results.recall() >= 0.5f);
    CHECK(results.f1_score() >= 0.5f);
  }

  SUBCASE("phrase discrimination") {
    struct DiscriminationTest {
      std::string utterance;
      std::string should_match;
      std::string should_not_match;
    };

    std::vector<DiscriminationTest> discrimination_tests = {
        {"turn on the lights", "lights_on", "lights_off"},
        {"turn off the lights", "lights_off", "lights_on"},
        {"play music", "music_play", "music_stop"},
        {"stop the music", "music_stop", "music_play"},
        {"volume up", "volume_up", "volume_down"},
        {"volume down", "volume_down", "volume_up"},
    };

    int correct_discriminations = 0;
    int total_discriminations = 0;

    for (const auto &test : discrimination_tests) {
      auto [phrase, score] = matcher.best_match(test.utterance, 0.0f);
      REQUIRE(!phrase.empty());
      std::string matched_key = keys_by_phrase[phrase];

      total_discriminations++;
      if (matched_key == test.should_match) {
        correct_discriminations++;
      } else {
        MESSAGE("DISCRIMINATION FAIL: '", test.utterance, "' -> got '",
                matched_key, "', expected '", test.should_match, "'");
      }

      CHECK(matched_key != test.should_not_match);
    }

    float discrimination_accuracy =
        static_cast<float>(correct_discriminations) / total_discriminations;
    MESSAGE("Discrimination accuracy: ", discrimination_accuracy, " (",
            correct_discriminations, "/", total_discriminations, ")");

    CHECK(discrimination_accuracy >= 0.8f);
  }

  SUBCASE("similarity scores for exact matches") {
    std::vector<std::pair<std::string, std::string>> exact_matches = {
        {"turn on the lights", "lights_on"},
        {"turn off the lights", "lights_off"},
        {"what is the weather", "weather"},
        {"set a timer", "timer"},
        {"play some music", "music_play"},
    };

    for (const auto &[utterance, expected_key] : exact_matches) {
      auto [phrase, score] = matcher.best_match(utterance, 0.0f);
      REQUIRE(!phrase.empty());
      CHECK(keys_by_phrase[phrase] == expected_key);
      CHECK(score >= 0.95f);

      MESSAGE("Exact match '", utterance, "' -> ", expected_key,
              " (similarity: ", score, ")");
    }
  }
}

TEST_CASE("C API moonshine_calculate_embedding") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  int32_t handle = moonshine_create_embedding_model(
      EMBEDDING_MODEL_DIR.c_str(), MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
      "q4");
  REQUIRE(handle >= 0);

  SUBCASE("basic embedding calculation") {
    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_embedding(
        handle, "hello world", &embedding, &embedding_size, nullptr);
    CHECK(err == MOONSHINE_ERROR_NONE);
    REQUIRE(embedding != nullptr);
    CHECK(embedding_size > 0);

    bool all_zero = true;
    for (uint64_t i = 0; i < embedding_size; ++i) {
      if (embedding[i] != 0.0f) {
        all_zero = false;
        break;
      }
    }
    CHECK(!all_zero);
    moonshine_free_embedding(embedding);
  }

  SUBCASE("null sentence returns error") {
    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_embedding(handle, nullptr, &embedding,
                                                &embedding_size, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("null out_embedding returns error") {
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_embedding(handle, "hello", nullptr,
                                                &embedding_size, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("null out_embedding_size returns error") {
    float *embedding = nullptr;
    int32_t err = moonshine_calculate_embedding(handle, "hello", &embedding,
                                                nullptr, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("invalid handle returns error") {
    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_embedding(-1, "hello", &embedding,
                                                &embedding_size, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_HANDLE);
  }

  moonshine_free_embedding_model(handle);
}

TEST_CASE("C API moonshine_free_embedding") {
  SUBCASE("safe on nullptr") { moonshine_free_embedding(nullptr); }

  SUBCASE("frees malloc-allocated buffer") {
    float *buf = static_cast<float *>(std::malloc(768 * sizeof(float)));
    REQUIRE(buf != nullptr);
    moonshine_free_embedding(buf);
  }
}

TEST_CASE("C API moonshine_calculate_embedding_distance") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  int32_t handle = moonshine_create_embedding_model(
      EMBEDDING_MODEL_DIR.c_str(), MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
      "q4");
  REQUIRE(handle >= 0);

  SUBCASE("identical embeddings have similarity ~1.0") {
    float *emb = nullptr;
    uint64_t emb_size = 0;
    int32_t err = moonshine_calculate_embedding(handle, "hello world", &emb,
                                                &emb_size, nullptr);
    REQUIRE(err == MOONSHINE_ERROR_NONE);
    REQUIRE(emb != nullptr);

    float similarity = 0.0f;
    err = moonshine_calculate_embedding_distance(handle, emb, emb, emb_size,
                                                 &similarity);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(similarity > 0.99f);
    moonshine_free_embedding(emb);
  }

  SUBCASE("similar sentences have high similarity") {
    float *emb_a = nullptr;
    float *emb_b = nullptr;
    uint64_t size_a = 0, size_b = 0;
    moonshine_calculate_embedding(handle, "turn on the lights", &emb_a, &size_a,
                                  nullptr);
    moonshine_calculate_embedding(handle, "switch on the lamps", &emb_b,
                                  &size_b, nullptr);
    REQUIRE(emb_a != nullptr);
    REQUIRE(emb_b != nullptr);
    REQUIRE(size_a == size_b);

    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(handle, emb_a, emb_b,
                                                         size_a, &similarity);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(similarity > 0.7f);
    moonshine_free_embedding(emb_a);
    moonshine_free_embedding(emb_b);
  }

  SUBCASE("dissimilar sentences have low similarity") {
    float *emb_a = nullptr;
    float *emb_b = nullptr;
    uint64_t size_a = 0, size_b = 0;
    moonshine_calculate_embedding(handle, "turn on the lights", &emb_a, &size_a,
                                  nullptr);
    moonshine_calculate_embedding(handle, "the stock market crashed", &emb_b,
                                  &size_b, nullptr);
    REQUIRE(emb_a != nullptr);
    REQUIRE(emb_b != nullptr);
    REQUIRE(size_a == size_b);

    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(handle, emb_a, emb_b,
                                                         size_a, &similarity);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(similarity < 0.5f);
    moonshine_free_embedding(emb_a);
    moonshine_free_embedding(emb_b);
  }

  SUBCASE("null embedding_a returns error") {
    float dummy = 1.0f;
    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(
        handle, nullptr, &dummy, 1, &similarity);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("null embedding_b returns error") {
    float dummy = 1.0f;
    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(
        handle, &dummy, nullptr, 1, &similarity);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("null out_similarity returns error") {
    float dummy = 1.0f;
    int32_t err = moonshine_calculate_embedding_distance(handle, &dummy, &dummy,
                                                         1, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("zero embedding_size returns error") {
    float dummy = 1.0f;
    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(handle, &dummy, &dummy,
                                                         0, &similarity);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("invalid handle returns error") {
    float dummy = 1.0f;
    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(-1, &dummy, &dummy, 1,
                                                         &similarity);
    CHECK(err == MOONSHINE_ERROR_INVALID_HANDLE);
  }

  moonshine_free_embedding_model(handle);
}

TEST_CASE("TextEmbedder loads the embedding model from memory buffers") {
  if (!memory_model_available()) {
    MESSAGE("Skipping tests - all-in-one .ort model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  std::vector<uint8_t> model_bytes =
      read_file_bytes(EMBEDDING_MODEL_DIR + "/model_q4.ort");
  std::vector<uint8_t> tokenizer_bytes =
      read_file_bytes(EMBEDDING_MODEL_DIR + "/tokenizer.bin");
  REQUIRE(!model_bytes.empty());
  REQUIRE(!tokenizer_bytes.empty());

  SUBCASE("C++ TextEmbedderOptions memory path matches disk loading") {
    TextEmbedderOptions options;
    options.model_arch = EmbeddingModelArch::GEMMA_300M;
    options.model_variant = "q4";
    options.model_data = model_bytes.data();
    options.model_data_size = model_bytes.size();
    options.tokenizer_data = tokenizer_bytes.data();
    options.tokenizer_data_size = tokenizer_bytes.size();

    TextEmbedder embedder(options);
    PhraseMatcher matcher(embedder, {"turn on the lights", "play some music"});
    auto [phrase, score] = matcher.best_match("turn on the lights", 0.0f);
    CHECK(phrase == "turn on the lights");
    CHECK(score >= 0.95f);
  }

  SUBCASE("C API moonshine_create_embedding_model_from_memory") {
    const char *filenames[] = {"model_q4.ort", "tokenizer.bin"};
    const uint8_t *memory[] = {model_bytes.data(), tokenizer_bytes.data()};
    const uint64_t sizes[] = {model_bytes.size(), tokenizer_bytes.size()};

    int32_t handle = moonshine_create_embedding_model_from_memory(
        MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M, "q4", filenames, 2, memory,
        sizes, nullptr, 0, MOONSHINE_HEADER_VERSION);
    REQUIRE(handle >= 0);

    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    CHECK(moonshine_calculate_embedding(handle, "turn on the lights",
                                        &embedding, &embedding_size,
                                        nullptr) == MOONSHINE_ERROR_NONE);
    CHECK(embedding_size > 0);
    moonshine_free_embedding(embedding);
    moonshine_free_embedding_model(handle);
  }

  SUBCASE("tokenizer-only buffers (no model) fails") {
    const char *filenames[] = {"tokenizer.bin"};
    const uint8_t *memory[] = {tokenizer_bytes.data()};
    const uint64_t sizes[] = {tokenizer_bytes.size()};

    int32_t handle = moonshine_create_embedding_model_from_memory(
        MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M, "q4", filenames, 1, memory,
        sizes, nullptr, 0, MOONSHINE_HEADER_VERSION);
    CHECK(handle < 0);
  }

  SUBCASE("empty file list fails") {
    int32_t handle = moonshine_create_embedding_model_from_memory(
        MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M, "q4", nullptr, 0, nullptr,
        nullptr, nullptr, 0, MOONSHINE_HEADER_VERSION);
    CHECK(handle < 0);
  }
}
