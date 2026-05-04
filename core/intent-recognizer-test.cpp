#include "intent-recognizer.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

#include "gemma-embedding-model.h"
#include "moonshine-c-api.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

// Path to the Gemma embedding model
static const std::string EMBEDDING_MODEL_DIR = "embeddinggemma-300m-ONNX";

IntentRecognizerOptions make_options() {
  IntentRecognizerOptions options;
  options.model_path = EMBEDDING_MODEL_DIR;
  options.model_arch = EmbeddingModelArch::GEMMA_300M;
  options.model_variant = "q4";
  return options;
}

bool embedding_model_available() {
  return std::filesystem::exists(EMBEDDING_MODEL_DIR);
}

TEST_CASE("intent-recognizer unit tests") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  IntentRecognizer recognizer(make_options());

  SUBCASE("register and count intents") {
    CHECK(recognizer.get_intent_count() == 0);

    recognizer.register_intent("turn on the lights");
    CHECK(recognizer.get_intent_count() == 1);

    recognizer.register_intent("turn off the lights");
    CHECK(recognizer.get_intent_count() == 2);
  }

  SUBCASE("unregister intent") {
    recognizer.register_intent("turn on the lights");
    CHECK(recognizer.get_intent_count() == 1);

    bool removed = recognizer.unregister_intent("turn on the lights");
    CHECK(removed == true);
    CHECK(recognizer.get_intent_count() == 0);
  }

  SUBCASE("unregister nonexistent intent") {
    bool removed = recognizer.unregister_intent("does not exist");
    CHECK(removed == false);
  }

  SUBCASE("clear intents") {
    recognizer.register_intent("intent1");
    recognizer.register_intent("intent2");
    CHECK(recognizer.get_intent_count() == 2);

    recognizer.clear_intents();
    CHECK(recognizer.get_intent_count() == 0);
  }

  SUBCASE("rank_intents returns empty for empty utterance") {
    recognizer.register_intent("turn on the lights");
    CHECK(recognizer.rank_intents("", 0.0f, 6).empty());
  }

  SUBCASE("rank_intents sorts by similarity descending and respects max") {
    recognizer.register_intent("turn on the lights");
    recognizer.register_intent("turn off the lights");
    recognizer.register_intent("what is the weather");

    auto ranked = recognizer.rank_intents("turn on the lights", 0.0f, 6);
    CHECK(ranked.size() <= 6);
    CHECK(ranked.size() >= 1);
    CHECK(ranked[0].first == "turn on the lights");
    for (size_t i = 1; i < ranked.size(); ++i) {
      CHECK(ranked[i - 1].second >= ranked[i].second);
    }

    auto strict = recognizer.rank_intents("completely unrelated text", 0.99f, 6);
    CHECK(strict.empty());
  }

  SUBCASE("rank_intents with max_results limit") {
    recognizer.register_intent("turn on the lights");
    recognizer.register_intent("turn off the lights");
    recognizer.register_intent("what is the weather");

    auto ranked = recognizer.rank_intents("turn on the lights", 0.0f, 1);
    CHECK(ranked.size() == 1);
    CHECK(ranked[0].first == "turn on the lights");
  }
}

// ============================================================================
// Precision/Recall tests with real GemmaEmbeddingModel
// ============================================================================

struct IntentTestCase {
  std::string utterance;
  std::string expected_intent;  // empty = no match expected
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

TEST_CASE("intent-recognizer precision/recall with GemmaEmbeddingModel") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping Gemma intent tests - model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  float threshold = 0.6f;
  IntentRecognizer recognizer(make_options());

  std::map<std::string, std::string> intents = {
      {"lights_on", "turn on the lights"},
      {"lights_off", "turn off the lights"},
      {"weather", "what is the weather"},
      {"timer", "set a timer"},
      {"music_play", "play some music"},
      {"music_stop", "stop the music"},
      {"volume_up", "turn up the volume"},
      {"volume_down", "turn down the volume"},
  };

  std::map<std::string, std::string> phrase_to_intent;
  for (const auto &[intent_name, phrase] : intents) {
    recognizer.register_intent(phrase);
    phrase_to_intent[phrase] = intent_name;
  }

  SUBCASE("basic intent matching") {
    auto ranked = recognizer.rank_intents("turn on the lights", threshold, 1);
    REQUIRE(ranked.size() >= 1);
    CHECK(phrase_to_intent[ranked[0].first] == "lights_on");

    ranked = recognizer.rank_intents("what is the weather", threshold, 1);
    REQUIRE(ranked.size() >= 1);
    CHECK(phrase_to_intent[ranked[0].first] == "weather");

    ranked = recognizer.rank_intents("play some music", threshold, 1);
    REQUIRE(ranked.size() >= 1);
    CHECK(phrase_to_intent[ranked[0].first] == "music_play");
  }

  SUBCASE("precision/recall evaluation") {
    std::vector<IntentTestCase> test_cases = {
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
      auto ranked =
          recognizer.rank_intents(test_case.utterance, threshold, 1);

      bool expected_match = !test_case.expected_intent.empty();
      bool matched = !ranked.empty();
      std::string matched_intent =
          matched ? phrase_to_intent[ranked[0].first] : "";

      if (expected_match) {
        if (matched && matched_intent == test_case.expected_intent) {
          results.true_positives++;
        } else if (matched) {
          results.false_positives++;
          MESSAGE("WRONG INTENT: '", test_case.utterance, "' -> got '",
                  matched_intent, "', expected '", test_case.expected_intent,
                  "' (similarity: ", ranked[0].second, ")");
        } else {
          results.false_negatives++;
          MESSAGE("MISSED: '", test_case.utterance, "' -> expected '",
                  test_case.expected_intent, "'");
        }
      } else {
        if (!matched) {
          results.true_negatives++;
        } else {
          results.false_positives++;
          MESSAGE("FALSE POSITIVE: '", test_case.utterance, "' -> matched '",
                  matched_intent, "' (similarity: ", ranked[0].second,
                  "), expected no match");
        }
      }
    }

    MESSAGE("=== Intent Recognition Results (threshold=", threshold, ") ===");
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

  SUBCASE("intent discrimination") {
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
      auto ranked = recognizer.rank_intents(test.utterance, 0.0f, 1);
      REQUIRE(!ranked.empty());
      std::string matched_intent = phrase_to_intent[ranked[0].first];

      total_discriminations++;
      if (matched_intent == test.should_match) {
        correct_discriminations++;
      } else {
        MESSAGE("DISCRIMINATION FAIL: '", test.utterance, "' -> got '",
                matched_intent, "', expected '", test.should_match, "'");
      }

      CHECK(matched_intent != test.should_not_match);
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

    for (const auto &[utterance, expected_intent] : exact_matches) {
      auto ranked = recognizer.rank_intents(utterance, 0.0f, 1);
      REQUIRE(!ranked.empty());
      CHECK(phrase_to_intent[ranked[0].first] == expected_intent);
      CHECK(ranked[0].second >= 0.95f);

      MESSAGE("Exact match '", utterance, "' -> ", expected_intent,
              " (similarity: ", ranked[0].second, ")");
    }
  }
}

TEST_CASE("intent-recognizer register with pre-computed embedding") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  IntentRecognizer recognizer(make_options());

  SUBCASE("register with NULL embedding auto-computes") {
    recognizer.register_intent("turn on the lights", nullptr, 0, 0);
    CHECK(recognizer.get_intent_count() == 1);

    auto ranked = recognizer.rank_intents("turn on the lights", 0.0f, 6);
    CHECK(ranked.size() >= 1);
    CHECK(ranked[0].first == "turn on the lights");
    CHECK(ranked[0].second >= 0.95f);
  }

  SUBCASE("register with pre-computed embedding") {
    std::vector<float> emb =
        recognizer.calculate_embedding("turn on the lights");
    CHECK(!emb.empty());

    recognizer.register_intent("turn on the lights", emb.data(),
                               static_cast<uint64_t>(emb.size()), 0);
    CHECK(recognizer.get_intent_count() == 1);

    auto ranked = recognizer.rank_intents("turn on the lights", 0.0f, 6);
    CHECK(ranked.size() >= 1);
    CHECK(ranked[0].second >= 0.95f);
  }

  SUBCASE("update existing intent preserves count") {
    recognizer.register_intent("hello", nullptr, 0, 0);
    CHECK(recognizer.get_intent_count() == 1);

    recognizer.register_intent("hello", nullptr, 0, 5);
    CHECK(recognizer.get_intent_count() == 1);
  }
}

TEST_CASE("intent-recognizer priority ranking") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  IntentRecognizer recognizer(make_options());

  SUBCASE("higher priority intent ranks first regardless of similarity") {
    recognizer.register_intent("turn on the lights", nullptr, 0, 1);
    recognizer.register_intent("switch on the lamps", nullptr, 0, 10);

    auto ranked = recognizer.rank_intents("turn on the lights", 0.0f, 6);
    REQUIRE(ranked.size() == 2);
    CHECK(ranked[0].first == "switch on the lamps");
    CHECK(ranked[1].first == "turn on the lights");
  }

  SUBCASE("equal priority falls back to similarity ordering") {
    recognizer.register_intent("turn on the lights", nullptr, 0, 0);
    recognizer.register_intent("what is the weather", nullptr, 0, 0);

    auto ranked = recognizer.rank_intents("turn on the lights", 0.0f, 6);
    REQUIRE(ranked.size() >= 2);
    CHECK(ranked[0].first == "turn on the lights");
    CHECK(ranked[0].second >= ranked[1].second);
  }
}

TEST_CASE("intent-recognizer calculate_embedding") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  IntentRecognizer recognizer(make_options());

  SUBCASE("returns non-empty embedding") {
    auto emb = recognizer.calculate_embedding("hello world");
    CHECK(!emb.empty());
    CHECK(emb.size() > 0);
  }

  SUBCASE("get_embedding_size returns correct dimension") {
    size_t dim = recognizer.get_embedding_size();
    CHECK(dim > 0);
    auto emb = recognizer.calculate_embedding("test");
    CHECK(emb.size() == dim);
  }

  SUBCASE("same text produces same embedding") {
    auto emb1 = recognizer.calculate_embedding("hello");
    auto emb2 = recognizer.calculate_embedding("hello");
    REQUIRE(emb1.size() == emb2.size());
    for (size_t i = 0; i < emb1.size(); ++i) {
      CHECK(emb1[i] == doctest::Approx(emb2[i]));
    }
  }
}

TEST_CASE("C API intent registration with embedding and priority") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  int32_t handle = moonshine_create_intent_recognizer(
      EMBEDDING_MODEL_DIR.c_str(), MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
      "q4");
  REQUIRE(handle >= 0);

  SUBCASE("register with NULL embedding succeeds") {
    int32_t err = moonshine_register_intent(handle, "hello", nullptr, 0, 0);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(moonshine_get_intent_count(handle) == 1);
  }

  SUBCASE("register with nullptr canonical_phrase fails") {
    int32_t err = moonshine_register_intent(handle, nullptr, nullptr, 0, 0);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("register multiple intents with different priorities") {
    CHECK(moonshine_register_intent(handle, "lights on", nullptr, 0, 1) ==
          MOONSHINE_ERROR_NONE);
    CHECK(moonshine_register_intent(handle, "lights off", nullptr, 0, 5) ==
          MOONSHINE_ERROR_NONE);
    CHECK(moonshine_get_intent_count(handle) == 2);
  }

  SUBCASE("unregister and clear work") {
    moonshine_register_intent(handle, "a", nullptr, 0, 0);
    moonshine_register_intent(handle, "b", nullptr, 0, 0);
    CHECK(moonshine_get_intent_count(handle) == 2);

    CHECK(moonshine_unregister_intent(handle, "a") == MOONSHINE_ERROR_NONE);
    CHECK(moonshine_get_intent_count(handle) == 1);

    CHECK(moonshine_clear_intents(handle) == MOONSHINE_ERROR_NONE);
    CHECK(moonshine_get_intent_count(handle) == 0);
  }

  moonshine_free_intent_recognizer(handle);
}

TEST_CASE("C API moonshine_calculate_intent_embedding") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  int32_t handle = moonshine_create_intent_recognizer(
      EMBEDDING_MODEL_DIR.c_str(), MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
      "q4");
  REQUIRE(handle >= 0);

  SUBCASE("basic embedding calculation") {
    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_intent_embedding(
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
    moonshine_free_intent_embedding(embedding);
  }

  SUBCASE("null sentence returns error") {
    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_intent_embedding(
        handle, nullptr, &embedding, &embedding_size, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("null out_embedding returns error") {
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_intent_embedding(
        handle, "hello", nullptr, &embedding_size, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("null out_embedding_size returns error") {
    float *embedding = nullptr;
    int32_t err = moonshine_calculate_intent_embedding(
        handle, "hello", &embedding, nullptr, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("invalid handle returns error") {
    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_intent_embedding(
        -1, "hello", &embedding, &embedding_size, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_HANDLE);
  }

  SUBCASE("round-trip: compute embedding then register with it") {
    float *embedding = nullptr;
    uint64_t embedding_size = 0;
    int32_t err = moonshine_calculate_intent_embedding(
        handle, "turn on the lights", &embedding, &embedding_size, nullptr);
    REQUIRE(err == MOONSHINE_ERROR_NONE);
    REQUIRE(embedding != nullptr);
    REQUIRE(embedding_size > 0);

    err = moonshine_register_intent(handle, "turn on the lights",
                                    embedding, embedding_size, 0);
    CHECK(err == MOONSHINE_ERROR_NONE);
    moonshine_free_intent_embedding(embedding);

    moonshine_intent_match_t *matches = nullptr;
    uint64_t count = 0;
    err = moonshine_get_closest_intents(handle, "turn on the lights", 0.9f,
                                        &matches, &count);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(count >= 1);
    if (count > 0) {
      CHECK(std::string(matches[0].canonical_phrase) == "turn on the lights");
      CHECK(matches[0].similarity >= 0.95f);
    }
    moonshine_free_intent_matches(matches, count);
  }

  moonshine_free_intent_recognizer(handle);
}

TEST_CASE("C API moonshine_free_intent_embedding") {
  SUBCASE("safe on nullptr") {
    moonshine_free_intent_embedding(nullptr);
  }

  SUBCASE("frees malloc-allocated buffer") {
    float *buf = static_cast<float *>(std::malloc(768 * sizeof(float)));
    REQUIRE(buf != nullptr);
    moonshine_free_intent_embedding(buf);
  }
}

TEST_CASE("C API moonshine_calculate_embedding_distance") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  int32_t handle = moonshine_create_intent_recognizer(
      EMBEDDING_MODEL_DIR.c_str(), MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
      "q4");
  REQUIRE(handle >= 0);

  SUBCASE("identical embeddings have similarity ~1.0") {
    float *emb = nullptr;
    uint64_t emb_size = 0;
    int32_t err = moonshine_calculate_intent_embedding(
        handle, "hello world", &emb, &emb_size, nullptr);
    REQUIRE(err == MOONSHINE_ERROR_NONE);
    REQUIRE(emb != nullptr);

    float similarity = 0.0f;
    err = moonshine_calculate_embedding_distance(handle, emb, emb,
                                                  emb_size, &similarity);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(similarity > 0.99f);
    moonshine_free_intent_embedding(emb);
  }

  SUBCASE("similar sentences have high similarity") {
    float *emb_a = nullptr;
    float *emb_b = nullptr;
    uint64_t size_a = 0, size_b = 0;
    moonshine_calculate_intent_embedding(handle, "turn on the lights",
                                         &emb_a, &size_a, nullptr);
    moonshine_calculate_intent_embedding(handle, "switch on the lamps",
                                         &emb_b, &size_b, nullptr);
    REQUIRE(emb_a != nullptr);
    REQUIRE(emb_b != nullptr);
    REQUIRE(size_a == size_b);

    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(
        handle, emb_a, emb_b, size_a, &similarity);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(similarity > 0.7f);
    moonshine_free_intent_embedding(emb_a);
    moonshine_free_intent_embedding(emb_b);
  }

  SUBCASE("dissimilar sentences have low similarity") {
    float *emb_a = nullptr;
    float *emb_b = nullptr;
    uint64_t size_a = 0, size_b = 0;
    moonshine_calculate_intent_embedding(handle, "turn on the lights",
                                         &emb_a, &size_a, nullptr);
    moonshine_calculate_intent_embedding(handle, "the stock market crashed",
                                         &emb_b, &size_b, nullptr);
    REQUIRE(emb_a != nullptr);
    REQUIRE(emb_b != nullptr);
    REQUIRE(size_a == size_b);

    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(
        handle, emb_a, emb_b, size_a, &similarity);
    CHECK(err == MOONSHINE_ERROR_NONE);
    CHECK(similarity < 0.5f);
    moonshine_free_intent_embedding(emb_a);
    moonshine_free_intent_embedding(emb_b);
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
    int32_t err = moonshine_calculate_embedding_distance(
        handle, &dummy, &dummy, 1, nullptr);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("zero embedding_size returns error") {
    float dummy = 1.0f;
    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(
        handle, &dummy, &dummy, 0, &similarity);
    CHECK(err == MOONSHINE_ERROR_INVALID_ARGUMENT);
  }

  SUBCASE("invalid handle returns error") {
    float dummy = 1.0f;
    float similarity = 0.0f;
    int32_t err = moonshine_calculate_embedding_distance(
        -1, &dummy, &dummy, 1, &similarity);
    CHECK(err == MOONSHINE_ERROR_INVALID_HANDLE);
  }

  moonshine_free_intent_recognizer(handle);
}

TEST_CASE("C API moonshine_get_closest_intents with priority") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  int32_t handle = moonshine_create_intent_recognizer(
      EMBEDDING_MODEL_DIR.c_str(), MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M,
      "q4");
  REQUIRE(handle >= 0);

  moonshine_register_intent(handle, "turn on the lights", nullptr, 0, 1);
  moonshine_register_intent(handle, "switch on the lamps", nullptr, 0, 10);

  moonshine_intent_match_t *matches = nullptr;
  uint64_t count = 0;
  int32_t err = moonshine_get_closest_intents(handle, "turn on the lights",
                                              0.0f, &matches, &count);
  CHECK(err == MOONSHINE_ERROR_NONE);
  REQUIRE(count == 2);
  CHECK(std::string(matches[0].canonical_phrase) == "switch on the lamps");
  CHECK(std::string(matches[1].canonical_phrase) == "turn on the lights");

  moonshine_free_intent_matches(matches, count);
  moonshine_free_intent_recognizer(handle);
}
