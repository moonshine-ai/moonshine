#include "intent-recognizer.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>

#include "gemma-embedding-model.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

// Path to the Gemma embedding model
static const std::string EMBEDDING_MODEL_DIR = "embeddinggemma-300m-ONNX";

/**
 * Helper function to create IntentRecognizerOptions with default model path.
 */
IntentRecognizerOptions make_options(float threshold = 0.7f) {
  IntentRecognizerOptions options;
  options.model_path = EMBEDDING_MODEL_DIR;
  options.model_arch = EmbeddingModelArch::GEMMA_300M;
  options.model_variant = "q4";
  options.threshold = threshold;
  return options;
}

/**
 * Helper function to check if the embedding model is available.
 */
bool embedding_model_available() {
  return std::filesystem::exists(EMBEDDING_MODEL_DIR);
}

TEST_CASE("intent-recognizer unit tests") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  IntentRecognizer recognizer(make_options(0.7f));

  SUBCASE("register and count intents") {
    CHECK(recognizer.get_intent_count() == 0);

    recognizer.register_intent("turn on the lights",
                               [](const std::string &, float) {});
    CHECK(recognizer.get_intent_count() == 1);

    recognizer.register_intent("turn off the lights",
                               [](const std::string &, float) {});
    CHECK(recognizer.get_intent_count() == 2);
  }

  SUBCASE("unregister intent") {
    recognizer.register_intent("turn on the lights",
                               [](const std::string &, float) {});
    CHECK(recognizer.get_intent_count() == 1);

    bool removed = recognizer.unregister_intent("turn on the lights");
    CHECK(removed == true);
    CHECK(recognizer.get_intent_count() == 0);
  }

  SUBCASE("clear intents") {
    recognizer.register_intent("intent1", [](const std::string &, float) {});
    recognizer.register_intent("intent2", [](const std::string &, float) {});
    CHECK(recognizer.get_intent_count() == 2);

    recognizer.clear_intents();
    CHECK(recognizer.get_intent_count() == 0);
  }

  SUBCASE("threshold getter and setter") {
    CHECK(recognizer.get_threshold() == doctest::Approx(0.7f));
    recognizer.set_threshold(0.8f);
    CHECK(recognizer.get_threshold() == doctest::Approx(0.8f));
  }

  SUBCASE("process_utterance triggers callback for exact match") {
    bool callback_triggered = false;
    recognizer.register_intent(
        "turn on the lights",
        [&](const std::string &, float) { callback_triggered = true; });

    bool matched = recognizer.process_utterance("turn on the lights");
    CHECK(matched == true);
    CHECK(callback_triggered == true);
  }

  SUBCASE("empty utterance does not trigger") {
    bool callback_triggered = false;
    recognizer.register_intent(
        "turn on the lights",
        [&](const std::string &, float) { callback_triggered = true; });

    bool matched = recognizer.process_utterance("");
    CHECK(matched == false);
    CHECK(callback_triggered == false);
  }
}

TEST_CASE("intent-recognizer with transcript") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping tests - embedding model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  IntentRecognizer recognizer(make_options(0.7f));

  SUBCASE("process_transcript handles null transcript") {
    recognizer.process_transcript(nullptr);
  }

  SUBCASE("process_transcript only processes complete lines") {
    int callback_count = 0;
    recognizer.register_intent(
        "hello world", [&](const std::string &, float) { callback_count++; });

    transcript_line_t lines[2];
    lines[0].text = const_cast<char *>("hello world");
    lines[0].is_complete = false;
    lines[0].id = 1;

    lines[1].text = const_cast<char *>("hello world");
    lines[1].is_complete = true;
    lines[1].id = 2;

    struct transcript_t transcript = {.lines = lines, .line_count = 2};
    recognizer.process_transcript(&transcript);

    CHECK(callback_count == 1);
  }

  SUBCASE("process_transcript does not reprocess same line") {
    int callback_count = 0;
    recognizer.register_intent(
        "hello world", [&](const std::string &, float) { callback_count++; });

    transcript_line_t lines[1];
    lines[0].text = const_cast<char *>("hello world");
    lines[0].is_complete = true;
    lines[0].id = 1;

    struct transcript_t transcript = {.lines = lines, .line_count = 1};

    recognizer.process_transcript(&transcript);
    CHECK(callback_count == 1);

    recognizer.process_transcript(&transcript);
    CHECK(callback_count == 1);  // Should still be 1
  }
}

// ============================================================================
// Precision/Recall tests with real GemmaEmbeddingModel
// ============================================================================

/**
 * Test data structure for intent recognition evaluation.
 */
struct IntentTestCase {
  std::string utterance;  // The test utterance
  std::string
      expected_intent;  // Expected intent name (empty = no match expected)
};

/**
 * Calculate precision and recall for intent recognition.
 *
 * Precision = True Positives / (True Positives + False Positives)
 * Recall = True Positives / (True Positives + False Negatives)
 */
struct PrecisionRecallResult {
  int true_positives = 0;   // Correctly matched to expected intent
  int false_positives = 0;  // Matched to wrong intent or matched when shouldn't
  int false_negatives = 0;  // Did not match when should have
  int true_negatives = 0;   // Correctly did not match

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
  // Check if Gemma model is available
  if (!embedding_model_available()) {
    MESSAGE("Skipping Gemma intent tests - model not found at: ",
            EMBEDDING_MODEL_DIR);
    return;
  }

  // Use a threshold of 0.6 for intent matching
  float threshold = 0.6f;
  IntentRecognizer recognizer(make_options(threshold));

  // Define intents with canonical phrases
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

  // Track which intent was triggered
  std::string triggered_intent;
  float triggered_similarity = 0.0f;

  // Register all intents (copy intent_name so lambda can capture it; structured
  // bindings cannot be captured)
  for (const auto &[intent_name, phrase] : intents) {
    std::string captured_intent = intent_name;
    recognizer.register_intent(
        phrase, [&triggered_intent, &triggered_similarity,
                 captured_intent](const std::string &, float similarity) {
          triggered_intent = captured_intent;
          triggered_similarity = similarity;
        });
  }

  SUBCASE("basic intent matching") {
    // Test exact matches
    triggered_intent.clear();
    recognizer.process_utterance("turn on the lights");
    CHECK(triggered_intent == "lights_on");

    triggered_intent.clear();
    recognizer.process_utterance("what is the weather");
    CHECK(triggered_intent == "weather");

    triggered_intent.clear();
    recognizer.process_utterance("play some music");
    CHECK(triggered_intent == "music_play");
  }

  SUBCASE("precision/recall evaluation") {
    // Test cases: utterances and their expected intents
    std::vector<IntentTestCase> test_cases = {
        // lights_on variations
        {"turn on the lights", "lights_on"},
        {"switch on the lights", "lights_on"},
        {"lights on please", "lights_on"},
        {"can you turn the lights on", "lights_on"},
        {"illuminate the room", "lights_on"},

        // lights_off variations
        {"turn off the lights", "lights_off"},
        {"switch off the lights", "lights_off"},
        {"lights off", "lights_off"},
        {"kill the lights", "lights_off"},

        // weather variations
        {"what is the weather", "weather"},
        {"how is the weather today", "weather"},
        {"what's the forecast", "weather"},
        {"is it going to rain", "weather"},
        {"weather report please", "weather"},

        // timer variations
        {"set a timer", "timer"},
        {"start a timer for 5 minutes", "timer"},
        {"timer for 10 minutes", "timer"},
        {"set an alarm", "timer"},

        // music_play variations
        {"play some music", "music_play"},
        {"play a song", "music_play"},
        {"start playing music", "music_play"},
        {"put on some tunes", "music_play"},

        // music_stop variations
        {"stop the music", "music_stop"},
        {"pause the music", "music_stop"},
        {"stop playing", "music_stop"},

        // volume_up variations
        {"turn up the volume", "volume_up"},
        {"louder please", "volume_up"},
        {"increase the volume", "volume_up"},
        {"volume up", "volume_up"},

        // volume_down variations
        {"turn down the volume", "volume_down"},
        {"quieter please", "volume_down"},
        {"decrease the volume", "volume_down"},
        {"volume down", "volume_down"},

        // Out of scope (should NOT match any intent)
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
      triggered_intent.clear();
      triggered_similarity = 0.0f;

      bool matched = recognizer.process_utterance(test_case.utterance);

      bool expected_match = !test_case.expected_intent.empty();
      bool correct_intent = (triggered_intent == test_case.expected_intent);

      if (expected_match) {
        if (matched && correct_intent) {
          results.true_positives++;
        } else if (matched && !correct_intent) {
          results.false_positives++;
          MESSAGE("WRONG INTENT: '", test_case.utterance, "' -> got '",
                  triggered_intent, "', expected '", test_case.expected_intent,
                  "' (similarity: ", triggered_similarity, ")");
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
                  triggered_intent, "' (similarity: ", triggered_similarity,
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

    // Check that we achieve reasonable precision and recall
    // These thresholds can be adjusted based on expected model performance
    CHECK(results.precision() >= 0.7f);
    CHECK(results.recall() >= 0.5f);
    CHECK(results.f1_score() >= 0.5f);
  }

  SUBCASE("intent discrimination") {
    // Test that similar but different intents are correctly distinguished
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
      triggered_intent.clear();
      recognizer.process_utterance(test.utterance);

      total_discriminations++;
      if (triggered_intent == test.should_match) {
        correct_discriminations++;
      } else {
        MESSAGE("DISCRIMINATION FAIL: '", test.utterance, "' -> got '",
                triggered_intent, "', expected '", test.should_match, "'");
      }

      // Also check it's not matching the wrong intent
      CHECK(triggered_intent != test.should_not_match);
    }

    float discrimination_accuracy =
        static_cast<float>(correct_discriminations) / total_discriminations;
    MESSAGE("Discrimination accuracy: ", discrimination_accuracy, " (",
            correct_discriminations, "/", total_discriminations, ")");

    CHECK(discrimination_accuracy >= 0.8f);
  }

  SUBCASE("similarity scores for matching intents") {
    // Verify that matching intents have reasonable similarity scores
    std::vector<std::pair<std::string, std::string>> exact_matches = {
        {"turn on the lights", "lights_on"},
        {"turn off the lights", "lights_off"},
        {"what is the weather", "weather"},
        {"set a timer", "timer"},
        {"play some music", "music_play"},
    };

    for (const auto &[utterance, expected_intent] : exact_matches) {
      triggered_intent.clear();
      triggered_similarity = 0.0f;

      recognizer.process_utterance(utterance);

      CHECK(triggered_intent == expected_intent);
      // Exact matches should have very high similarity
      CHECK(triggered_similarity >= 0.95f);

      MESSAGE("Exact match '", utterance, "' -> ", expected_intent,
              " (similarity: ", triggered_similarity, ")");
    }
  }
}

TEST_CASE("intent-recognizer threshold tuning") {
  if (!embedding_model_available()) {
    MESSAGE("Skipping threshold tuning tests - model not found");
    return;
  }

  SUBCASE("evaluate different thresholds") {
    std::vector<float> thresholds = {0.5f, 0.55f, 0.6f, 0.65f,
                                     0.7f, 0.75f, 0.8f};

    // Simple test set
    std::vector<std::pair<std::string, bool>> test_utterances = {
        {"turn on the lights", true},   // Should match
        {"switch on the light", true},  // Should match (variation)
        {"hello world", false},         // Should not match
        {"tell me a joke", false},      // Should not match
    };

    MESSAGE("=== Threshold Evaluation ===");

    for (float threshold : thresholds) {
      IntentRecognizer recognizer(make_options(threshold));

      std::string triggered;
      recognizer.register_intent("turn on the lights",
                                 [&triggered](const std::string &, float) {
                                   triggered = "lights_on";
                                 });

      int correct = 0;
      for (const auto &[utterance, should_match] : test_utterances) {
        triggered.clear();
        bool matched = recognizer.process_utterance(utterance);

        if ((matched && should_match) || (!matched && !should_match)) {
          correct++;
        }
      }

      float accuracy = static_cast<float>(correct) / test_utterances.size();
      MESSAGE("Threshold ", threshold, ": accuracy = ", accuracy, " (", correct,
              "/", test_utterances.size(), ")");
    }
  }
}
