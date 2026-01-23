#ifndef INTENT_RECOGNIZER_H
#define INTENT_RECOGNIZER_H

#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include "embedding-model.h"
#include "transcriber.h"

/**
 * Callback function type for intent handlers.
 * The callback receives the matched utterance and the similarity score.
 */
using IntentCallback =
    std::function<void(const std::string &utterance, float similarity)>;

/**
 * Represents a registered intent with its trigger phrase, embedding, and
 * callback.
 */
struct Intent {
  std::string trigger_phrase;
  std::vector<float> embedding;
  IntentCallback callback;
};

/**
 * IntentRecognizer allows users to bind trigger phrases to callback functions.
 * When an utterance is received, it compares the utterance against all
 * registered trigger phrases and invokes the callback of the most similar one
 * if the similarity exceeds a threshold.
 */
class IntentRecognizer {
 public:
  /**
   * Construct an IntentRecognizer with an embedding model.
   * @param embedding_model The embedding model to use for computing
   * similarities.
   * @param threshold The minimum similarity threshold to trigger a callback
   * (default 0.7).
   */
  explicit IntentRecognizer(EmbeddingModel *embedding_model,
                            float threshold = 0.7f);

  /**
   * Construct an IntentRecognizer with an embedding model and a transcriber.
   * The recognizer will automatically process transcripts from the transcriber.
   * @param embedding_model The embedding model to use for computing
   * similarities.
   * @param transcriber The transcriber to get utterances from.
   * @param threshold The minimum similarity threshold to trigger a callback
   * (default 0.7).
   */
  IntentRecognizer(EmbeddingModel *embedding_model, Transcriber *transcriber,
                   float threshold = 0.7f);

  ~IntentRecognizer() = default;

  /**
   * Register an intent with a trigger phrase and callback.
   * @param trigger_phrase The phrase that triggers this intent.
   * @param callback The function to call when this intent is recognized.
   */
  void register_intent(const std::string &trigger_phrase,
                       IntentCallback callback);

  /**
   * Remove a registered intent.
   * @param trigger_phrase The trigger phrase of the intent to remove.
   * @return True if the intent was found and removed, false otherwise.
   */
  bool unregister_intent(const std::string &trigger_phrase);

  /**
   * Process an utterance and invoke the callback of the most similar intent
   * if the similarity exceeds the threshold.
   * @param utterance The utterance to process.
   * @return True if an intent was recognized and callback invoked, false
   * otherwise.
   */
  bool process_utterance(const std::string &utterance);

  /**
   * Process a transcript from the transcriber.
   * This will process all complete lines that haven't been processed yet.
   * @param transcript The transcript to process.
   */
  void process_transcript(const struct transcript_t *transcript);

  /**
   * Set the similarity threshold.
   * @param threshold The new threshold value (should be between 0 and 1).
   */
  void set_threshold(float threshold);

  /**
   * Get the current similarity threshold.
   * @return The current threshold value.
   */
  float get_threshold() const;

  /**
   * Get the number of registered intents.
   * @return The number of registered intents.
   */
  size_t get_intent_count() const;

  /**
   * Clear all registered intents.
   */
  void clear_intents();

  /**
   * Get the associated transcriber, if any.
   * @return Pointer to the transcriber, or nullptr if none.
   */
  Transcriber *get_transcriber() const;

 private:
  EmbeddingModel *embedding_model_;
  Transcriber *transcriber_;
  float threshold_;
  std::vector<Intent> intents_;
  mutable std::mutex mutex_;

  // Track which transcript lines have been processed
  std::vector<uint64_t> processed_line_ids_;

  /**
   * Find the best matching intent for an utterance.
   * @param utterance The utterance to match.
   * @param out_similarity Output parameter for the similarity score.
   * @return Pointer to the best matching intent, or nullptr if none found.
   */
  const Intent *find_best_intent(const std::string &utterance,
                                 float &out_similarity);
};

#endif  // INTENT_RECOGNIZER_H
