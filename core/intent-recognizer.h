#ifndef INTENT_RECOGNIZER_H
#define INTENT_RECOGNIZER_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "embedding-model.h"

/**
 * Supported embedding model architectures.
 */
enum class EmbeddingModelArch {
  GEMMA_300M = 0,  // embeddinggemma-300m (768-dim embeddings)
};

/**
 * Options for configuring an IntentRecognizer.
 */
struct IntentRecognizerOptions {
  // Path to the embedding model directory
  std::string model_path;

  // Embedding model architecture
  EmbeddingModelArch model_arch = EmbeddingModelArch::GEMMA_300M;

  // Model variant: "fp32", "fp16", "q8", "q4", or "q4f16"
  std::string model_variant = "q4";
};

/**
 * Represents a registered intent with its trigger phrase and embedding.
 */
struct Intent {
  std::string trigger_phrase;
  std::vector<float> embedding;
  int32_t priority = 0;
};

/**
 * IntentRecognizer matches utterances against registered canonical phrases
 * using semantic similarity. Use rank_intents() or the C API
 * moonshine_get_closest_intents() to retrieve ranked matches.
 */
class IntentRecognizer {
 public:
  /**
   * Construct an IntentRecognizer from options.
   * The embedding model will be loaded from the path specified in options.
   * @param options The configuration options for the recognizer.
   */
  explicit IntentRecognizer(const IntentRecognizerOptions &options);

  /**
   * Destructor - cleans up owned embedding model.
   */
  ~IntentRecognizer();

  /**
   * Register an intent with a trigger phrase.
   * @param trigger_phrase The canonical phrase for this intent.
   */
  void register_intent(const std::string &trigger_phrase);

  /**
   * Register an intent with an optional pre-computed embedding and priority.
   * @param trigger_phrase The canonical phrase for this intent.
   * @param embedding Optional pre-computed embedding (NULL to auto-compute).
   * @param embedding_size Number of floats in the embedding array.
   * @param priority Higher priority intents rank above lower priority ones
   *                 within the tolerance threshold, even if similarity is lower.
   */
  void register_intent(const std::string &trigger_phrase,
                       const float *embedding, uint64_t embedding_size,
                       int32_t priority);

  /**
   * Remove a registered intent.
   * @param trigger_phrase The trigger phrase of the intent to remove.
   * @return True if the intent was found and removed, false otherwise.
   */
  bool unregister_intent(const std::string &trigger_phrase);

  /**
   * Rank registered intents by semantic similarity to an utterance.
   * @param utterance The text to match (empty yields no results).
   * @param threshold Minimum similarity (inclusive) for a candidate to be kept.
   * @param max_results Maximum number of matches to return (e.g. 6).
   * @return Matches with trigger phrase and score, sorted by priority
   *         descending then similarity descending.
   */
  std::vector<std::pair<std::string, float>> rank_intents(
      const std::string &utterance, float threshold, size_t max_results);

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
   * Calculate the embedding for a given sentence using the loaded model.
   * @param sentence The input text.
   * @return The embedding vector.
   */
  std::vector<float> calculate_embedding(const std::string &sentence) const;

  /**
   * Compute cosine similarity between two precomputed embeddings.
   * @param a The first embedding vector.
   * @param b The second embedding vector.
   * @return Cosine similarity in [-1, 1].
   */
  float calculate_similarity(const std::vector<float> &a,
                             const std::vector<float> &b) const;

  /**
   * Get the embedding dimension of the loaded model.
   * @return The number of floats per embedding.
   */
  size_t get_embedding_size() const;

 private:
  std::unique_ptr<EmbeddingModel> embedding_model_;
  std::vector<Intent> intents_;
  mutable std::mutex mutex_;
};

#endif  // INTENT_RECOGNIZER_H
