#ifndef TEXT_EMBEDDER_H
#define TEXT_EMBEDDER_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "embedding-model.h"

/**
 * Supported embedding model architectures.
 */
enum class EmbeddingModelArch {
  GEMMA_300M = 0,  // embeddinggemma-300m (768-dim embeddings)
};

/**
 * Options for configuring a TextEmbedder.
 */
struct TextEmbedderOptions {
  // Path to the embedding model directory (used when the in-memory buffers
  // below are not supplied).
  std::string model_path;

  // Embedding model architecture
  EmbeddingModelArch model_arch = EmbeddingModelArch::GEMMA_300M;

  // Model variant: "fp32", "fp16", "q8", "q4", or "q4f16"
  std::string model_variant = "q4";

  // Optional in-memory model source. When ``model_data`` is non-null the
  // embedder loads the embedding model from these buffers instead of
  // ``model_path``. ``model_data`` must be a self-contained model (an
  // all-in-one ``.ort``); ``tokenizer_data`` is the ``tokenizer.bin`` bytes.
  // The buffers only need to remain valid for the duration of construction -
  // both the ORT session and the tokenizer copy the bytes they need.
  const uint8_t *model_data = nullptr;
  size_t model_data_size = 0;
  const uint8_t *tokenizer_data = nullptr;
  size_t tokenizer_data_size = 0;
};

/**
 * TextEmbedder owns a loaded embedding model and turns text into vectors that
 * callers can compare themselves. Phrase matching lives in the language
 * bindings' AgentFlow implementations, which embed their candidate phrases
 * once and score utterances against them with calculate_similarity().
 */
class TextEmbedder {
 public:
  /**
   * Construct a TextEmbedder from options.
   * The embedding model will be loaded from the path specified in options.
   * @param options The configuration options for the embedder.
   */
  explicit TextEmbedder(const TextEmbedderOptions &options);

  /**
   * Destructor - cleans up owned embedding model.
   */
  ~TextEmbedder();

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
  mutable std::mutex mutex_;
};

#endif  // TEXT_EMBEDDER_H
