#ifndef EMBEDDING_MODEL_H
#define EMBEDDING_MODEL_H

#include <cmath>
#include <string>
#include <vector>

/**
 * Abstract interface for embedding models that convert text to vector
 * representations.
 */
class EmbeddingModel {
 public:
  virtual ~EmbeddingModel() = default;

  /**
   * Get the embedding vector for the given text.
   * @param text The input text to embed.
   * @return A vector of floats representing the embedding.
   */
  virtual std::vector<float> get_embeddings(const std::string &text) = 0;

  /**
   * Compute the similarity between two text strings.
   * @param a The first text string.
   * @param b The second text string.
   * @return A similarity score between -1 and 1 (cosine similarity).
   */
  float get_similarity(const std::string &a, const std::string &b) {
    std::vector<float> embedding_a = get_embeddings(a);
    std::vector<float> embedding_b = get_embeddings(b);
    return cosine_similarity(embedding_a, embedding_b);
  }

  /**
   * Compute the similarity between a text string and a precomputed embedding.
   * @param text The text string to compare.
   * @param embedding The precomputed embedding vector.
   * @return A similarity score between -1 and 1 (cosine similarity).
   */
  float get_similarity(const std::string &text,
                       const std::vector<float> &embedding) {
    std::vector<float> text_embedding = get_embeddings(text);
    return cosine_similarity(text_embedding, embedding);
  }

  /**
   * Compute the similarity between two precomputed embeddings.
   * @param embedding_a The first embedding vector.
   * @param embedding_b The second embedding vector.
   * @return A similarity score between -1 and 1 (cosine similarity).
   */
  float get_similarity(const std::vector<float> &embedding_a,
                       const std::vector<float> &embedding_b) {
    return cosine_similarity(embedding_a, embedding_b);
  }

 private:
  /**
   * Compute the cosine similarity between two vectors.
   * @param a The first vector.
   * @param b The second vector.
   * @return The cosine similarity between -1 and 1.
   */
  float cosine_similarity(const std::vector<float> &a,
                          const std::vector<float> &b) {
    if (a.size() != b.size() || a.empty()) {
      return 0.0f;
    }

    float dot_product = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
      dot_product += a[i] * b[i];
      norm_a += a[i] * a[i];
      norm_b += b[i] * b[i];
    }

    float denominator = std::sqrt(norm_a) * std::sqrt(norm_b);
    if (denominator == 0.0f) {
      return 0.0f;
    }

    return dot_product / denominator;
  }
};

#endif  // EMBEDDING_MODEL_H
