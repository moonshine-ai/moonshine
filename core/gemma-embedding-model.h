#ifndef GEMMA_EMBEDDING_MODEL_H
#define GEMMA_EMBEDDING_MODEL_H

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "bin-tokenizer.h"
#include "embedding-model.h"
#include "moonshine-ort-allocator.h"
#include "onnxruntime_c_api.h"

/**
 * Configuration for the Gemma embedding model.
 */
struct GemmaEmbeddingConfig {
  int embedding_dim = 768;    // Output embedding dimension
  int max_seq_length = 2048;  // Maximum input sequence length
  int vocab_size = 262144;    // Vocabulary size
  int bos_token_id = 2;       // Beginning of sequence token
  int eos_token_id = 1;       // End of sequence token
  int pad_token_id = 0;       // Padding token
};

/**
 * Gemma Embedding Model implementation using ONNX Runtime C API.
 * Produces vector representations of text for search, retrieval,
 * classification, clustering, and semantic similarity tasks.
 */
class GemmaEmbeddingModel : public EmbeddingModel {
 public:
  /**
   * Construct a new Gemma Embedding Model.
   */
  GemmaEmbeddingModel();

  /**
   * Destructor - releases ONNX Runtime resources.
   */
  ~GemmaEmbeddingModel();

  /**
   * Load the model from a directory containing model files.
   * @param model_dir Directory containing model.onnx (or quantized variants)
   *                  and tokenizer.bin.
   * @param model_variant Optional variant name: "fp32", "fp16", "q8", "q4",
   *                      "q4f16". Default is "q4" for efficiency.
   * @return 0 on success, non-zero on failure.
   */
  int load(const char *model_dir, const char *model_variant = "q4");

  /**
   * Load the model from memory buffers.
   * @param model_data Pointer to ONNX model data.
   * @param model_data_size Size of model data in bytes.
   * @param tokenizer_data Pointer to tokenizer.bin data.
   * @param tokenizer_data_size Size of tokenizer data in bytes.
   * @return 0 on success, non-zero on failure.
   */
  int load_from_memory(const uint8_t *model_data, size_t model_data_size,
                       const uint8_t *tokenizer_data,
                       size_t tokenizer_data_size);

  /**
   * Get the embedding vector for the given text.
   * @param text The input text to embed.
   * @return A vector of floats representing the embedding.
   */
  std::vector<float> get_embeddings(const std::string &text) override;

  /**
   * Get embeddings with a specific prefix (for query vs document embeddings).
   * @param text The input text to embed.
   * @param prefix The prefix to prepend (e.g., "task: search result | query:
   * ").
   * @return A vector of floats representing the embedding.
   */
  std::vector<float> get_embeddings_with_prefix(const std::string &text,
                                                const std::string &prefix);

  /**
   * Get query embeddings (uses query prefix).
   * @param query The query text.
   * @return A vector of floats representing the query embedding.
   */
  std::vector<float> get_query_embeddings(const std::string &query);

  /**
   * Get document embeddings (uses document prefix).
   * @param document The document text.
   * @return A vector of floats representing the document embedding.
   */
  std::vector<float> get_document_embeddings(const std::string &document);

  /**
   * Truncate an embedding to a smaller dimension using MRL.
   * @param embedding The original embedding.
   * @param target_dim Target dimension (128, 256, 512, or 768).
   * @return The truncated and renormalized embedding.
   */
  static std::vector<float> truncate_embedding(
      const std::vector<float> &embedding, int target_dim);

  /**
   * Check if the model is loaded.
   * @return True if the model is loaded and ready for inference.
   */
  bool is_loaded() const;

  /**
   * Get the model configuration.
   * @return The model configuration.
   */
  const GemmaEmbeddingConfig &get_config() const;

 private:
  // ONNX Runtime resources
  const OrtApi *ort_api_;
  OrtEnv *ort_env_;
  OrtSessionOptions *ort_session_options_;
  OrtMemoryInfo *ort_memory_info_;
  MoonshineOrtAllocator *ort_allocator_;
  OrtSession *session_;

  // Memory-mapped model data (for file loading)
  const char *mmapped_data_;
  size_t mmapped_data_size_;

  // Model configuration
  GemmaEmbeddingConfig config_;

  // Tokenizer
  BinTokenizer *tokenizer_;

  // Thread safety
  mutable std::mutex mutex_;

  // Prefixes for different embedding types
  static constexpr const char *QUERY_PREFIX = "task: search result | query: ";
  static constexpr const char *DOCUMENT_PREFIX = "title: none | text: ";

  /**
   * Load the tokenizer from a file path.
   * @param tokenizer_path Path to tokenizer.bin.
   * @return 0 on success, non-zero on failure.
   */
  int load_tokenizer(const char *tokenizer_path);

  /**
   * Load the tokenizer from memory.
   * @param data Pointer to tokenizer.bin data.
   * @param data_size Size of data in bytes.
   * @return 0 on success, non-zero on failure.
   */
  int load_tokenizer_from_memory(const uint8_t *data, size_t data_size);

  /**
   * Tokenize input text to token IDs.
   * @param text The input text.
   * @return Vector of token IDs.
   */
  std::vector<int64_t> tokenize(const std::string &text);

  /**
   * Run inference to get embeddings for token IDs.
   * @param input_ids The token IDs.
   * @param attention_mask The attention mask.
   * @return The embedding vector.
   */
  std::vector<float> run_inference(const std::vector<int64_t> &input_ids,
                                   const std::vector<int64_t> &attention_mask);

  /**
   * Normalize an embedding vector to unit length.
   * @param embedding The embedding to normalize (modified in place).
   */
  static void normalize_embedding(std::vector<float> &embedding);
};

#endif  // GEMMA_EMBEDDING_MODEL_H
