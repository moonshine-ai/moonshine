#include "text-embedder.h"

#include <stdexcept>

#include "gemma-embedding-model.h"

namespace {

std::unique_ptr<EmbeddingModel> load_embedding_model(
    const TextEmbedderOptions &options) {
  switch (options.model_arch) {
    case EmbeddingModelArch::GEMMA_300M: {
      auto model = std::make_unique<GemmaEmbeddingModel>();
      int result;
      if (options.model_data != nullptr && options.model_data_size > 0) {
        result = model->load_from_memory(
            options.model_data, options.model_data_size, options.tokenizer_data,
            options.tokenizer_data_size);
        if (result != 0) {
          throw std::runtime_error(
              "Failed to load embedding model from memory");
        }
      } else {
        result = model->load(options.model_path.c_str(),
                             options.model_variant.c_str());
        if (result != 0) {
          throw std::runtime_error("Failed to load embedding model from: " +
                                   options.model_path);
        }
      }
      return model;
    }
    default:
      throw std::runtime_error("Unknown embedding model architecture");
  }
}

}  // namespace

TextEmbedder::TextEmbedder(const TextEmbedderOptions &options)
    : embedding_model_(load_embedding_model(options)) {}

TextEmbedder::~TextEmbedder() = default;

std::vector<float> TextEmbedder::calculate_embedding(
    const std::string &sentence) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return embedding_model_->get_embeddings(sentence);
}

float TextEmbedder::calculate_similarity(const std::vector<float> &a,
                                         const std::vector<float> &b) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return embedding_model_->get_similarity(a, b);
}

size_t TextEmbedder::get_embedding_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto probe = embedding_model_->get_embeddings("");
  return probe.size();
}
