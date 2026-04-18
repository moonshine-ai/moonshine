#include "intent-recognizer.h"

#include <algorithm>
#include <limits>
#include <stdexcept>

#include "gemma-embedding-model.h"

namespace {

std::unique_ptr<EmbeddingModel> create_embedding_model(
    const IntentRecognizerOptions &options) {
  switch (options.model_arch) {
    case EmbeddingModelArch::GEMMA_300M: {
      auto model = std::make_unique<GemmaEmbeddingModel>();
      int result =
          model->load(options.model_path.c_str(), options.model_variant.c_str());
      if (result != 0) {
        throw std::runtime_error("Failed to load embedding model from: " +
                                 options.model_path);
      }
      return model;
    }
    default:
      throw std::runtime_error("Unknown embedding model architecture");
  }
}

}  // namespace

IntentRecognizer::IntentRecognizer(const IntentRecognizerOptions &options)
    : embedding_model_(create_embedding_model(options)) {}

IntentRecognizer::~IntentRecognizer() = default;

void IntentRecognizer::register_intent(const std::string &trigger_phrase) {
  register_intent(trigger_phrase, nullptr, 0, 0);
}

void IntentRecognizer::register_intent(const std::string &trigger_phrase,
                                       const float *embedding,
                                       uint64_t embedding_size,
                                       int32_t priority) {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<float> emb;
  if (embedding != nullptr && embedding_size > 0) {
    emb.assign(embedding, embedding + embedding_size);
  } else {
    emb = embedding_model_->get_embeddings(trigger_phrase);
  }

  for (auto &intent : intents_) {
    if (intent.trigger_phrase == trigger_phrase) {
      intent.embedding = std::move(emb);
      intent.priority = priority;
      return;
    }
  }

  Intent intent;
  intent.trigger_phrase = trigger_phrase;
  intent.embedding = std::move(emb);
  intent.priority = priority;
  intents_.push_back(std::move(intent));
}

bool IntentRecognizer::unregister_intent(const std::string &trigger_phrase) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = std::find_if(intents_.begin(), intents_.end(),
                         [&trigger_phrase](const Intent &intent) {
                           return intent.trigger_phrase == trigger_phrase;
                         });

  if (it != intents_.end()) {
    intents_.erase(it);
    return true;
  }

  return false;
}

std::vector<std::pair<std::string, float>> IntentRecognizer::rank_intents(
    const std::string &utterance, float threshold, size_t max_results) {
  struct RankedEntry {
    std::string phrase;
    float similarity;
    int32_t priority;
  };

  std::vector<std::pair<std::string, float>> ranked;
  if (utterance.empty() || max_results == 0) {
    return ranked;
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (intents_.empty()) {
    return ranked;
  }

  std::vector<float> utterance_embedding =
      embedding_model_->get_embeddings(utterance);

  std::vector<RankedEntry> entries;
  entries.reserve(intents_.size());
  for (const auto &intent : intents_) {
    float similarity =
        embedding_model_->get_similarity(utterance_embedding, intent.embedding);
    if (similarity >= threshold) {
      entries.push_back({intent.trigger_phrase, similarity, intent.priority});
    }
  }

  std::sort(entries.begin(), entries.end(), [](const auto &a, const auto &b) {
    if (a.priority != b.priority) return a.priority > b.priority;
    return a.similarity > b.similarity;
  });
  if (entries.size() > max_results) {
    entries.resize(max_results);
  }

  ranked.reserve(entries.size());
  for (auto &e : entries) {
    ranked.emplace_back(std::move(e.phrase), e.similarity);
  }
  return ranked;
}

size_t IntentRecognizer::get_intent_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return intents_.size();
}

void IntentRecognizer::clear_intents() {
  std::lock_guard<std::mutex> lock(mutex_);
  intents_.clear();
}

std::vector<float> IntentRecognizer::calculate_embedding(
    const std::string &sentence) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return embedding_model_->get_embeddings(sentence);
}

size_t IntentRecognizer::get_embedding_size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  auto probe = embedding_model_->get_embeddings("");
  return probe.size();
}
