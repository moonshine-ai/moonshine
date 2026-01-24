#include "intent-recognizer.h"

#include <algorithm>
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
    : embedding_model_(create_embedding_model(options)),
      transcriber_(nullptr),
      threshold_(options.threshold) {}

IntentRecognizer::~IntentRecognizer() = default;

void IntentRecognizer::register_intent(const std::string &trigger_phrase,
                                       IntentCallback callback) {
  std::lock_guard<std::mutex> lock(mutex_);

  // Check if intent with this trigger phrase already exists
  for (auto &intent : intents_) {
    if (intent.trigger_phrase == trigger_phrase) {
      // Update existing intent
      intent.callback = callback;
      intent.embedding = embedding_model_->get_embeddings(trigger_phrase);
      return;
    }
  }

  // Add new intent
  Intent intent;
  intent.trigger_phrase = trigger_phrase;
  intent.callback = callback;
  intent.embedding = embedding_model_->get_embeddings(trigger_phrase);
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

bool IntentRecognizer::process_utterance(const std::string &utterance) {
  if (utterance.empty()) {
    return false;
  }

  float similarity = 0.0f;
  const Intent *best_intent = find_best_intent(utterance, similarity);

  if (best_intent != nullptr && similarity >= threshold_) {
    // Invoke the callback
    best_intent->callback(utterance, similarity);
    return true;
  }

  return false;
}

void IntentRecognizer::process_transcript(
    const struct transcript_t *transcript) {
  if (transcript == nullptr || transcript->lines == nullptr) {
    return;
  }

  for (size_t i = 0; i < transcript->line_count; ++i) {
    const transcript_line_t &line = transcript->lines[i];

    // Only process complete lines
    if (!line.is_complete) {
      continue;
    }

    // Check if we've already processed this line
    auto it = std::find(processed_line_ids_.begin(), processed_line_ids_.end(),
                        line.id);
    if (it != processed_line_ids_.end()) {
      continue;
    }

    // Process the utterance
    if (line.text != nullptr) {
      std::string utterance(line.text);
      process_utterance(utterance);
    }

    // Mark as processed
    processed_line_ids_.push_back(line.id);
  }
}

void IntentRecognizer::set_threshold(float threshold) {
  std::lock_guard<std::mutex> lock(mutex_);
  threshold_ = threshold;
}

float IntentRecognizer::get_threshold() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return threshold_;
}

size_t IntentRecognizer::get_intent_count() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return intents_.size();
}

void IntentRecognizer::clear_intents() {
  std::lock_guard<std::mutex> lock(mutex_);
  intents_.clear();
}

Transcriber *IntentRecognizer::get_transcriber() const { return transcriber_; }

void IntentRecognizer::set_transcriber(Transcriber *transcriber) {
  transcriber_ = transcriber;
}

const Intent *IntentRecognizer::find_best_intent(const std::string &utterance,
                                                 float &out_similarity) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (intents_.empty()) {
    out_similarity = 0.0f;
    return nullptr;
  }

  // Get embedding for the utterance
  std::vector<float> utterance_embedding =
      embedding_model_->get_embeddings(utterance);

  const Intent *best_intent = nullptr;
  float best_similarity = -1.0f;

  for (const auto &intent : intents_) {
    float similarity =
        embedding_model_->get_similarity(utterance_embedding, intent.embedding);

    if (similarity > best_similarity) {
      best_similarity = similarity;
      best_intent = &intent;
    }
  }

  out_similarity = best_similarity;
  return best_intent;
}
