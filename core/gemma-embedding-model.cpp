#include "gemma-embedding-model.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "bin-tokenizer.h"
#include "string-utils.h"

#define DEBUG_ALLOC_ENABLED 1
#include "debug-utils.h"
#include "ort-utils.h"

GemmaEmbeddingModel::GemmaEmbeddingModel()
    : ort_api_(nullptr),
      ort_env_(nullptr),
      ort_session_options_(nullptr),
      ort_memory_info_(nullptr),
      ort_allocator_(nullptr),
      session_(nullptr),
      mmapped_data_(nullptr),
      mmapped_data_size_(0),
      tokenizer_(nullptr) {
  ort_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);

  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING,
                                    "GemmaEmbeddingModel", &ort_env_));

  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateCpuMemoryInfo(
                    OrtDeviceAllocator, OrtMemTypeDefault, &ort_memory_info_));

  ort_allocator_ = new MoonshineOrtAllocator(ort_memory_info_);

  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateSessionOptions(&ort_session_options_));
  LOG_ORT_ERROR(ort_api_, ort_api_->SetSessionGraphOptimizationLevel(
                              ort_session_options_, ORT_ENABLE_ALL));
  LOG_ORT_ERROR(ort_api_,
                ort_api_->SetIntraOpNumThreads(ort_session_options_, 1));
}

GemmaEmbeddingModel::~GemmaEmbeddingModel() {
  if (session_) ort_api_->ReleaseSession(session_);
  if (ort_session_options_)
    ort_api_->ReleaseSessionOptions(ort_session_options_);
  if (ort_memory_info_) ort_api_->ReleaseMemoryInfo(ort_memory_info_);
  if (ort_env_) ort_api_->ReleaseEnv(ort_env_);
  delete ort_allocator_;
  delete tokenizer_;

#ifndef _WIN32
  if (mmapped_data_) {
    munmap(const_cast<char *>(mmapped_data_), mmapped_data_size_);
  }
#endif
}

int GemmaEmbeddingModel::load(const char *model_dir,
                              const char *model_variant) {
  if (model_dir == nullptr) {
    LOG("Model directory is null\n");
    return 1;
  }

  // Build model path based on variant
  std::string variant = model_variant ? model_variant : "q4";
  std::string model_filename;

  if (variant == "fp32") {
    model_filename = "model.onnx";
  } else if (variant == "fp16") {
    model_filename = "model_fp16.onnx";
  } else if (variant == "q8" || variant == "quantized") {
    model_filename = "model_quantized.onnx";
  } else if (variant == "q4") {
    model_filename = "model_q4.onnx";
  } else if (variant == "q4f16") {
    model_filename = "model_q4f16.onnx";
  } else {
    LOGF("Unknown model variant: %s\n", variant.c_str());
    return 1;
  }

  std::string model_path =
      append_path_component(model_dir, model_filename.c_str());

  std::string tokenizer_path =
      append_path_component(model_dir, "tokenizer.bin");

  // Load ONNX model
  RETURN_ON_ERROR(ort_session_from_path(
      ort_api_, ort_env_, ort_session_options_, model_path.c_str(), &session_,
      &mmapped_data_, &mmapped_data_size_));
  RETURN_ON_NULL(session_);

  // Load tokenizer
  RETURN_ON_ERROR(load_tokenizer(tokenizer_path.c_str()));

  return 0;
}

int GemmaEmbeddingModel::load_from_memory(const uint8_t *model_data,
                                          size_t model_data_size,
                                          const uint8_t *tokenizer_data,
                                          size_t tokenizer_data_size) {
  if (model_data == nullptr || model_data_size == 0) {
    LOG("Model data is null or empty\n");
    return 1;
  }

  RETURN_ON_ERROR(ort_session_from_memory(ort_api_, ort_env_,
                                          ort_session_options_, model_data,
                                          model_data_size, &session_));
  RETURN_ON_NULL(session_);

  RETURN_ON_ERROR(
      load_tokenizer_from_memory(tokenizer_data, tokenizer_data_size));

  return 0;
}

int GemmaEmbeddingModel::load_tokenizer(const char *tokenizer_path) {
  try {
    // Gemma uses ▁ (U+2581) as the space character in SentencePiece
    tokenizer_ = new BinTokenizer(tokenizer_path, "▁");
    return 0;
  } catch (const std::exception &e) {
    LOGF("Failed to load tokenizer: %s\n", e.what());
    return 1;
  }
}

int GemmaEmbeddingModel::load_tokenizer_from_memory(const uint8_t *data,
                                                    size_t data_size) {
  if (data == nullptr || data_size == 0) {
    LOG("Tokenizer data is null or empty\n");
    return 1;
  }

  try {
    // Gemma uses ▁ (U+2581) as the space character in SentencePiece
    tokenizer_ = new BinTokenizer(data, data_size, "▁");
    LOGF("Tokenizer loaded with %zu tokens\n",
         tokenizer_->tokens_to_bytes.size());
    return 0;
  } catch (const std::exception &e) {
    LOGF("Failed to load tokenizer from memory: %s\n", e.what());
    return 1;
  }
}

std::vector<int64_t> GemmaEmbeddingModel::tokenize(const std::string &text) {
  if (!tokenizer_) {
    LOG("Tokenizer not loaded\n");
    return {};
  }

  // Use BinTokenizer to convert text to tokens
  std::vector<int64_t> tokens = tokenizer_->text_to_tokens<int64_t>(text);

  // Prepend BOS and append EOS
  std::vector<int64_t> result;
  result.reserve(tokens.size() + 2);
  result.push_back(config_.bos_token_id);  // <bos>
  result.insert(result.end(), tokens.begin(), tokens.end());
  result.push_back(config_.eos_token_id);  // <eos>

  // Truncate to max sequence length if needed
  if (result.size() > static_cast<size_t>(config_.max_seq_length)) {
    result.resize(config_.max_seq_length);
    result.back() = config_.eos_token_id;  // Ensure EOS at end
  }

  return result;
}

std::vector<float> GemmaEmbeddingModel::run_inference(
    const std::vector<int64_t> &input_ids,
    const std::vector<int64_t> &attention_mask) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!session_) {
    LOG("Model not loaded\n");
    return {};
  }

  int64_t batch_size = 1;
  int64_t seq_length = static_cast<int64_t>(input_ids.size());

  // Create input tensors
  std::vector<int64_t> input_shape = {batch_size, seq_length};

  OrtValue *input_ids_tensor = nullptr;
  OrtStatus *status = ort_api_->CreateTensorWithDataAsOrtValue(
      ort_memory_info_, const_cast<int64_t *>(input_ids.data()),
      input_ids.size() * sizeof(int64_t), input_shape.data(),
      input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      &input_ids_tensor);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    return {};
  }

  OrtValue *attention_mask_tensor = nullptr;
  status = ort_api_->CreateTensorWithDataAsOrtValue(
      ort_memory_info_, const_cast<int64_t *>(attention_mask.data()),
      attention_mask.size() * sizeof(int64_t), input_shape.data(),
      input_shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      &attention_mask_tensor);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseValue(input_ids_tensor);
    return {};
  }

  // Input/output names
  const char *input_names[] = {"input_ids", "attention_mask"};
  const char *output_names[] = {"sentence_embedding"};

  OrtValue *inputs[] = {input_ids_tensor, attention_mask_tensor};
  OrtValue *outputs[] = {nullptr};

  // Run inference
  status = ort_api_->Run(session_, nullptr, input_names, inputs, 2,
                         output_names, 1, outputs);

  // Release input tensors
  ort_api_->ReleaseValue(input_ids_tensor);
  ort_api_->ReleaseValue(attention_mask_tensor);

  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    return {};
  }

  // Extract output embedding
  OrtTensorTypeAndShapeInfo *output_info = nullptr;
  status = ort_api_->GetTensorTypeAndShape(outputs[0], &output_info);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }

  size_t num_dims = 0;
  status = ort_api_->GetDimensionsCount(output_info, &num_dims);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseTensorTypeAndShapeInfo(output_info);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }

  std::vector<int64_t> output_shape(num_dims);
  status = ort_api_->GetDimensions(output_info, output_shape.data(), num_dims);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseTensorTypeAndShapeInfo(output_info);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }
  ort_api_->ReleaseTensorTypeAndShapeInfo(output_info);

  // Calculate output size
  size_t output_size = 1;
  for (int64_t dim : output_shape) {
    output_size *= static_cast<size_t>(dim);
  }

  // Copy output data
  float *output_data = nullptr;
  status = ort_api_->GetTensorMutableData(outputs[0], (void **)&output_data);
  if (status != nullptr) {
    LOG_ORT_ERROR(ort_api_, status);
    ort_api_->ReleaseValue(outputs[0]);
    return {};
  }

  std::vector<float> embedding(output_data, output_data + output_size);

  ort_api_->ReleaseValue(outputs[0]);

  // Normalize the embedding
  normalize_embedding(embedding);

  return embedding;
}

std::vector<float> GemmaEmbeddingModel::get_embeddings(
    const std::string &text) {
  if (!is_loaded()) {
    LOG("Model not loaded\n");
    return {};
  }

  // Tokenize the input
  std::vector<int64_t> input_ids = tokenize(text);

  // Create attention mask (all 1s for actual tokens)
  std::vector<int64_t> attention_mask(input_ids.size(), 1);

  // Run inference
  return run_inference(input_ids, attention_mask);
}

std::vector<float> GemmaEmbeddingModel::get_embeddings_with_prefix(
    const std::string &text, const std::string &prefix) {
  return get_embeddings(prefix + text);
}

std::vector<float> GemmaEmbeddingModel::get_query_embeddings(
    const std::string &query) {
  return get_embeddings_with_prefix(query, QUERY_PREFIX);
}

std::vector<float> GemmaEmbeddingModel::get_document_embeddings(
    const std::string &document) {
  return get_embeddings_with_prefix(document, DOCUMENT_PREFIX);
}

std::vector<float> GemmaEmbeddingModel::truncate_embedding(
    const std::vector<float> &embedding, int target_dim) {
  if (target_dim <= 0 || static_cast<size_t>(target_dim) >= embedding.size()) {
    return embedding;
  }

  // Truncate to target dimension (MRL - Matryoshka Representation Learning)
  std::vector<float> truncated(embedding.begin(),
                               embedding.begin() + target_dim);

  // Renormalize
  normalize_embedding(truncated);

  return truncated;
}

void GemmaEmbeddingModel::normalize_embedding(std::vector<float> &embedding) {
  if (embedding.empty()) return;

  float norm = 0.0f;
  for (float v : embedding) {
    norm += v * v;
  }
  norm = std::sqrt(norm);

  if (norm > 0.0f) {
    for (float &v : embedding) {
      v /= norm;
    }
  }
}

bool GemmaEmbeddingModel::is_loaded() const { return session_ != nullptr; }

const GemmaEmbeddingConfig &GemmaEmbeddingModel::get_config() const {
  return config_;
}
