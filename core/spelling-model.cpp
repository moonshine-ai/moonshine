#include "spelling-model.h"

#ifndef _WIN32
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "debug-utils.h"
#include "moonshine-tensor-view.h"
#include "ort-utils.h"
#include "spelling-fusion-data.h"

namespace {

// Read a single key from the model's custom_metadata_map, returning
// nullopt when the key isn't present or any ORT call fails. Caller
// frees the value through ``allocator`` when non-null.
std::optional<std::string> lookup_metadata(const OrtApi *ort_api,
                                           const OrtModelMetadata *meta,
                                           OrtAllocator *allocator,
                                           const char *key) {
  char *raw = nullptr;
  OrtStatus *status =
      ort_api->ModelMetadataLookupCustomMetadataMap(meta, allocator, key, &raw);
  if (status != nullptr) {
    ort_api->ReleaseStatus(status);
    return std::nullopt;
  }
  if (raw == nullptr) return std::nullopt;
  std::string out(raw);
  allocator->Free(allocator, raw);
  if (out.empty()) return std::nullopt;
  return out;
}

// Trim ASCII whitespace.
std::string trim(const std::string &s) {
  size_t lo = 0;
  size_t hi = s.size();
  while (lo < hi && std::isspace(static_cast<unsigned char>(s[lo]))) ++lo;
  while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1]))) --hi;
  return s.substr(lo, hi - lo);
}

// Parse a JSON string array of class labels (e.g. ``["a","b",...]``).
// We deliberately don't pull in nlohmann/json for this one tiny case so
// the model wrapper stays self-contained; the input format is
// well-defined (the trainer's own metadata) so a hand-rolled split is
// fine. Returns an empty vector when parsing fails (caller falls back
// to compiled-in defaults).
std::vector<std::string> parse_class_list_json(const std::string &raw) {
  std::vector<std::string> result;
  std::string s = trim(raw);
  if (s.empty() || s.front() != '[' || s.back() != ']') return result;
  s = s.substr(1, s.size() - 2);
  size_t i = 0;
  while (i < s.size()) {
    while (i < s.size() &&
           (std::isspace(static_cast<unsigned char>(s[i])) || s[i] == ',')) {
      ++i;
    }
    if (i >= s.size()) break;
    if (s[i] != '"') {
      // Malformed entry; bail and let the caller fall back to defaults.
      return {};
    }
    ++i;
    std::string token;
    while (i < s.size() && s[i] != '"') {
      if (s[i] == '\\' && i + 1 < s.size()) {
        token.push_back(s[i + 1]);
        i += 2;
        continue;
      }
      token.push_back(s[i]);
      ++i;
    }
    if (i >= s.size()) return {};  // unterminated string
    ++i;
    result.push_back(std::move(token));
  }
  return result;
}

}  // namespace

SpellingModel::SpellingModel(bool log_ort_run) : log_ort_run(log_ort_run) {
  ort_api_ = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SpellingModel",
                                    &ort_env_));
  LOG_ORT_ERROR(ort_api_,
                ort_api_->CreateCpuMemoryInfo(OrtDeviceAllocator,
                                              OrtMemTypeDefault,
                                              &ort_memory_info_));
  initialize_session_options();
  apply_default_metadata();
}

SpellingModel::~SpellingModel() {
  if (ort_session_ != nullptr) {
    ort_api_->ReleaseSession(ort_session_);
    ort_session_ = nullptr;
  }
  if (ort_session_options_ != nullptr) {
    ort_api_->ReleaseSessionOptions(ort_session_options_);
    ort_session_options_ = nullptr;
  }
  if (ort_memory_info_ != nullptr) {
    ort_api_->ReleaseMemoryInfo(ort_memory_info_);
    ort_memory_info_ = nullptr;
  }
  if (ort_env_ != nullptr) {
    ort_api_->ReleaseEnv(ort_env_);
    ort_env_ = nullptr;
  }
#ifndef _WIN32
  if (mmapped_data_ != nullptr) {
    munmap(const_cast<char *>(mmapped_data_), mmapped_data_size_);
    mmapped_data_ = nullptr;
    mmapped_data_size_ = 0;
  }
#endif
}

void SpellingModel::initialize_session_options() {
  LOG_ORT_ERROR(ort_api_, ort_api_->CreateSessionOptions(&ort_session_options_));
  LOG_ORT_ERROR(ort_api_, ort_api_->SetSessionGraphOptimizationLevel(
                              ort_session_options_, ORT_ENABLE_EXTENDED));
  LOG_ORT_ERROR(ort_api_, ort_api_->AddSessionConfigEntry(
                              ort_session_options_,
                              "session.load_model_format", "ORT"));
  LOG_ORT_ERROR(ort_api_, ort_api_->AddSessionConfigEntry(
                              ort_session_options_,
                              "session.use_ort_model_bytes_directly", "1"));
  LOG_ORT_ERROR(ort_api_, ort_api_->AddSessionConfigEntry(
                              ort_session_options_,
                              "session.disable_prepacking", "1"));
  LOG_ORT_ERROR(ort_api_, ort_api_->DisableCpuMemArena(ort_session_options_));
}

void SpellingModel::apply_default_metadata() {
  const auto &meta = spelling_fusion_data::default_meta();
  sample_rate_ = meta.sample_rate;
  clip_seconds_ = meta.clip_seconds;
  input_name_ = meta.input_name;
  output_name_ = meta.output_name;
  classes_ = meta.classes;
  target_samples_ =
      static_cast<size_t>(std::lround(sample_rate_ * clip_seconds_));
  if (target_samples_ == 0) target_samples_ = 1;
}

int SpellingModel::load(const char *model_path) {
  RETURN_ON_ERROR(ort_session_from_path(ort_api_, ort_env_,
                                         ort_session_options_, model_path,
                                         &ort_session_, &mmapped_data_,
                                         &mmapped_data_size_));
  RETURN_ON_NULL(ort_session_);
  return populate_metadata_from_session();
}

int SpellingModel::load_from_memory(const uint8_t *model_data,
                                     size_t model_data_size) {
  RETURN_ON_ERROR(ort_session_from_memory(ort_api_, ort_env_,
                                           ort_session_options_, model_data,
                                           model_data_size, &ort_session_));
  RETURN_ON_NULL(ort_session_);
  return populate_metadata_from_session();
}

int SpellingModel::populate_metadata_from_session() {
  // Best-effort overrides from the model's custom_metadata_map. Any
  // failure leaves the compiled-in defaults in place; the model still
  // works as long as the trainer kept the canonical waveform shape.
  OrtModelMetadata *meta = nullptr;
  OrtStatus *status = ort_api_->SessionGetModelMetadata(ort_session_, &meta);
  if (status != nullptr) {
    ort_api_->ReleaseStatus(status);
    return 0;
  }
  OrtAllocator *allocator = nullptr;
  status = ort_api_->GetAllocatorWithDefaultOptions(&allocator);
  if (status != nullptr) {
    ort_api_->ReleaseStatus(status);
    ort_api_->ReleaseModelMetadata(meta);
    return 0;
  }

  if (auto v = lookup_metadata(ort_api_, meta, allocator, "sample_rate");
      v.has_value()) {
    try {
      sample_rate_ = std::stoi(*v);
    } catch (const std::exception &) {
      // Ignore — keep default.
    }
  }
  if (auto v = lookup_metadata(ort_api_, meta, allocator, "clip_seconds");
      v.has_value()) {
    try {
      clip_seconds_ = std::stof(*v);
    } catch (const std::exception &) {
      // Ignore — keep default.
    }
  }
  if (auto v = lookup_metadata(ort_api_, meta, allocator, "input_name");
      v.has_value()) {
    input_name_ = *v;
  }
  if (auto v = lookup_metadata(ort_api_, meta, allocator, "output_name");
      v.has_value()) {
    output_name_ = *v;
  }
  if (auto v = lookup_metadata(ort_api_, meta, allocator, "classes");
      v.has_value()) {
    auto parsed = parse_class_list_json(*v);
    if (!parsed.empty()) {
      classes_ = std::move(parsed);
    }
  }

  ort_api_->ReleaseModelMetadata(meta);
  target_samples_ =
      static_cast<size_t>(std::lround(sample_rate_ * clip_seconds_));
  if (target_samples_ == 0) target_samples_ = 1;
  return 0;
}

int SpellingModel::predict(const float *audio, size_t audio_size,
                            int32_t sample_rate,
                            SpellingPrediction *out_prediction) {
  if (out_prediction == nullptr) return -1;
  if (ort_session_ == nullptr) return -1;
  if (audio == nullptr || audio_size == 0) return -1;
  if (sample_rate != sample_rate_) {
    // We deliberately don't ship a resampler in the predictor path so
    // callers don't silently get wrong predictions on a mismatched
    // rate. Resample upstream before calling predict().
    LOGF("SpellingModel::predict sample_rate mismatch: got %d, expected %d",
         sample_rate, sample_rate_);
    return -1;
  }

  std::lock_guard<std::mutex> lock(processing_mutex_);

  std::vector<float> clip(target_samples_, 0.0f);
  size_t copy_count = std::min(audio_size, target_samples_);
  std::copy(audio, audio + copy_count, clip.begin());

  const std::vector<int64_t> input_shape = {
      1, static_cast<int64_t>(target_samples_)};
  MoonshineTensorView *input_view = new MoonshineTensorView(
      input_shape, ort_get_input_type(ort_api_, ort_session_, 0), clip.data(),
      input_name_.c_str());
  OrtValue *input_value = input_view->create_ort_value(ort_api_, ort_memory_info_);

  const char *input_names[1] = {input_name_.c_str()};
  const char *output_names[1] = {output_name_.c_str()};
  OrtValue *output_value = nullptr;
  OrtStatus *run_status =
      ORT_RUN(ort_api_, ort_session_, input_names, &input_value, 1,
              output_names, 1, &output_value);
  ort_api_->ReleaseValue(input_value);
  delete input_view;
  if (run_status != nullptr) {
    const char *msg = ort_api_->GetErrorMessage(run_status);
    LOGF("SpellingModel::predict ORT error: %s", msg);
    ort_api_->ReleaseStatus(run_status);
    if (output_value != nullptr) ort_api_->ReleaseValue(output_value);
    return -1;
  }

  // Output tensor shape is (1, num_classes); softmax over the row.
  MoonshineTensorView output_view(ort_api_, output_value, "spelling_logits");
  ort_api_->ReleaseValue(output_value);
  size_t element_count = output_view.element_count();
  if (element_count == 0) return -1;
  const float *raw_logits = output_view.data<float>();

  // Guard against a model with a bigger logits row than our class list
  // (we'd mis-label). Trim to whatever we have classes for.
  size_t row_size = element_count;
  if (!classes_.empty() && row_size > classes_.size()) {
    row_size = classes_.size();
  }
  if (row_size == 0) return -1;

  float row_max = raw_logits[0];
  for (size_t i = 1; i < row_size; ++i) {
    if (raw_logits[i] > row_max) row_max = raw_logits[i];
  }
  std::vector<float> probs(row_size);
  float sum = 0.0f;
  for (size_t i = 0; i < row_size; ++i) {
    probs[i] = std::exp(raw_logits[i] - row_max);
    sum += probs[i];
  }
  if (sum == 0.0f || !std::isfinite(sum)) return -1;
  size_t best_idx = 0;
  float best_prob = probs[0] / sum;
  for (size_t i = 1; i < row_size; ++i) {
    float p = probs[i] / sum;
    if (p > best_prob) {
      best_prob = p;
      best_idx = i;
    }
  }

  std::string raw_class = (best_idx < classes_.size())
                              ? classes_[best_idx]
                              : std::to_string(best_idx);
  std::string canonical = raw_class;
  const auto &class_to_char = spelling_fusion_data::spell_class_to_char();
  auto it = class_to_char.find(raw_class);
  if (it != class_to_char.end()) {
    canonical = it->second;
  }

  out_prediction->raw_class = std::move(raw_class);
  out_prediction->character = std::move(canonical);
  out_prediction->probability = best_prob;
  return 0;
}
