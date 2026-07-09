// stt -- on-device speech-to-text (isolated-letter/digit) classifier.
//
// Single public header for the module. It exposes:
//
//   * Classifier -- a TFLM wrapper around the int8 mel-mode SpellingCNN. Run()
//                   takes fp32 log-mel features (from the feature-generation
//                   module) and returns dequantized fp32 logits.
//
//   * Argmax / SoftmaxProb -- small stateless helpers to turn logits into a
//                   prediction index + top-1 probability for logging.
//
// The classifier is shape- and model-agnostic: the model bytes, arena, and the
// expected (n_mels, target_frames, n_classes) are all supplied by the caller,
// so the module carries no model or platform dependency beyond
// feature-generation and TFLM. A single Classifier instance must live for the
// whole run (the MicroInterpreter holds pointers into the arena handed to it).

#ifndef STT_STT_H_
#define STT_STT_H_

#include <cstddef>
#include <cstdint>

namespace tflite {
class MicroProfilerInterface;
}  // namespace tflite

namespace spelling {

// ---------------------------------------------------------------------------
// Model wrapper: Classifier.
// ---------------------------------------------------------------------------

// Quantization parameters at a model tensor boundary.
struct TensorQuant {
  float scale = 0.0f;
  int zero_point = 0;
};

class Classifier {
 public:
  // Construct from the flatbuffer + a caller-owned tensor arena (which must
  // outlive the Classifier). expected_* are sanity-checked against the model on
  // a noisy halt -- no recovery path makes sense this early in boot.
  // `profiler`, if non-null, is handed to the interpreter so per-op timings are
  // recorded on every Invoke(); it must outlive the Classifier.
  Classifier(const unsigned char* model_data, unsigned int model_size,
             uint8_t* tensor_arena, std::size_t tensor_arena_size,
             int expected_n_mels, int expected_target_frames,
             int expected_n_classes,
             tflite::MicroProfilerInterface* profiler = nullptr);

  Classifier(const Classifier&) = delete;
  Classifier& operator=(const Classifier&) = delete;

  // Run a single inference. `features` is the row-major (n_mels *
  // target_frames) fp32 log-mel buffer; `logits_out` must hold n_classes()
  // floats. We quantize into the int8 input, Invoke, dequantize the int8
  // output. `features` may safely point at feature_scratch().
  void Run(const float* features, float* logits_out) const;

  // Scratch for the fp32 log-mel features, carved from the arena's activation
  // overlay (dead until Invoke()). Contract: produce features here, then call
  // Run(feature_scratch()). The values do NOT survive Run().
  float* feature_scratch() const { return feature_scratch_; }

  TensorQuant input_quant() const { return input_quant_; }
  TensorQuant output_quant() const { return output_quant_; }
  int n_classes() const { return n_classes_; }
  std::size_t input_count() const { return input_count_; }
  std::size_t arena_used_bytes() const { return arena_used_bytes_; }

 private:
  struct Impl;
  Impl* impl_ = nullptr;

  TensorQuant input_quant_{};
  TensorQuant output_quant_{};
  int n_classes_ = 0;
  std::size_t input_count_ = 0;
  std::size_t arena_used_bytes_ = 0;
  float* feature_scratch_ = nullptr;  // borrowed from the arena overlay
};

// ---------------------------------------------------------------------------
// Prediction helpers.
// ---------------------------------------------------------------------------

// Index of the largest logit. Ties go to the lowest index.
int Argmax(const float* logits, int n_logits);

// Stable-softmax probability of `index` (subtracts max(logits) before exp()).
float SoftmaxProb(const float* logits, int n_logits, int index);

}  // namespace spelling

#endif  // STT_STT_H_
