// vad -- on-device voice activity detection.
//
// Single public header for the module. It exposes the two halves of the VAD:
//
//   * Vad            -- a TFLM wrapper around the int8 TinyVadCNN.
//                       Given a normalised (n_mels x window_frames) log-mel
//                       window it returns ONE speech probability in [0, 1].
//                       One call per ~32 ms streaming hop.
//
//   * VadSegmenter   -- a heap-free moving-average smoother + 1 s clip
//                       extractor. It turns the per-frame probabilities into speech-segment
//                       boundaries (absolute sample indices on the input
//                       stream) without storing any audio.
//
// The streaming log-mel front-end that feeds Vad lives in the
// feature-generation module (MelStreamer); compose them in your application
// (see examples/rp2350's audio path). The VAD model is shape-agnostic: the model
// bytes, arena, and (n_mels, window_frames) are all supplied by the caller, so
// the module carries no model or platform dependency beyond feature-generation
// and TFLM.

#ifndef VAD_VAD_H_
#define VAD_VAD_H_

#include <cstddef>
#include <cstdint>

namespace tflite {
class MicroProfilerInterface;
}  // namespace tflite

namespace spelling {

// ---------------------------------------------------------------------------
// Model wrapper: Vad.
// ---------------------------------------------------------------------------

struct VadTensorQuant {
  float scale = 0.0f;
  int zero_point = 0;
};

class Vad {
 public:
  // Construct from the flatbuffer + a caller-owned tensor arena (which must
  // outlive the Vad). expected_* are sanity-checked against the model on a
  // noisy halt. Because the VAD never Invoke()s at the same time as the STT
  // classifier, it can share the STT tensor arena.
  Vad(const unsigned char* model_data, unsigned int model_size,
      uint8_t* tensor_arena, std::size_t tensor_arena_size, int expected_n_mels,
      int expected_window_frames,
      tflite::MicroProfilerInterface* profiler = nullptr);

  Vad(const Vad&) = delete;
  Vad& operator=(const Vad&) = delete;

  // Run one inference on a (n_mels * window_frames) row-major fp32 log-mel
  // window; returns the speech probability in [0, 1] (sigmoid of the
  // dequantized logit). `features` may point at feature_scratch().
  float Predict(const float* features) const;

  // Scratch for the fp32 log-mel window, borrowed from the arena overlay (dead
  // until Invoke()). Contract: produce features here, then Predict(it).
  float* feature_scratch() const { return feature_scratch_; }

  VadTensorQuant input_quant() const { return input_quant_; }
  VadTensorQuant output_quant() const { return output_quant_; }
  std::size_t input_count() const { return input_count_; }
  std::size_t arena_used_bytes() const { return arena_used_bytes_; }

 private:
  struct Impl;
  Impl* impl_ = nullptr;
  VadTensorQuant input_quant_{};
  VadTensorQuant output_quant_{};
  std::size_t input_count_ = 0;
  std::size_t arena_used_bytes_ = 0;
  float* feature_scratch_ = nullptr;
};

// ---------------------------------------------------------------------------
// Smoothing + segmentation: VadSegmenter.
// ---------------------------------------------------------------------------

enum class VadEvent {
  kNone,
  kSpeechStart,
  kSpeechContinuing,
  kSpeechEnd,
};

// Upper bound on the moving-average window so prob_window_ is a fixed array (no
// heap). 64 frames @ 32 ms = ~2 s, far beyond the ~0.5 s default.
constexpr int kVadSmoothWindowMax = 64;

class VadSegmenter {
 public:
  VadSegmenter(float threshold, int window_frames, int hop,
               std::size_t look_behind_samples,
               std::size_t max_segment_samples);

  // Reset all state to begin a fresh stream.
  void Start();

  // Advance one frame (== hop samples) given the model's RAW (un-smoothed)
  // speech probability. Returns the transition that occurred, if any.
  VadEvent ProcessFrame(float raw_probability);

  // Flush a trailing open segment at end-of-stream.
  VadEvent Finish();

  // Boundaries (absolute sample indices) of the most-recent segment.
  std::size_t segment_start_sample() const { return segment_start_; }
  std::size_t segment_end_sample() const { return segment_end_; }
  std::size_t samples_processed() const { return samples_processed_; }

 private:
  const float threshold_;
  const int window_frames_;
  const int hop_;
  const std::size_t look_behind_samples_;
  const std::size_t max_segment_samples_;

  float prob_window_[kVadSmoothWindowMax];
  int prob_index_;
  std::size_t samples_processed_;
  std::size_t current_segment_samples_;
  std::size_t segment_start_;
  std::size_t segment_end_;
  bool previous_is_voice_;
};

// Front-aligned pad/truncate of src[start, end) into out[0, clip_len): take
// from the segment start, zero-pad the tail. `out` must hold clip_len floats.
void ExtractClipFrontAligned(const float* src, std::size_t src_len,
                             std::size_t start, std::size_t end, float* out,
                             std::size_t clip_len);

}  // namespace spelling

#endif  // VAD_VAD_H_
