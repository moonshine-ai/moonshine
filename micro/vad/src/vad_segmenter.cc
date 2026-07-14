#include "tensorflow/lite/micro/micro_log.h"
#include "vad/vad.h"

namespace spelling {

VadSegmenter::VadSegmenter(float threshold, int window_frames, int hop,
                           std::size_t look_behind_samples,
                           std::size_t max_segment_samples)
    : threshold_(threshold),
      window_frames_(window_frames > kVadSmoothWindowMax ? kVadSmoothWindowMax
                                                         : window_frames),
      hop_(hop),
      look_behind_samples_(look_behind_samples),
      max_segment_samples_(max_segment_samples) {
  if (window_frames > kVadSmoothWindowMax) {
    MicroPrintf("VadSegmenter: window_frames %d > max %d; clamped",
                window_frames, kVadSmoothWindowMax);
  }
  Start();
}

void VadSegmenter::Start() {
  for (int i = 0; i < window_frames_; ++i) prob_window_[i] = 0.0f;
  prob_index_ = 0;
  samples_processed_ = 0;
  current_segment_samples_ = 0;
  segment_start_ = 0;
  segment_end_ = 0;
  previous_is_voice_ = false;
}

VadEvent VadSegmenter::ProcessFrame(float raw_probability) {
  samples_processed_ += static_cast<std::size_t>(hop_);

  float smoothed;
  if (threshold_ > 0.0f) {
    prob_window_[prob_index_] = raw_probability;
    prob_index_ = (prob_index_ + 1) % window_frames_;
    float sum = 0.0f;
    for (int i = 0; i < window_frames_; ++i) sum += prob_window_[i];
    smoothed = sum / static_cast<float>(window_frames_);
  } else {
    smoothed = 1.0f;  // threshold 0 -> always voice (still split on max).
  }

  // Max-segment linear fade: force a break in long continuous speech.
  const std::size_t fade_samples = (max_segment_samples_ * 2) / 3;
  if (max_segment_samples_ && current_segment_samples_ > fade_samples) {
    const float fade_factor =
        static_cast<float>(current_segment_samples_ - fade_samples) /
        static_cast<float>(fade_samples);
    smoothed = smoothed * fade_factor;
  }

  const bool current_is_voice = smoothed > threshold_;
  VadEvent event = VadEvent::kNone;

  if (current_is_voice && !previous_is_voice_) {
    // The look-behind ring already holds the current hop and the clamp uses
    // the post-increment sample count. Segment spans [proc - look_behind,
    // proc]; current length == look_behind.
    const std::size_t look_behind = look_behind_samples_ < samples_processed_
                                        ? look_behind_samples_
                                        : samples_processed_;
    segment_start_ = samples_processed_ - look_behind;
    segment_end_ = samples_processed_;
    current_segment_samples_ = look_behind;
    event = VadEvent::kSpeechStart;
  } else if (!current_is_voice && previous_is_voice_) {
    current_segment_samples_ += static_cast<std::size_t>(hop_);
    segment_end_ = samples_processed_;
    current_segment_samples_ = 0;
    event = VadEvent::kSpeechEnd;
  } else if (current_is_voice && previous_is_voice_) {
    current_segment_samples_ += static_cast<std::size_t>(hop_);
    segment_end_ = samples_processed_;
    event = VadEvent::kSpeechContinuing;
  }

  previous_is_voice_ = current_is_voice;
  return event;
}

VadEvent VadSegmenter::Finish() {
  if (previous_is_voice_) {
    segment_end_ = samples_processed_;
    previous_is_voice_ = false;
    return VadEvent::kSpeechEnd;
  }
  return VadEvent::kNone;
}

void ExtractClipFrontAligned(const float* src, std::size_t src_len,
                             std::size_t start, std::size_t end, float* out,
                             std::size_t clip_len) {
  std::size_t s = start;
  std::size_t e = end;
  if (e > src_len) e = src_len;
  if (s > e) s = e;
  std::size_t n = e - s;
  if (n > clip_len) n = clip_len;
  for (std::size_t i = 0; i < n; ++i) out[i] = src[s + i];
  for (std::size_t i = n; i < clip_len; ++i) out[i] = 0.0f;
}

std::size_t EnergyCentroidIndex(const int16_t* buf, std::size_t start,
                                std::size_t end) {
  if (end <= start) return start;
  // int64 accumulators: for a 1 s / 16 kHz int16 window the sums stay well
  // within range (den <= ~1.6e13, num <= ~2.7e17 << 9.2e18), so we avoid any
  // float in the hot-ish path and get a deterministic result.
  int64_t num = 0;
  int64_t den = 0;
  for (std::size_t i = start; i < end; ++i) {
    const int64_t e = static_cast<int64_t>(buf[i]) * static_cast<int64_t>(buf[i]);
    num += static_cast<int64_t>(i) * e;
    den += e;
  }
  if (den <= 0) return (start + end) / 2;  // silent region -> midpoint
  // Round to nearest sample.
  return static_cast<std::size_t>((num + den / 2) / den);
}

}  // namespace spelling
