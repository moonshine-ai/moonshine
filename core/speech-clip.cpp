#include "speech-clip.h"

#include <algorithm>
#include <cmath>

#include "resampler.h"
#include "voice-activity-detector.h"

namespace {

constexpr int32_t kClipSampleRate = 16000;
// Granularity of the window search, in seconds. Matches the step the web
// client used before this moved into the core.
constexpr float kWindowStepSeconds = 0.1f;

}  // namespace

SpeechClip extract_speech_clip(const float *audio_data, size_t audio_data_size,
                               int32_t sample_rate,
                               const SpeechClipOptions &options) {
  SpeechClip result;
  if (audio_data == nullptr || audio_data_size == 0 || sample_rate <= 0 ||
      options.clip_duration_seconds <= 0.0f) {
    return result;
  }

  const std::vector<float> input(audio_data, audio_data + audio_data_size);
  const std::vector<float> audio =
      resample_audio(input, static_cast<float>(sample_rate), kClipSampleRate);
  const size_t clip_sample_count = static_cast<size_t>(
      std::lround(options.clip_duration_seconds * kClipSampleRate));
  if (audio.size() < clip_sample_count) {
    // Not enough recording yet to fill a single window.
    return result;
  }

  VoiceActivityDetector detector(options.vad_threshold);
  detector.start();
  detector.process_audio(audio.data(), audio.size(), kClipSampleRate);
  detector.stop();

  std::vector<std::pair<float, float>> segments;
  for (const VoiceActivitySegment &segment : *detector.get_segments()) {
    if (segment.end_time > segment.start_time) {
      segments.emplace_back(segment.start_time, segment.end_time);
    }
  }
  if (segments.empty()) {
    return result;
  }

  const float total_seconds =
      static_cast<float>(audio.size()) / kClipSampleRate;
  const float last_start = total_seconds - options.clip_duration_seconds;

  float best_start = 0.0f;
  float best_coverage = 0.0f;
  // Count the windows up front and derive each start by multiplication rather
  // than walking a float counter, which would drift over a long recording.
  const int64_t window_count =
      last_start < 0.0f
          ? 0
          : static_cast<int64_t>((last_start + 1e-6f) / kWindowStepSeconds) + 1;
  for (int64_t window = 0; window < window_count; ++window) {
    const float start = static_cast<float>(window) * kWindowStepSeconds;
    const float end = start + options.clip_duration_seconds;
    float coverage = 0.0f;
    for (const auto &[segment_start, segment_end] : segments) {
      coverage += std::max(
          0.0f, std::min(segment_end, end) - std::max(segment_start, start));
    }
    if (coverage > best_coverage) {
      best_coverage = coverage;
      best_start = start;
    }
  }

  result.start_time_seconds = best_start;
  result.speech_seconds = best_coverage;
  if (best_coverage < options.minimum_speech_seconds) {
    // A window exists but it is mostly silence, so the caller should keep
    // recording. Reporting the coverage lets them show progress.
    return result;
  }

  size_t from = static_cast<size_t>(std::lround(best_start * kClipSampleRate));
  from = std::min(from, audio.size() - clip_sample_count);
  size_t to = from + clip_sample_count;
  if (options.tail_pad_seconds > 0.0f) {
    const size_t pad_samples = static_cast<size_t>(
        std::lround(options.tail_pad_seconds * kClipSampleRate));
    to = std::min(from + clip_sample_count + pad_samples, audio.size());
  }
  result.audio.assign(audio.begin() + static_cast<ptrdiff_t>(from),
                      audio.begin() + static_cast<ptrdiff_t>(to));
  result.start_time_seconds = static_cast<float>(from) / kClipSampleRate;
  result.is_complete = true;
  return result;
}
