#include "voice-activity-detector.h"

#include <cassert>

#include <numeric>

#include "debug-utils.h"
#include "resampler.h"
#include "ten_vad.h"

#include <mutex>

namespace {
constexpr int32_t vad_sample_rate = 16000;

// Only allow one instance of the VAD model to be created. This is to avoid
// memory bloat and latency issues, but it also means that only one thread at a
// time can use the VAD model. This is enforced by the mutex.
std::mutex ten_vad_mutex;
ten_vad_handle_t ten_vad_handle_singleton = nullptr;

int32_t get_or_create_ten_vad_handle(ten_vad_handle_t **out_handle,
                                     int32_t hop_size, float threshold) {
  std::lock_guard<std::mutex> lock(ten_vad_mutex);
  if (ten_vad_handle_singleton == nullptr) {
    int status = ten_vad_create(&ten_vad_handle_singleton, hop_size, threshold);
    if (status < 0) {
      throw std::runtime_error("Failed to create VAD model");
    }
  }
  *out_handle = &ten_vad_handle_singleton;
  return 0;
}
float seconds_from_sample_count(size_t sample_count) {
  return static_cast<float>(sample_count) / vad_sample_rate;
}
} // namespace

VoiceActivityDetector::VoiceActivityDetector(float threshold, int32_t hop_size,
                                             int32_t window_size,
                                             size_t look_behind_sample_count,
                                             size_t max_segment_sample_count)
    : threshold(threshold), hop_size(hop_size), window_size(window_size),
      look_behind_sample_count(look_behind_sample_count),
      max_segment_sample_count(max_segment_sample_count), handle(nullptr) {
  int status = get_or_create_ten_vad_handle(&handle, hop_size, threshold);
  if (status < 0) {
    throw std::runtime_error("Failed to create VAD model");
  }
  probability_window.resize(window_size);
  probability_window_index = 0;
  current_segment_audio_buffer.resize(0);
  look_behind_audio_buffer.resize(look_behind_sample_count);
  processing_remainder_audio_buffer.resize(0);
  previous_is_voice = false;
  _is_active = false;
}

VoiceActivityDetector::~VoiceActivityDetector() {}

void VoiceActivityDetector::start() {
  _is_active = true;
  samples_processed_count = 0;
  segments.clear();
  current_segment_audio_buffer.resize(0);
  look_behind_audio_buffer.resize(look_behind_sample_count, 0.0f);
  processing_remainder_audio_buffer.resize(0);
  probability_window.resize(window_size, 0.0f);
  probability_window_index = 0;
  previous_is_voice = false;
}

void VoiceActivityDetector::stop() {
  _is_active = false;
  if (previous_is_voice) {
    on_voice_end();
  }
}

void VoiceActivityDetector::process_audio(const float *audio_data,
                                          size_t audio_data_size,
                                          int32_t sample_rate) {
  if (!_is_active) {
    return;
  }
  if (handle == nullptr) {
    throw std::runtime_error("Ten VAD model not loaded");
  }
  for (VoiceActivitySegment &segment : segments) {
    segment.just_updated = false;
  }
  std::vector<float> input_audio_vector(audio_data,
                                        audio_data + audio_data_size);
  // The detection model expects 16000 Hz audio.
  const std::vector<float> resampled_audio_vector =
      resample_audio(input_audio_vector, sample_rate, vad_sample_rate);

  std::vector<float> processing_buffer = processing_remainder_audio_buffer;
  processing_buffer.insert(processing_buffer.end(),
                           resampled_audio_vector.begin(),
                           resampled_audio_vector.end());
  while (processing_buffer.size() >= (size_t)(hop_size)) {
    const float *audio_data = processing_buffer.data();
    size_t audio_data_size = hop_size;
    process_audio_chunk(audio_data, audio_data_size);
    processing_buffer.erase(processing_buffer.begin(),
                            processing_buffer.begin() + hop_size);
  }
  processing_remainder_audio_buffer = processing_buffer;
}

void VoiceActivityDetector::process_audio_chunk(const float *audio_data,
                                                size_t audio_data_size) {
  assert(audio_data_size == (size_t)(hop_size));
  samples_processed_count += audio_data_size;
  // Remove the oldest samples from look_behind_buffer and add the new samples
  // at the end.
  std::move(look_behind_audio_buffer.begin() + audio_data_size,
            look_behind_audio_buffer.end(), look_behind_audio_buffer.begin());
  std::copy(audio_data, audio_data + audio_data_size,
            look_behind_audio_buffer.end() - audio_data_size);

  std::vector<int16_t> audio_int16(audio_data_size);
  for (size_t i = 0; i < audio_data_size; i++) {
    audio_int16[i] = static_cast<int16_t>(audio_data[i] * 32767);
  }
  float smoothed_probability = 0.0f;
  if (threshold > 0.0f) {
    float current_probability = 0.0f;
    {
      std::lock_guard<std::mutex> lock(ten_vad_mutex);
      int current_flag;
      int32_t status =
          ten_vad_process(*handle, audio_int16.data(), (int32_t)audio_data_size,
                          &current_probability, &current_flag);
      if (status < 0) {
        throw std::runtime_error("Ten VAD failed to process audio, error: " +
                                 std::to_string(status));
      }
    }
    probability_window[probability_window_index] = current_probability;
    probability_window_index =
        (probability_window_index + 1) % probability_window.size();
    smoothed_probability = std::accumulate(probability_window.begin(),
                                           probability_window.end(), 0.0f) /
                           probability_window.size();
  } else {
    // If the threshold is 0.0f, assume the audio is always voice. The VAD will
    // still break audio into segments below if the input is longer than
    // max_segment_sample_count
    smoothed_probability = 1.0f;
  }
  // If the voice audio buffer is too long, set the smoothed probability to 0 to
  // force a voice end event.
  if (max_segment_sample_count && (current_segment_audio_buffer.size() >
                                   (size_t)(max_segment_sample_count))) {
    smoothed_probability = 0;
  }
  bool current_is_voice = smoothed_probability > threshold;
  if (current_is_voice && !previous_is_voice) {
    // Make sure we don't "look back" to before the start of the stream.
    const size_t look_behind_size =
        std::min(look_behind_sample_count, samples_processed_count);
    current_segment_audio_buffer = std::vector<float>(
        look_behind_audio_buffer.begin() +
            (look_behind_audio_buffer.size() - look_behind_size),
        look_behind_audio_buffer.end());
    on_voice_start();
  } else if (!current_is_voice && previous_is_voice) {
    current_segment_audio_buffer.insert(current_segment_audio_buffer.end(),
                                        audio_data,
                                        audio_data + audio_data_size);
    on_voice_end();
    current_segment_audio_buffer.resize(0);
    look_behind_audio_buffer.resize(look_behind_sample_count, 0.0f);
  } else if (current_is_voice && previous_is_voice) {
    current_segment_audio_buffer.insert(current_segment_audio_buffer.end(),
                                        audio_data,
                                        audio_data + audio_data_size);
    on_voice_continuing();
  }
  previous_is_voice = current_is_voice;
}

void VoiceActivityDetector::on_voice_start() {
  segments.push_back(VoiceActivitySegment());
  VoiceActivitySegment *segment = &(segments.back());
  const float current_time = seconds_from_sample_count(samples_processed_count);
  const float segment_start_time =
      current_time -
      seconds_from_sample_count(current_segment_audio_buffer.size());
  segment->audio_data = current_segment_audio_buffer;
  segment->start_time = segment_start_time;
  segment->end_time = current_time;
  segment->is_complete = false;
  segment->just_updated = true;
}

void VoiceActivityDetector::on_voice_continuing() {
  VoiceActivitySegment *segment = &(segments.back());
  const float current_time = seconds_from_sample_count(samples_processed_count);
  segment->audio_data = current_segment_audio_buffer;
  segment->end_time = current_time;
  segment->is_complete = false;
  segment->just_updated = true;
}

void VoiceActivityDetector::on_voice_end() {
  VoiceActivitySegment *segment = &(segments.back());
  const float current_time = seconds_from_sample_count(samples_processed_count);
  segment->audio_data = current_segment_audio_buffer;
  segment->end_time = current_time;
  segment->is_complete = true;
  segment->just_updated = true;
}

std::string VoiceActivitySegment::to_string() const {
  std::string result =
      "VoiceActivitySegment(start_time=" + std::to_string(start_time);
  result += ", end_time=" + std::to_string(end_time);
  result += ", is_complete=" + std::to_string(is_complete);
  result += ")";
  result += ", audio_data=" + float_vector_stats_to_string(audio_data);
  return result;
}

std::string VoiceActivityDetector::to_string() const {
  std::string result = "VoiceActivityDetector(segments=[";
  for (const VoiceActivitySegment &segment : segments) {
    result += segment.to_string() + ", ";
  }
  result += "])";
  return result;
}
