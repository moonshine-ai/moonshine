#ifndef VOICE_ACTIVITY_DETECTOR_H
#define VOICE_ACTIVITY_DETECTOR_H

#include <string>
#include <vector>

#include "silero-vad.h"

struct VoiceActivitySegment {
  std::vector<float> audio_data;
  float start_time;
  float end_time;
  // A flag to indicate that the talking in this segment has ended.
  bool is_complete;
  // A "dirty" flag to indicate that the segment has been updated in the last
  // call to process_audio.
  bool just_updated;
  // Debug representation of the segment.
  std::string to_string() const;
};

class VoiceActivityDetector {
 private:
  const float threshold;
  const int32_t window_size;
  const int32_t hop_size;
  const size_t look_behind_sample_count;
  const size_t max_segment_sample_count;

  // Raw pointer intentionally not deleted to avoid static destruction order
  // issues
  static SileroVad *silero_vad;

  bool _is_active;
  std::vector<float> probability_window;
  int32_t probability_window_index;
  std::vector<VoiceActivitySegment> segments;
  size_t samples_processed_count;
  std::vector<float> current_segment_audio_buffer;
  std::vector<float> look_behind_audio_buffer;
  std::vector<float> processing_remainder_audio_buffer;
  bool previous_is_voice;

 public:
  VoiceActivityDetector(float threshold = 0.5f, int32_t window_size = 32,
                        int32_t hop_size = 512,
                        size_t look_behind_sample_count = 4096,
                        size_t max_segment_sample_count = 15 * 16000);
  ~VoiceActivityDetector();

  void start();
  void stop();
  bool is_active() const { return _is_active; }
  void process_audio(const float *audio_data, size_t audio_data_size,
                     int32_t sample_rate);
  const std::vector<VoiceActivitySegment> *get_segments() const {
    return &segments;
  }
  std::string to_string() const;

 private:
  void clear();
  void on_voice_start();
  void on_voice_end();
  void on_voice_continuing();
  void process_audio_chunk(const float *audio_data, size_t audio_data_size);
};

#endif