#ifndef TRANSCRIBER_H
#define TRANSCRIBER_H

#include <cinttypes>

#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "moonshine-model.h"
#include "moonshine-streaming-model.h"
#include "voice-activity-detector.h"

struct TranscriberLine {
  std::string *text = nullptr;
  std::vector<float> audio_data;
  float start_time;
  float duration;
  bool is_complete;
  bool just_updated;
  bool is_new;
  bool has_text_changed;
  uint64_t id;
  uint32_t last_transcription_latency_ms;

  TranscriberLine();
  TranscriberLine(const TranscriberLine &other);
  TranscriberLine &operator=(const TranscriberLine &other);
  ~TranscriberLine();
  std::string to_string() const;
};

struct TranscriptStreamOutput {
  std::map<uint64_t, TranscriberLine> internal_lines_map;
  std::vector<uint64_t> ordered_internal_line_ids;
  std::vector<transcript_line_t> output_lines;

  struct transcript_t transcript = {.lines = nullptr, .line_count = 0};
  void clear_update_flags();
  void mark_all_lines_as_complete();
  void add_or_update_line(TranscriberLine &line);
  void update_transcript_from_lines();
};

class TranscriberStream {
public:
  VoiceActivityDetector *vad = nullptr;
  std::mutex vad_mutex;
  TranscriptStreamOutput *transcript_output;
  std::vector<float> new_audio_buffer;
  std::string save_input_wav_path = "";
  std::vector<float> save_input_data;
  int32_t last_save_sample_rate = 0;
  int32_t stream_id = -1;

  TranscriberStream(VoiceActivityDetector *vad, int32_t stream_id,
                    const std::string &save_input_wav_path = "");
  ~TranscriberStream() {
    delete this->vad;
    delete this->transcript_output;
  }
  void add_to_new_audio_buffer(const float *audio_data, uint64_t audio_length,
                               int32_t sample_rate);
  void clear_new_audio_buffer();

  void start();
  void stop();

  void save_audio_data_to_wav(const float *audio_data, uint64_t audio_length,
                              int32_t sample_rate);
  std::string get_wav_filename();
};

typedef std::map<int32_t, TranscriberStream *> TranscriberStreamMap;

struct TranscriberOptions {
  enum ModelSource {
    FILES,
    MEMORY,
    NONE,
  };
  ModelSource model_source = ModelSource::FILES;
  const char *model_path = nullptr;
  uint32_t model_arch = -1;
  const uint8_t *encoder_model_data = nullptr;
  size_t encoder_model_data_size = 0;
  const uint8_t *decoder_model_data = nullptr;
  size_t decoder_model_data_size = 0;
  const uint8_t *tokenizer_data = nullptr;
  size_t tokenizer_data_size = 0;
  float transcription_interval = 0.5f;
  float vad_threshold = 0.5f;
  std::string save_input_wav_path = "";
};

class Transcriber {
private:
  TranscriberOptions options;

  // Non-streaming model (used for TINY and BASE architectures)
  MoonshineModel *stt_model;
  std::mutex stt_model_mutex;

  // Streaming model (used for TINY_STREAMING and BASE_STREAMING architectures)
  MoonshineStreamingModel *streaming_model;
  std::mutex streaming_model_mutex;

  TranscriberStreamMap streams;
  int32_t next_stream_id;
  std::mutex streams_mutex;
  std::atomic<uint64_t> next_line_id = 0;

  TranscriberStream *batch_stream = nullptr;
  std::mutex batch_stream_mutex;

public:
  Transcriber(const TranscriberOptions &options = TranscriberOptions());
  ~Transcriber();

  void transcribe_without_streaming(const float *audio_data,
                                    uint64_t audio_length, int32_t sample_rate,
                                    uint32_t flags,
                                    struct transcript_t **out_transcript);

  int32_t create_stream();
  void free_stream(int32_t stream_id);
  void start_stream(int32_t stream_id);
  void stop_stream(int32_t stream_id);
  void add_audio_to_stream(int32_t stream_id, const float *audio_data,
                           uint64_t audio_length, int32_t sample_rate);
  void transcribe_stream(int32_t stream_id, uint32_t flags,
                         struct transcript_t **out_transcript);
  static std::string
  transcript_to_string(const struct transcript_t *transcript);

  static std::string
  transcript_line_to_string(const struct transcript_line_t *line);

  static std::string *sanitize_text(const char *text);

private:
  void update_transcript_from_segments(
      const std::vector<VoiceActivitySegment> &segments,
      TranscriberStream *stream, struct transcript_t **out_transcript);

  void load_from_files(const char *model_path, uint32_t model_arch);
  void load_from_memory(const uint8_t *encoder_model_data,
                        size_t encoder_model_data_size,
                        const uint8_t *decoder_model_data,
                        size_t decoder_model_data_size,
                        const uint8_t *tokenizer_data,
                        size_t tokenizer_data_size, uint32_t model_arch);

  // Transcribe a single segment using the streaming model
  std::string *transcribe_segment_with_streaming_model(const float *audio_data,
                                                       size_t audio_length);
};

#endif