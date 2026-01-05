#ifndef TRANSCRIBER_H
#define TRANSCRIBER_H

#include <cinttypes>

#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "moonshine-model.h"
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

  TranscriberLine();
  TranscriberLine(const TranscriberLine &other);
  TranscriberLine &operator=(const TranscriberLine &other);
  ~TranscriberLine();
};

struct TranscriptStreamOutput {
  std::vector<TranscriberLine> internal_lines;
  std::vector<transcript_line_t> output_lines;
  struct transcript_t transcript = {.lines = nullptr, .line_count = 0};
  void clear_update_flags();
  void add_or_update_line(TranscriberLine &line);
  void update_transcript_from_lines();
};

struct TranscriberStream {
  VoiceActivityDetector *vad = nullptr;
  std::mutex vad_mutex;
  TranscriptStreamOutput *transcript_output;
  std::vector<float> new_audio_buffer;
  void add_to_new_audio_buffer(const float *audio_data, uint64_t audio_length,
                               int32_t sample_rate);
  void clear_new_audio_buffer();
  TranscriberStream(VoiceActivityDetector *vad)
      : vad(vad), transcript_output(new TranscriptStreamOutput()) {}
  ~TranscriberStream() {
    delete this->vad;
    delete this->transcript_output;
  }
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
};

class Transcriber {
private:
  TranscriberOptions options;

  MoonshineModel *stt_model;
  std::mutex stt_model_mutex;

  TranscriberStreamMap streams;
  int32_t next_stream_id;
  std::mutex streams_mutex;

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

  static std::string* sanitize_text(const char *text);

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
};

#endif