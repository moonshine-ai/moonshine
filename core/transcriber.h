#ifndef TRANSCRIBER_H
#define TRANSCRIBER_H

#include <atomic>
#include <chrono>
#include <cinttypes>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "moonshine-model.h"
#include "moonshine-streaming-model.h"
#include "voice-activity-detector.h"
#include "speaker-embedding-model.h"
#include "speaker-embedding-model-data.h"
#include "online-clusterer.h"

struct TranscriberLine {
  std::string *text = nullptr;
  std::vector<float> audio_data;
  float start_time;
  float duration;
  bool is_complete;
  bool just_updated;
  bool is_new;
  bool has_text_changed;
  bool has_speaker_id;
  uint64_t id;
  uint32_t last_transcription_latency_ms;
  uint64_t speaker_id;
  uint32_t speaker_index;

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
  std::mutex mutex;

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
  bool identify_speakers = true;
  float transcription_interval = 0.5f;
  float vad_threshold = 0.5f;  
  float vad_window_duration = 0.5f;
  int32_t vad_hop_size = 512;
  size_t vad_look_behind_sample_count = 8192;
  float vad_max_segment_duration = 15.0f;
  float max_tokens_per_second = 6.5f;
  float speaker_id_cluster_threshold = 0.6f;
  std::string save_input_wav_path = "";
  bool log_ort_run = false;
  bool return_audio_data = true;
  bool log_output_text = false;
  // When true, encoder memory and KV caches persist across segment
  // boundaries.  A full reset only happens when memory is full.
  bool keep_context = false;
};

class Transcriber {
 private:
  TranscriberOptions options;

  // Non-streaming model (used for TINY and BASE architectures)
  MoonshineModel *stt_model;
  std::mutex stt_model_mutex;

  // Streaming model (used for TINY_STREAMING and BASE_STREAMING architectures)
  MoonshineStreamingModel *streaming_model;
  MoonshineStreamingState streaming_state;
  std::mutex streaming_model_mutex;

  SpeakerEmbeddingModel *speaker_embedding_model;
  std::mutex speaker_embedding_model_mutex;
  OnlineClusterer *online_clusterer;
  uint32_t next_speaker_index = 0;
  std::map<uint64_t, uint32_t> speaker_index_map;

  // Track current segment for incremental processing
  uint64_t current_streaming_segment_id = UINT64_MAX;
  size_t streaming_samples_processed = 0;

  // Tokens from the most recent decode of the CURRENT segment.
  std::vector<int64_t> sw_current_tokens_;
  // Most recent text output of the CURRENT segment.
  std::string sw_current_line_text_;
  // Fixed teacher-forcing prefix for the current segment.  Updated only at
  // segment boundaries: empty (=> [BOS] prefix) after a full reset, or the
  // previous segment's output when keep_context preserved memory across
  // the boundary.  Using a FIXED prefix for all in-segment decodes prevents
  // committing to unstable tokens from short early partial decodes.
  std::vector<int64_t> sw_decode_prefix_;
  // Cached decoder self-attention KV state AFTER feeding all-but-the-last
  // token of sw_decode_prefix_.  Avoids re-feeding the whole prefix on
  // every in-segment decode call.  Invalidated on segment boundaries.
  // Also tracks the encoder memory_len at snapshot time: each layer's
  // self-attention K/V depends on the residual stream which includes
  // cross-attention output — so if memory grew since the snapshot, the
  // cached K/V at those positions is stale and must not be reused.
  std::vector<float> sw_prefix_k_self_;
  std::vector<float> sw_prefix_v_self_;
  int sw_prefix_cache_seq_len_ = 0;
  int sw_prefix_memory_len_ = 0;
  bool sw_prefix_kv_cached_ = false;

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
  static std::string transcript_to_string(
      const struct transcript_t *transcript);

  static std::string transcript_line_to_string(
      const struct transcript_line_t *line);

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

  std::string *transcribe_segment_with_streaming_model(const float *audio_data,
                                                       size_t audio_length,
                                                       uint64_t segment_id,
                                                       bool is_final);
};

#endif