#ifndef SPEAKER_DIARIZER_H
#define SPEAKER_DIARIZER_H

#include <cstdint>
#include <memory>
#include <vector>

// One contiguous span of speech attributed to a single speaker on the
// stream timeline.
struct SpeakerTurn {
  // Time offset from the start of the stream in seconds.
  float start_time = 0.0f;
  // Length of the span in seconds.
  float duration = 0.0f;
  // Stable identifier for the speaker. Speaker labels coming out of the
  // underlying clustering algorithm can be renumbered when clusters merge or
  // split, so the diarizer maps them onto identifiers that are carried across
  // re-clustering passes by matching speech-time overlap.
  uint64_t speaker_id = 0;
  // The order in which the speaker first appeared, starting at zero.
  uint32_t speaker_index = 0;
};

struct SpeakerDiarizerOptions {
  // Minimum seconds of new audio between re-clustering passes.
  double cluster_cadence = 2.0;
  // Seconds between segmentation/embedding model runs. Zero means use the
  // model default (1 second for the community-1 models). Must be <= 10.
  double analyze_cadence = 0.0;
  // Maximum seconds of audio history fed to VBx per refresh. Zero means
  // unlimited (full-history re-clustering). Default 120 for streaming.
  double cluster_window_sec = 120.0;
};

// Speaker diarization built on the cpp-annote port of the pyannote
// community-1 pipeline (see cpp-annote/README.md). Streaming sessions
// re-cluster a bounded sliding window of recent audio on a cadence; older
// turns are frozen. One-shot diarize() uses full-history clustering.
// Turns from get_turns() can still move within the active window as more
// audio arrives. Callers should treat every call's result as the current
// best estimate for the whole stream.
class SpeakerDiarizer {
 public:
  explicit SpeakerDiarizer(
      const SpeakerDiarizerOptions &options = SpeakerDiarizerOptions());
  ~SpeakerDiarizer();

  SpeakerDiarizer(const SpeakerDiarizer &) = delete;
  SpeakerDiarizer &operator=(const SpeakerDiarizer &) = delete;

  int32_t create_stream();
  void free_stream(int32_t stream_id);
  void start_stream(int32_t stream_id);

  // Appends audio to the stream. Runs the segmentation and embedding models
  // on every new analysis chunk, and a re-clustering pass on the cluster
  // cadence, so this call can be expensive.
  void add_audio_to_stream(int32_t stream_id, const float *audio_data,
                           uint64_t audio_length, int32_t sample_rate);

  // Returns the latest turns for the whole stream, with stable speaker IDs.
  // Does not force any additional computation.
  std::vector<SpeakerTurn> get_turns(int32_t stream_id);

  // Forces a final re-clustering pass and returns the final turns. The
  // result is also cached, so later get_turns() calls return the same data.
  std::vector<SpeakerTurn> finish_stream(int32_t stream_id);

  // One-shot diarization of a complete audio buffer.
  std::vector<SpeakerTurn> diarize(const float *audio_data,
                                   uint64_t audio_length, int32_t sample_rate);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};

#endif  // SPEAKER_DIARIZER_H
