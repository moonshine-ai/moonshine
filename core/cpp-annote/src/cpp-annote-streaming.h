// SPDX-License-Identifier: MIT
// Realtime-oriented session API: append PCM chunks at arbitrary rates, run full
// diarization on a bounded model-rate buffer on a coarse cadence (VBx /
// reconstruct are batch over the window). Internal implementation detail —
// public callers should use CppAnnote (cpp-annote.h).

#ifndef CPP_ANNOTE_STREAMING_H_
#define CPP_ANNOTE_STREAMING_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "cpp-annote-engine.h"

namespace cppannote {

struct StreamingDiarizationConfig {
  /// Seconds between segmentation+embedding model runs (sliding-window step).
  /// Must be >0 and <=10.  Defaults to 0 which means "use the model's
  /// built-in ``chunk_step_sec``" (typically 1.0 s for community-1).
  double analyze_cadence = 0.0;

  /// Minimum seconds of new audio between VBx re-clustering passes.  Converted
  /// internally to an analysis-chunk count using the effective analyze cadence.
  double cluster_cadence = 2.0;

  /// Maximum seconds of seg/emb history fed to VBx on each refresh.  Older
  /// chunks are evicted from the cache and their turns are frozen.  Zero means
  /// unlimited (full-history re-clustering).
  double cluster_window_sec = 120.0;
};

struct StreamingDiarizationTurn : DiarizationTurn {
  /// Last time this ``(start, end, speaker)`` matched a prior snapshot; bumped
  /// when overlap match fails or bounds/label change beyond tolerance after a
  /// refresh.
  double last_updated_at_input_end_sec = 0.;
};

struct StreamingDiarizationSnapshot {
  std::vector<StreamingDiarizationTurn> turns;
  /// Cumulative duration of audio appended on this session (input timeline,
  /// from chunk lengths).
  double input_end_sec = 0.;
  /// Time on the input timeline corresponding to ``buffer[0]`` (after
  /// trimming).
  double window_start_sec = 0.;
  int refresh_generation = 0;
};

/// Session bound to a ``CppAnnoteEngine``; the engine must outlive the session.
class StreamingDiarizationSession {
 public:
  StreamingDiarizationSession(CppAnnoteEngine& engine,
                              StreamingDiarizationConfig config = {});

  void start_session();
  /// Append ``num_samples`` mono ``pcm`` at ``sample_rate`` Hz; resamples each
  /// chunk to the engine model rate and concatenates on the session timeline.
  /// At most one new complete analysis window (segmentation + embedding) runs
  /// per call, so a backlog cannot turn one transcription pass into several
  /// overlapping ORT windows. Remaining complete windows stay buffered and
  /// are analyzed on later calls; ``end_session`` drains them.
  void add_audio_chunk(const float* pcm, std::size_t num_samples,
                       int sample_rate);
  /// Current best snapshot (updated on refresh cadence; ``input_end_sec``
  /// advances every chunk).
  StreamingDiarizationSnapshot snapshot() const;

  /// Force a VBx refresh and return the updated snapshot. Drains any complete
  /// analysis windows that were deferred by the one-window-per-append cap.
  StreamingDiarizationSnapshot refresh_and_snapshot();

  /// Drain remaining complete windows, run a final refresh (including the
  /// padded tail), and return the snapshot.
  StreamingDiarizationSnapshot end_session();

  StreamingDiarizationSession(const StreamingDiarizationSession&) = delete;
  StreamingDiarizationSession& operator=(const StreamingDiarizationSession&) =
      delete;

 private:
  struct CachedChunk {
    std::vector<float> seg;  // (F * K)
    std::vector<float> emb;  // (K * dim)
  };

  /// Analyzes up to ``max_chunks`` new complete windows (``-1`` = all).
  /// Returns how many were added to the cache.
  int cache_new_chunks(int max_chunks);
  CachedChunk analyze_buffer_chunk(int64_t buf_off, int64_t num_samples);
  int64_t complete_chunk_count(int64_t num_samples) const;
  int step_samples() const;
  void trim_buffer_if_needed();
  void evict_chunk_cache_if_needed();
  void maybe_refresh(bool force, bool allow_tail_ort);
  static void carry_last_updated_times(
      std::vector<StreamingDiarizationTurn>& next,
      const std::vector<StreamingDiarizationTurn>& prev, double input_end_sec);
  static void append_frozen_turn_if_new(
      std::vector<StreamingDiarizationTurn>& frozen,
      const StreamingDiarizationTurn& turn);
  void merge_frozen_and_active_turns(
      std::vector<StreamingDiarizationTurn> active_turns);
  // Relabels the raw per-window clustering labels of `active_turns` into a
  // persistent namespace that is stable across refreshes, so frozen turns and
  // the active window always use the same integer for the same speaker.
  void relabel_active_turns(std::vector<StreamingDiarizationTurn>& active_turns);
  double cluster_overlap_margin_sec() const;
  double cluster_decode_margin_sec() const;
  double active_cluster_window_start_sec() const;
  double cluster_decode_window_start_sec() const;
  int64_t abs_sample_offset_for_sec(double sec) const;

  CppAnnoteEngine& engine_;
  StreamingDiarizationConfig cfg_{};
  double effective_step_sec_ = 1.0;  // resolved analyze_cadence
  int cluster_every_chunks_ =
      1;  // derived from cfg_.cluster_cadence / effective_step_sec_

  std::vector<float> buffer_;
  double input_end_sec_ = 0.;
  double window_start_sec_ = 0.;
  int64_t buffer_abs_start_samples_ = 0;

  std::unordered_map<int64_t, CachedChunk> chunk_cache_;

  int last_refresh_analyzed_complete_ = -1;
  int64_t analyzed_complete_chunks_ = 0;

  std::vector<StreamingDiarizationTurn> frozen_turns_;
  double freeze_cutoff_sec_ = 0.;

  // Last refresh's active turns, already mapped into the persistent label
  // namespace, plus the next unused persistent label.  Used to stitch each
  // new window's raw clustering labels onto a stable namespace.
  std::vector<StreamingDiarizationTurn> prev_active_turns_;
  int next_persistent_label_ = 0;

  DiarizationProfile cumulative_profile_{};
  int refresh_count_ = 0;

  StreamingDiarizationSnapshot snapshot_;
};

}  // namespace cppannote

#endif  // CPP_ANNOTE_STREAMING_H_
