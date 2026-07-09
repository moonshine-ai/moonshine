// SPDX-License-Identifier: MIT

#include "cpp-annote-streaming.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <utility>
#include <vector>

#include "wav_pcm_float32.h"

namespace cppannote {
namespace {

double segment_iou(double a0, double a1, double b0, double b1) {
  const double inter = std::max(0., std::min(a1, b1) - std::max(a0, b0));
  const double span = std::max(a1, b1) - std::min(a0, b0);
  if (span <= 1e-12) {
    return 0.;
  }
  return inter / span;
}

bool turns_match(const StreamingDiarizationTurn& a,
                 const StreamingDiarizationTurn& b) {
  constexpr double kTol = 0.05;
  return a.speaker == b.speaker && std::abs(a.start - b.start) < kTol &&
         std::abs(a.end - b.end) < kTol;
}

}  // namespace

StreamingDiarizationSession::StreamingDiarizationSession(
    CppAnnoteEngine& engine, StreamingDiarizationConfig config)
    : engine_(engine), cfg_(std::move(config)) {
  if (cfg_.analyze_cadence == 0.0) {
    effective_step_sec_ = engine_.segmentation_chunk_step_sec();
  } else {
    if (cfg_.analyze_cadence <= 0.0 || cfg_.analyze_cadence > 10.0) {
      throw std::runtime_error(
          "StreamingDiarizationConfig::analyze_cadence must be >0 and <=10");
    }
    effective_step_sec_ = cfg_.analyze_cadence;
  }
  cfg_.cluster_cadence = std::max(0.0, cfg_.cluster_cadence);
  if (cfg_.cluster_window_sec < 0.0) {
    cfg_.cluster_window_sec = 0.0;
  }
  cluster_every_chunks_ =
      (effective_step_sec_ > 0.0)
          ? std::max(1, static_cast<int>(
                            std::lrint(cfg_.cluster_cadence / effective_step_sec_)))
          : 1;
}

void StreamingDiarizationSession::start_session() {
  buffer_.clear();
  input_end_sec_ = 0.;
  window_start_sec_ = 0.;
  buffer_abs_start_samples_ = 0;
  chunk_cache_.clear();
  last_refresh_total_chunks_ = -1;
  frozen_turns_.clear();
  freeze_cutoff_sec_ = 0.;
  prev_active_turns_.clear();
  next_persistent_label_ = 0;
  cumulative_profile_ = DiarizationProfile{};
  refresh_count_ = 0;
  snapshot_ = StreamingDiarizationSnapshot{};
}

double StreamingDiarizationSession::cluster_overlap_margin_sec() const {
  return engine_.segmentation_chunk_duration_sec();
}

double StreamingDiarizationSession::cluster_decode_margin_sec() const {
  // Extra seconds of audio decoded before the clustering window's oldest edge.
  // Frames within one segmentation chunk of the oldest edge are covered by
  // fewer overlapping segmentation windows and get under-detected; since turns
  // are frozen exactly at that edge, that degraded decode would be captured
  // permanently.  Two chunk-durations of left context ensures the frozen band
  // is decoded with the same coverage it had while it was mid-window.
  return 2.0 * engine_.segmentation_chunk_duration_sec();
}

double StreamingDiarizationSession::active_cluster_window_start_sec() const {
  if (cfg_.cluster_window_sec <= 0.0) {
    return 0.0;
  }
  return std::max(0.0, input_end_sec_ - cfg_.cluster_window_sec);
}

double StreamingDiarizationSession::cluster_decode_window_start_sec() const {
  if (cfg_.cluster_window_sec <= 0.0) {
    return 0.0;
  }
  // Decode from a margin before both the active window start and the current
  // freeze cutoff, so every turn about to be frozen this refresh has full
  // left context.  freeze_cutoff_sec_ never exceeds the window start, so this
  // guarantees coverage of the (freeze_cutoff, window_start] band.
  const double base =
      std::min(active_cluster_window_start_sec(), freeze_cutoff_sec_);
  return std::max(0.0, base - cluster_decode_margin_sec());
}

int64_t StreamingDiarizationSession::abs_sample_offset_for_sec(double sec) const {
  const int sr_model = engine_.segmentation_model_sample_rate();
  if (sr_model <= 0) {
    return 0;
  }
  return static_cast<int64_t>(std::llround(sec * static_cast<double>(sr_model)));
}

void StreamingDiarizationSession::evict_chunk_cache_if_needed() {
  if (cfg_.cluster_window_sec <= 0.0) {
    return;
  }
  const double evict_before_sec = std::max(
      0.0, cluster_decode_window_start_sec() - cluster_overlap_margin_sec());
  const int64_t min_abs_off = abs_sample_offset_for_sec(evict_before_sec);
  for (auto it = chunk_cache_.begin(); it != chunk_cache_.end();) {
    if (it->first < min_abs_off) {
      it = chunk_cache_.erase(it);
    } else {
      ++it;
    }
  }
}

void StreamingDiarizationSession::trim_buffer_if_needed() {
  const int sr = engine_.segmentation_model_sample_rate();
  if (sr <= 0) {
    return;
  }
  // Only the tail analysis chunk ever needs raw audio (completed chunks are
  // fully captured in the seg/emb cache).  Keep the chunk window plus two
  // steps of margin so the tail always has room to be recomputed.
  const double keep_sec = engine_.segmentation_chunk_duration_sec() +
                          2.0 * effective_step_sec_;
  const int step_samples = static_cast<int>(std::lrint(
      effective_step_sec_ * static_cast<double>(sr)));
  const std::size_t cap = static_cast<std::size_t>(std::max(1., keep_sec) *
                                                   static_cast<double>(sr));
  if (buffer_.size() <= cap) {
    return;
  }
  std::size_t drop = buffer_.size() - cap;
  if (step_samples > 0) {
    drop = (drop / static_cast<std::size_t>(step_samples)) *
           static_cast<std::size_t>(step_samples);
  }
  if (drop == 0) {
    return;
  }
  buffer_.erase(buffer_.begin(),
                buffer_.begin() + static_cast<std::ptrdiff_t>(drop));
  window_start_sec_ += static_cast<double>(drop) / static_cast<double>(sr);
  buffer_abs_start_samples_ += static_cast<int64_t>(drop);
}

void StreamingDiarizationSession::cache_new_chunks() {
  const int sr_model = engine_.segmentation_model_sample_rate();
  const int num_channels = engine_.segmentation_num_channels();
  const int chunk_num_samples = engine_.segmentation_chunk_num_samples();
  const int step_samples = static_cast<int>(std::lrint(
      effective_step_sec_ * static_cast<double>(sr_model)));
  if (step_samples <= 0 || chunk_num_samples <= 0) {
    return;
  }
  const int64_t num_samples_i = static_cast<int64_t>(buffer_.size());
  int64_t num_complete_chunks = 0;
  if (num_samples_i >= chunk_num_samples) {
    num_complete_chunks =
        (num_samples_i - chunk_num_samples) / step_samples + 1;
  }
  for (int64_t c = 0; c < num_complete_chunks; ++c) {
    const int64_t buf_off = c * step_samples;
    const int64_t abs_off = buffer_abs_start_samples_ + buf_off;
    if (chunk_cache_.count(abs_off)) {
      continue;
    }
    auto chunk_buf = CppAnnoteEngine::extract_chunk_audio(
        buffer_.data(), num_samples_i, buf_off, chunk_num_samples,
        num_channels);
    auto seg = engine_.run_segmentation_ort_single(chunk_buf.data());
    auto mono = CppAnnoteEngine::extract_chunk_audio(
        buffer_.data(), num_samples_i, buf_off, chunk_num_samples, 1);
    auto emb_chunk = engine_.run_embedding_ort_single(mono.data(), seg.data());
    chunk_cache_[abs_off] = CachedChunk{std::move(seg), std::move(emb_chunk)};
  }
}

void StreamingDiarizationSession::add_audio_chunk(const float* pcm,
                                                  std::size_t num_samples,
                                                  int sample_rate) {
  if (pcm == nullptr || num_samples == 0) {
    snapshot_.input_end_sec = input_end_sec_;
    return;
  }
  if (sample_rate <= 0) {
    throw std::runtime_error(
        "StreamingDiarizationSession: sample_rate must be positive");
  }
  const int sr_model = engine_.segmentation_model_sample_rate();
  std::vector<float> chunk(pcm, pcm + num_samples);
  std::vector<float> res =
      wav_pcm::linear_resample(chunk, sample_rate, sr_model);
  buffer_.insert(buffer_.end(), res.begin(), res.end());
  input_end_sec_ +=
      static_cast<double>(num_samples) / static_cast<double>(sample_rate);
  cache_new_chunks();
  trim_buffer_if_needed();
  evict_chunk_cache_if_needed();
  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;
  maybe_refresh(false);
}

void StreamingDiarizationSession::carry_last_updated_times(
    std::vector<StreamingDiarizationTurn>& next,
    const std::vector<StreamingDiarizationTurn>& prev, double input_end_sec) {
  constexpr double kTol = 0.25;
  constexpr double kIouMin = 0.2;
  for (auto& t : next) {
    t.last_updated_at_input_end_sec = input_end_sec;
    double best_iou = 0.;
    const StreamingDiarizationTurn* best = nullptr;
    for (const auto& p : prev) {
      const double i = segment_iou(t.start, t.end, p.start, p.end);
      if (i > best_iou) {
        best_iou = i;
        best = &p;
      }
    }
    if (best != nullptr && best_iou >= kIouMin && best->speaker == t.speaker &&
        std::abs(t.start - best->start) < kTol &&
        std::abs(t.end - best->end) < kTol) {
      t.last_updated_at_input_end_sec = best->last_updated_at_input_end_sec;
    }
  }
}

void StreamingDiarizationSession::append_frozen_turn_if_new(
    std::vector<StreamingDiarizationTurn>& frozen,
    const StreamingDiarizationTurn& turn) {
  for (const auto& existing : frozen) {
    if (turns_match(existing, turn)) {
      return;
    }
  }
  frozen.push_back(turn);
}

void StreamingDiarizationSession::relabel_active_turns(
    std::vector<StreamingDiarizationTurn>& active_turns) {
  if (cfg_.cluster_window_sec <= 0.0 || active_turns.empty()) {
    return;
  }

  // Distinct raw labels in the current window, in first-appearance order.
  std::vector<int32_t> cur_labels;
  for (const auto& t : active_turns) {
    if (std::find(cur_labels.begin(), cur_labels.end(), t.speaker) ==
        cur_labels.end()) {
      cur_labels.push_back(t.speaker);
    }
  }

  std::map<int32_t, int32_t> remap;

  if (!prev_active_turns_.empty()) {
    // Consecutive windows overlap by (cluster_window_sec - cluster_cadence),
    // so both speakers are almost always present in both.  Match current raw
    // labels onto the persistent labels of the previous window by greatest
    // total speech-time overlap, each persistent label claimed at most once.
    std::vector<int32_t> prev_labels;
    for (const auto& t : prev_active_turns_) {
      if (std::find(prev_labels.begin(), prev_labels.end(), t.speaker) ==
          prev_labels.end()) {
        prev_labels.push_back(t.speaker);
      }
    }

    std::vector<std::pair<double, std::pair<int32_t, int32_t>>> candidates;
    for (int32_t cur : cur_labels) {
      for (int32_t prev : prev_labels) {
        double overlap = 0.0;
        for (const auto& a : active_turns) {
          if (a.speaker != cur) {
            continue;
          }
          for (const auto& b : prev_active_turns_) {
            if (b.speaker != prev) {
              continue;
            }
            const double o = std::min(a.end, b.end) - std::max(a.start, b.start);
            if (o > 0.0) {
              overlap += o;
            }
          }
        }
        if (overlap > 0.0) {
          candidates.push_back({overlap, {cur, prev}});
        }
      }
    }
    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    std::map<int32_t, bool> cur_taken;
    std::map<int32_t, bool> prev_taken;
    for (const auto& c : candidates) {
      const int32_t cur = c.second.first;
      const int32_t prev = c.second.second;
      if (cur_taken.count(cur) || prev_taken.count(prev)) {
        continue;
      }
      remap[cur] = prev;
      cur_taken[cur] = true;
      prev_taken[prev] = true;
    }
  }

  // Any current label without a match is a newly-seen speaker.
  for (int32_t cur : cur_labels) {
    if (!remap.count(cur)) {
      remap[cur] = next_persistent_label_++;
    }
  }

  for (auto& t : active_turns) {
    t.speaker = remap.at(t.speaker);
  }
  prev_active_turns_ = active_turns;
}

void StreamingDiarizationSession::merge_frozen_and_active_turns(
    std::vector<StreamingDiarizationTurn> active_turns) {
  if (cfg_.cluster_window_sec <= 0.0) {
    snapshot_.turns = std::move(active_turns);
    return;
  }

  const double window_start_sec = active_cluster_window_start_sec();
  const double prev_freeze_cutoff = freeze_cutoff_sec_;
  // Freeze turns once they slide past the clustering window start.
  const double new_freeze_cutoff =
      std::max(prev_freeze_cutoff, window_start_sec);
  freeze_cutoff_sec_ = new_freeze_cutoff;

  // Freeze only turns in the newly-passed band (prev_freeze_cutoff,
  // new_freeze_cutoff] from the current decode.  The decode window carries a
  // left-context margin (see cluster_decode_window_start_sec), so this band is
  // decoded with the same coverage it had while it was mid-window rather than
  // the under-detected oldest-edge decode.  Turns ending at or before the
  // previous cutoff were already frozen in an earlier refresh; the current run
  // still re-decodes them (they sit in the margin) but they must not be frozen
  // again.
  std::vector<StreamingDiarizationTurn> active_portion;
  active_portion.reserve(active_turns.size());
  for (auto& turn : active_turns) {
    if (turn.end <= new_freeze_cutoff) {
      if (turn.end > prev_freeze_cutoff) {
        append_frozen_turn_if_new(frozen_turns_, turn);
      }
    } else {
      active_portion.push_back(std::move(turn));
    }
  }

  std::vector<StreamingDiarizationTurn> merged;
  merged.reserve(frozen_turns_.size() + active_portion.size());
  merged.insert(merged.end(), frozen_turns_.begin(), frozen_turns_.end());
  merged.insert(merged.end(), active_portion.begin(), active_portion.end());
  std::sort(merged.begin(), merged.end(),
            [](const StreamingDiarizationTurn& a,
               const StreamingDiarizationTurn& b) {
              if (a.start != b.start) {
                return a.start < b.start;
              }
              if (a.end != b.end) {
                return a.end < b.end;
              }
              return a.speaker < b.speaker;
            });
  snapshot_.turns = std::move(merged);
}

void StreamingDiarizationSession::maybe_refresh(bool force) {
  using Clock = std::chrono::steady_clock;

  const int sr_model = engine_.segmentation_model_sample_rate();
  const int num_channels = engine_.segmentation_num_channels();
  const int chunk_num_samples = engine_.segmentation_chunk_num_samples();
  const int step_samples = static_cast<int>(
      std::lrint(effective_step_sec_ * static_cast<double>(sr_model)));
  if (step_samples <= 0 || chunk_num_samples <= 0) {
    return;
  }

  const int64_t num_samples_i = static_cast<int64_t>(buffer_.size());
  int64_t num_complete_chunks = 0;
  if (num_samples_i >= chunk_num_samples) {
    num_complete_chunks =
        (num_samples_i - chunk_num_samples) / step_samples + 1;
  }
  const bool has_last =
      (num_samples_i < chunk_num_samples) ||
      ((num_samples_i - chunk_num_samples) % step_samples > 0);
  const int64_t total_chunks = num_complete_chunks + (has_last ? 1 : 0);
  if (total_chunks <= 0) {
    return;
  }

  // Use total_chunks_ever (based on absolute stream position) for cadence, not
  // the buffer's chunk count which saturates once the buffer is full.
  const int64_t total_chunks_ever =
      (buffer_abs_start_samples_ + num_samples_i >= chunk_num_samples)
          ? (buffer_abs_start_samples_ + num_samples_i - chunk_num_samples) /
                    step_samples +
                1
          : 0;

  if (!force) {
    if (last_refresh_total_chunks_ >= 0) {
      if (total_chunks_ever <
          last_refresh_total_chunks_ + cluster_every_chunks_) {
        return;
      }
    }
  }

  const auto t_seg_start = Clock::now();

  // Complete chunks are already cached by cache_new_chunks().
  // Only the partial tail chunk (zero-padded) needs ORT here.
  int64_t tail_abs_off = -1;
  if (has_last) {
    const int64_t buf_off = num_complete_chunks * step_samples;
    const int64_t abs_off = buffer_abs_start_samples_ + buf_off;
    tail_abs_off = abs_off;
    auto chunk_buf = CppAnnoteEngine::extract_chunk_audio(
        buffer_.data(), num_samples_i, buf_off, chunk_num_samples,
        num_channels);
    auto seg = engine_.run_segmentation_ort_single(chunk_buf.data());
    auto mono = CppAnnoteEngine::extract_chunk_audio(
        buffer_.data(), num_samples_i, buf_off, chunk_num_samples, 1);
    auto emb_chunk = engine_.run_embedding_ort_single(mono.data(), seg.data());
    chunk_cache_[abs_off] = CachedChunk{std::move(seg), std::move(emb_chunk)};
  }

  const auto t_after_seg_emb = Clock::now();

  // Include a margin of chunks before the window start so the oldest edge
  // (where turns freeze) is decoded with full segmentation context.
  const double decode_window_start_sec = cluster_decode_window_start_sec();
  const int64_t min_abs_off =
      (cfg_.cluster_window_sec > 0.0)
          ? abs_sample_offset_for_sec(decode_window_start_sec)
          : 0;

  std::vector<int64_t> all_offsets;
  all_offsets.reserve(chunk_cache_.size());
  for (const auto& kv : chunk_cache_) {
    if (cfg_.cluster_window_sec <= 0.0 || kv.first >= min_abs_off) {
      all_offsets.push_back(kv.first);
    }
  }
  std::sort(all_offsets.begin(), all_offsets.end());

  const int F = engine_.seg_frames_per_chunk();
  const int K = engine_.seg_classes();
  const int dim = engine_.embedding_dimension();
  const int FK = F * K;
  const int C_full = static_cast<int>(all_offsets.size());
  if (C_full == 0) {
    return;
  }

  std::vector<float> seg_out(static_cast<size_t>(C_full) *
                             static_cast<size_t>(FK));
  std::vector<float> emb_all(static_cast<size_t>(C_full) *
                                 static_cast<size_t>(K) *
                                 static_cast<size_t>(dim),
                             std::numeric_limits<float>::quiet_NaN());

  for (int i = 0; i < C_full; ++i) {
    const auto& cached = chunk_cache_.at(all_offsets[i]);
    std::memcpy(&seg_out[static_cast<size_t>(i) * static_cast<size_t>(FK)],
                cached.seg.data(), static_cast<size_t>(FK) * sizeof(float));
    std::memcpy(
        &emb_all[static_cast<size_t>(i) * static_cast<size_t>(K) *
                 static_cast<size_t>(dim)],
        cached.emb.data(),
        static_cast<size_t>(K) * static_cast<size_t>(dim) * sizeof(float));
  }

  // Evict the tail chunk from cache — it was computed with zero-padded audio
  // and its content will change as more audio arrives.
  if (tail_abs_off >= 0) {
    chunk_cache_.erase(tail_abs_off);
  }

  const double chunks_start_sec =
      static_cast<double>(all_offsets.front()) / static_cast<double>(sr_model);

  DiarizationProfile prof;
  prof.segmentation_ort_sec = 0.;
  prof.embedding_ort_sec =
      std::chrono::duration<double>(t_after_seg_emb - t_seg_start).count();

  std::vector<DiarizationTurn> raw = engine_.cluster_and_decode(
      seg_out, emb_all, C_full, prof, effective_step_sec_, chunks_start_sec);

  prof.segmentation_ort_sec = 0.;
  prof.total_sec =
      std::chrono::duration<double>(Clock::now() - t_seg_start).count();
  cumulative_profile_.accumulate(prof);
  ++refresh_count_;

  std::vector<StreamingDiarizationTurn> active_turns;
  active_turns.reserve(raw.size());
  for (const DiarizationTurn& t : raw) {
    StreamingDiarizationTurn st;
    static_cast<DiarizationTurn&>(st) = t;
    active_turns.push_back(st);
  }

  relabel_active_turns(active_turns);

  const std::vector<StreamingDiarizationTurn> prev = snapshot_.turns;
  merge_frozen_and_active_turns(std::move(active_turns));
  carry_last_updated_times(snapshot_.turns, prev, input_end_sec_);

  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;
  ++snapshot_.refresh_generation;

  last_refresh_total_chunks_ = static_cast<int>(total_chunks_ever);
  evict_chunk_cache_if_needed();
}

StreamingDiarizationSnapshot StreamingDiarizationSession::snapshot() const {
  return snapshot_;
}

StreamingDiarizationSnapshot
StreamingDiarizationSession::refresh_and_snapshot() {
  maybe_refresh(true);
  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;
  return snapshot_;
}

StreamingDiarizationSnapshot StreamingDiarizationSession::end_session() {
  maybe_refresh(true);
  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;

  return snapshot_;
}

}  // namespace cppannote
