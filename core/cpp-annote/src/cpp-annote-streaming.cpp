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
  cumulative_profile_ = DiarizationProfile{};
  refresh_count_ = 0;
  snapshot_ = StreamingDiarizationSnapshot{};
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
  // Cache entries for trimmed audio are intentionally kept — VBx runs on the
  // full history of seg/emb results so that clustering quality does not degrade
  // when the audio buffer slides forward.
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

  int new_ort_count = 0;

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
    ++new_ort_count;
  }

  const auto t_after_seg_emb = Clock::now();

  // Assemble full-history tensors from ALL cache entries (sorted by abs
  // offset). VBx sees every chunk ever processed, producing batch-quality
  // clustering even though ORT inference is bounded to the current audio
  // buffer.
  std::vector<int64_t> all_offsets;
  all_offsets.reserve(chunk_cache_.size());
  for (const auto& kv : chunk_cache_) {
    all_offsets.push_back(kv.first);
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

  DiarizationProfile prof;
  prof.segmentation_ort_sec = 0.;
  prof.embedding_ort_sec =
      std::chrono::duration<double>(t_after_seg_emb - t_seg_start).count();

  std::vector<DiarizationTurn> raw =
      engine_.cluster_and_decode(seg_out, emb_all, C_full, prof,
                                 effective_step_sec_);

  prof.segmentation_ort_sec = 0.;
  prof.total_sec =
      std::chrono::duration<double>(Clock::now() - t_seg_start).count();
  cumulative_profile_.accumulate(prof);
  ++refresh_count_;

  // VBx ran on the full history so turns already have stream-absolute
  // timestamps.
  std::vector<StreamingDiarizationTurn> next;
  next.reserve(raw.size());
  for (const DiarizationTurn& t : raw) {
    StreamingDiarizationTurn st;
    static_cast<DiarizationTurn&>(st) = t;
    next.push_back(st);
  }

  const std::vector<StreamingDiarizationTurn> prev = std::move(snapshot_.turns);
  carry_last_updated_times(next, prev, input_end_sec_);

  snapshot_.turns = std::move(next);
  snapshot_.input_end_sec = input_end_sec_;
  snapshot_.window_start_sec = window_start_sec_;
  ++snapshot_.refresh_generation;

  last_refresh_total_chunks_ = static_cast<int>(total_chunks_ever);
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
