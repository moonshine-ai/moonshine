#include "chunk-policy.h"

#include <algorithm>
#include <cmath>
#include <numbers>

namespace moonshine_tts {
namespace {

/// Zero mean, unit spread, so two signals in different units can be summed.
std::vector<float> standardize(const std::vector<float>& values) {
  std::vector<float> out(values.size(), 0.0f);
  if (values.empty()) {
    return out;
  }
  double sum = 0.0;
  for (const float value : values) {
    sum += value;
  }
  const double mean = sum / static_cast<double>(values.size());
  double variance = 0.0;
  for (const float value : values) {
    const double centred = value - mean;
    variance += centred * centred;
  }
  const double spread =
      std::sqrt(variance / static_cast<double>(values.size()));
  if (spread < 1e-9) {
    return out;
  }
  for (size_t index = 0; index < values.size(); ++index) {
    out[index] = static_cast<float>((values[index] - mean) / spread);
  }
  return out;
}

/// Average adjacent pairs when the signal is at twice the frame rate.
std::vector<float> to_frame_rate(const std::vector<float>& values, int frames) {
  const size_t wanted = static_cast<size_t>(std::max(frames, 0));
  std::vector<float> out(wanted, 0.0f);
  if (wanted == 0) {
    return out;
  }
  if (values.size() >= wanted * 2) {
    for (size_t index = 0; index < wanted; ++index) {
      out[index] = 0.5f * (values[index * 2] + values[index * 2 + 1]);
    }
    return out;
  }
  const size_t shared = std::min(wanted, values.size());
  for (size_t index = 0; index < shared; ++index) {
    out[index] = values[index];
  }
  return out;
}

int seconds_to_frames(float seconds, int frames_per_second) {
  const float frames = seconds * static_cast<float>(frames_per_second);
  return static_cast<int>(std::lround(static_cast<double>(frames)));
}

}  // namespace

std::vector<float> boundary_cost(const std::vector<float>& f0,
                                 const std::vector<float>& energy, int frames) {
  if (frames <= 0) {
    return {};
  }
  const std::vector<float> f0_frames = to_frame_rate(f0, frames);
  const std::vector<float> energy_frames = to_frame_rate(energy, frames);

  std::vector<float> loudness(energy_frames.size());
  for (size_t index = 0; index < energy_frames.size(); ++index) {
    loudness[index] = std::abs(energy_frames[index]);
  }
  // Pitch above the top of the speaking range says nothing about how good a
  // cut is, so it is clamped before standardizing to stop a few high frames
  // from dominating the spread.
  std::vector<float> pitch(f0_frames.size());
  for (size_t index = 0; index < f0_frames.size(); ++index) {
    pitch[index] = std::min(f0_frames[index], 200.0f);
  }

  const std::vector<float> loud = standardize(loudness);
  const std::vector<float> voiced = standardize(pitch);
  std::vector<float> cost(static_cast<size_t>(frames), 0.0f);
  for (size_t index = 0; index < cost.size(); ++index) {
    const float a = index < loud.size() ? loud[index] : 0.0f;
    const float b = index < voiced.size() ? voiced[index] : 0.0f;
    cost[index] = a + 0.5f * b;
  }
  return cost;
}

std::vector<int> plan_boundaries(const std::vector<float>& cost, int frames,
                                 int frames_per_second,
                                 const ChunkPolicyOptions& options) {
  if (frames <= 0 || frames_per_second <= 0) {
    return {0, std::max(frames, 0)};
  }
  const bool can_snap = cost.size() == static_cast<size_t>(frames);
  const int search = std::max(
      0, seconds_to_frames(options.tolerance_seconds, frames_per_second));
  const float growth = std::max(1.0f, options.growth);

  std::vector<int> boundaries{0};
  int current = 0;
  double step = std::max(
      1, seconds_to_frames(options.first_chunk_seconds, frames_per_second));
  while (true) {
    const int span = std::max(1, static_cast<int>(std::lround(step)));
    const int nominal = current + span;
    if (nominal >= frames) {
      break;
    }
    int next = nominal;
    if (can_snap) {
      // Half-open [low, high), with the window reaching `search` frames either
      // side inclusive. Keep this identical to `snap` in
      // scripts/kokoro-stream-prototype.py: the listening tests and the word
      // error figures were measured there, and they only carry over to what
      // ships if both cut in the same places.
      const int low = std::max(current + 2, nominal - search);
      const int high = std::min(frames - 2, nominal + search + 1);
      if (high > low) {
        int best = low;
        for (int frame = low; frame < high; ++frame) {
          if (cost[static_cast<size_t>(frame)] <
              cost[static_cast<size_t>(best)]) {
            best = frame;
          }
        }
        next = best;
      }
    }
    if (next <= current) {
      break;
    }
    boundaries.push_back(next);
    current = next;
    step *= growth;
  }
  boundaries.push_back(frames);
  return boundaries;
}

std::vector<ChunkSpan> plan_spans(const std::vector<int>& boundaries,
                                  int frames, int samples_per_frame,
                                  int sample_rate,
                                  const ChunkPolicyOptions& options) {
  std::vector<ChunkSpan> spans;
  if (boundaries.size() < 2 || frames <= 0 || samples_per_frame <= 0) {
    return spans;
  }
  const size_t total_samples =
      static_cast<size_t>(frames) * static_cast<size_t>(samples_per_frame);
  const size_t crossfade = static_cast<size_t>(std::max(
      0, static_cast<int>(std::lround(options.crossfade_seconds *
                                      static_cast<float>(sample_rate)))));
  // Each chunk reaches half a crossfade past its boundary on both sides, so
  // neighbours overlap by exactly one crossfade and the joined length is the
  // same as an unchunked render.
  const size_t half = crossfade / 2;
  const int pad = std::max(0, options.pad_frames);

  spans.reserve(boundaries.size() - 1);
  for (size_t index = 0; index + 1 < boundaries.size(); ++index) {
    const size_t start = static_cast<size_t>(boundaries[index]) *
                         static_cast<size_t>(samples_per_frame);
    const size_t end = static_cast<size_t>(boundaries[index + 1]) *
                       static_cast<size_t>(samples_per_frame);
    ChunkSpan span;
    span.keep_first_sample = start > half ? start - half : 0;
    span.keep_last_sample = std::min(total_samples, end + half);
    const int first_frame = static_cast<int>(
        span.keep_first_sample / static_cast<size_t>(samples_per_frame));
    const int last_frame = static_cast<int>(
        (span.keep_last_sample + static_cast<size_t>(samples_per_frame) - 1) /
        static_cast<size_t>(samples_per_frame));
    span.decode_first_frame = std::max(0, first_frame - pad);
    span.decode_last_frame = std::min(frames, last_frame + pad);
    spans.push_back(span);
  }
  return spans;
}

void crossfade_append(std::vector<float>& accumulated,
                      const std::vector<float>& piece, size_t overlap) {
  if (accumulated.empty() || overlap == 0) {
    accumulated.insert(accumulated.end(), piece.begin(), piece.end());
    return;
  }
  overlap = std::min({overlap, accumulated.size(), piece.size()});
  if (overlap == 0) {
    accumulated.insert(accumulated.end(), piece.begin(), piece.end());
    return;
  }
  const size_t seam_start = accumulated.size() - overlap;
  for (size_t index = 0; index < overlap; ++index) {
    const float position =
        (static_cast<float>(index) + 0.5f) / static_cast<float>(overlap);
    const float angle = position * static_cast<float>(std::numbers::pi) * 0.5f;
    accumulated[seam_start + index] =
        accumulated[seam_start + index] * std::cos(angle) +
        piece[index] * std::sin(angle);
  }
  accumulated.insert(accumulated.end(),
                     piece.begin() + static_cast<long>(overlap), piece.end());
}

float max_sustainable_growth(const ChunkPolicyOptions& options,
                             float realtime_cost, int frames_per_second) {
  if (realtime_cost <= 0.0f) {
    return options.growth;
  }
  // The first join binds: chunk one must be decoded before chunk zero stops
  // playing. Padding adds a span that does not scale with the chunk, so it is
  // charged explicitly rather than folded into the realtime cost.
  const float first = std::max(options.first_chunk_seconds, 1e-3f);
  const float pad_seconds =
      frames_per_second > 0
          ? 2.0f * static_cast<float>(std::max(0, options.pad_frames)) /
                static_cast<float>(frames_per_second)
          : 0.0f;
  const float budget = first / realtime_cost - pad_seconds;
  return std::max(1.0f, budget / first);
}

bool growth_is_sustainable(const ChunkPolicyOptions& options,
                           float realtime_cost, int frames_per_second) {
  if (realtime_cost <= 0.0f) {
    return true;
  }
  return options.growth <=
         max_sustainable_growth(options, realtime_cost, frames_per_second);
}

}  // namespace moonshine_tts
