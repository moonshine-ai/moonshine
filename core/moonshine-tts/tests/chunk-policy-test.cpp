#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "chunk-policy.h"

#include <doctest/doctest.h>

#include <cmath>
#include <vector>

using namespace moonshine_tts;

namespace {

constexpr int kFramesPerSecond = 40;
constexpr int kSamplesPerFrame = 600;
constexpr int kSampleRate = 24000;

ChunkPolicyOptions default_options() {
  ChunkPolicyOptions options;
  options.first_chunk_seconds = 0.5f;
  options.tolerance_seconds = 0.0f;
  options.crossfade_seconds = 0.025f;
  options.growth = 2.0f;
  options.pad_frames = 8;
  return options;
}

/// Lengths in frames of each chunk the boundaries describe.
std::vector<int> chunk_lengths(const std::vector<int>& boundaries) {
  std::vector<int> lengths;
  for (size_t index = 0; index + 1 < boundaries.size(); ++index) {
    lengths.push_back(boundaries[index + 1] - boundaries[index]);
  }
  return lengths;
}

}  // namespace

TEST_CASE("growth doubles each chunk after the first") {
  ChunkPolicyOptions options = default_options();
  const int frames = 40 * 20;  // 20 seconds
  const std::vector<int> boundaries =
      plan_boundaries({}, frames, kFramesPerSecond, options);
  const std::vector<int> lengths = chunk_lengths(boundaries);

  REQUIRE(lengths.size() >= 4);
  CHECK(lengths[0] == 20);   // 0.5 s
  CHECK(lengths[1] == 40);   // 1.0 s
  CHECK(lengths[2] == 80);   // 2.0 s
  CHECK(lengths[3] == 160);  // 4.0 s
  CHECK(boundaries.front() == 0);
  CHECK(boundaries.back() == frames);
}

TEST_CASE("a growth of one gives the uniform grid") {
  ChunkPolicyOptions options = default_options();
  options.growth = 1.0f;
  const int frames = 40 * 5;
  const std::vector<int> lengths =
      chunk_lengths(plan_boundaries({}, frames, kFramesPerSecond, options));

  REQUIRE(lengths.size() >= 2);
  for (size_t index = 0; index + 1 < lengths.size(); ++index) {
    CHECK(lengths[index] == 20);
  }
}

TEST_CASE("growing produces fewer chunks than the uniform grid") {
  ChunkPolicyOptions growing = default_options();
  ChunkPolicyOptions uniform = default_options();
  uniform.growth = 1.0f;
  const int frames = 40 * 12;

  const size_t grown =
      plan_boundaries({}, frames, kFramesPerSecond, growing).size();
  const size_t flat =
      plan_boundaries({}, frames, kFramesPerSecond, uniform).size();
  CHECK(grown < flat);
}

TEST_CASE(
    "the first chunk is the same length either way, so latency is equal") {
  ChunkPolicyOptions growing = default_options();
  ChunkPolicyOptions uniform = default_options();
  uniform.growth = 1.0f;
  const int frames = 40 * 12;

  const std::vector<int> a =
      plan_boundaries({}, frames, kFramesPerSecond, growing);
  const std::vector<int> b =
      plan_boundaries({}, frames, kFramesPerSecond, uniform);
  REQUIRE(a.size() >= 2);
  REQUIRE(b.size() >= 2);
  CHECK(a[1] == b[1]);
}

TEST_CASE("a boundary snaps to the cheapest frame within tolerance") {
  ChunkPolicyOptions options = default_options();
  options.tolerance_seconds = 0.2f;  // 8 frames either side
  const int frames = 400;
  std::vector<float> cost(static_cast<size_t>(frames), 1.0f);
  // Nominal first boundary is frame 20; put a much better cut at 25.
  cost[25] = -5.0f;

  const std::vector<int> boundaries =
      plan_boundaries(cost, frames, kFramesPerSecond, options);
  REQUIRE(boundaries.size() >= 2);
  CHECK(boundaries[1] == 25);
}

TEST_CASE("snapping never moves a boundary beyond its tolerance") {
  ChunkPolicyOptions options = default_options();
  options.tolerance_seconds = 0.1f;  // 4 frames either side
  const int frames = 400;
  std::vector<float> cost(static_cast<size_t>(frames), 1.0f);
  cost[35] = -5.0f;  // far outside the window around frame 20

  const std::vector<int> boundaries =
      plan_boundaries(cost, frames, kFramesPerSecond, options);
  REQUIRE(boundaries.size() >= 2);
  CHECK(boundaries[1] >= 16);
  CHECK(boundaries[1] <= 24);
}

TEST_CASE("quiet unvoiced frames are the cheapest place to cut") {
  const int frames = 8;
  // Frame 5 is both silent and unvoiced, so it should win outright.
  std::vector<float> energy{1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f, 1.0f, 1.0f};
  std::vector<float> f0{150.0f, 150.0f, 150.0f, 150.0f,
                        150.0f, 0.0f,   150.0f, 150.0f};

  const std::vector<float> cost = boundary_cost(f0, energy, frames);
  REQUIRE(cost.size() == static_cast<size_t>(frames));
  for (int frame = 0; frame < frames; ++frame) {
    if (frame != 5) {
      CHECK(cost[5] < cost[static_cast<size_t>(frame)]);
    }
  }
}

TEST_CASE("f0 and energy at twice the frame rate are averaged down") {
  const int frames = 2;
  std::vector<float> energy{0.0f, 2.0f, 4.0f, 6.0f};
  std::vector<float> f0{0.0f, 2.0f, 4.0f, 6.0f};
  const std::vector<float> cost = boundary_cost(f0, energy, frames);
  CHECK(cost.size() == 2);
  // Averaging gives frames of 1 and 5, so the second frame must cost more.
  CHECK(cost[0] < cost[1]);
}

TEST_CASE("spans cover the utterance once and overlap by one crossfade") {
  ChunkPolicyOptions options = default_options();
  const int frames = 200;
  const std::vector<int> boundaries =
      plan_boundaries({}, frames, kFramesPerSecond, options);
  const std::vector<ChunkSpan> spans =
      plan_spans(boundaries, frames, kSamplesPerFrame, kSampleRate, options);

  REQUIRE(spans.size() == boundaries.size() - 1);
  const size_t crossfade = static_cast<size_t>(0.025f * kSampleRate);
  CHECK(spans.front().keep_first_sample == 0);
  CHECK(spans.back().keep_last_sample ==
        static_cast<size_t>(frames) * kSamplesPerFrame);
  for (size_t index = 0; index + 1 < spans.size(); ++index) {
    const size_t overlap =
        spans[index].keep_last_sample - spans[index + 1].keep_first_sample;
    CHECK(overlap == crossfade - crossfade % 2);
  }
}

TEST_CASE("decode spans carry padding but stay inside the utterance") {
  ChunkPolicyOptions options = default_options();
  const int frames = 200;
  const std::vector<int> boundaries =
      plan_boundaries({}, frames, kFramesPerSecond, options);
  const std::vector<ChunkSpan> spans =
      plan_spans(boundaries, frames, kSamplesPerFrame, kSampleRate, options);

  for (const ChunkSpan& span : spans) {
    CHECK(span.decode_first_frame >= 0);
    CHECK(span.decode_last_frame <= frames);
    CHECK(span.decode_first_frame < span.decode_last_frame);
    const size_t decoded_first =
        static_cast<size_t>(span.decode_first_frame) * kSamplesPerFrame;
    const size_t decoded_last =
        static_cast<size_t>(span.decode_last_frame) * kSamplesPerFrame;
    CHECK(decoded_first <= span.keep_first_sample);
    CHECK(decoded_last >= span.keep_last_sample);
  }
  // Interior chunks should actually be padded, not merely valid.
  if (spans.size() > 2) {
    CHECK(spans[1].decode_first_frame < boundaries[1]);
  }
}

TEST_CASE("a crossfade holds power constant through the join") {
  const size_t overlap = 600;
  std::vector<float> a(2000, 1.0f);
  const std::vector<float> b(2000, 1.0f);
  crossfade_append(a, b, overlap);

  CHECK(a.size() == 2000 + 2000 - overlap);
  // Sine-law on two correlated constant signals sums to at least the original
  // amplitude everywhere, and never overshoots by more than the 3 dB that
  // equal-power crossfading of identical material implies.
  for (size_t index = 1400; index < 2000; ++index) {
    CHECK(a[index] >= doctest::Approx(1.0f).epsilon(0.01));
    CHECK(a[index] <= doctest::Approx(std::sqrt(2.0f)).epsilon(0.01));
  }
}

TEST_CASE("appending to nothing copies the piece verbatim") {
  std::vector<float> empty;
  const std::vector<float> piece{1.0f, 2.0f, 3.0f};
  crossfade_append(empty, piece, 100);
  CHECK(empty == piece);
}

TEST_CASE("an overlap longer than either side is clamped") {
  std::vector<float> a{1.0f, 1.0f};
  const std::vector<float> b{1.0f, 1.0f, 1.0f};
  crossfade_append(a, b, 999);
  CHECK(a.size() == 3);
}

TEST_CASE("doubling needs a decoder at twice realtime, with padding on top") {
  ChunkPolicyOptions options = default_options();
  options.pad_frames = 0;
  options.growth = 2.0f;

  // Exactly 2x realtime is the marginal case with no padding to pay for.
  CHECK(growth_is_sustainable(options, 0.5f, kFramesPerSecond));
  CHECK_FALSE(growth_is_sustainable(options, 0.55f, kFramesPerSecond));
  CHECK(growth_is_sustainable(options, 0.083f, kFramesPerSecond));

  // Padding is charged per chunk whatever its length, so it weighs most on the
  // short early chunks. Eight frames either side is 0.4 s on top of a 1.0 s
  // second chunk, which is enough to turn a decoder that clears the no-padding
  // test into one that underruns.
  options.pad_frames = 8;
  CHECK(growth_is_sustainable(options, 0.35f, kFramesPerSecond));
  CHECK_FALSE(growth_is_sustainable(options, 0.4f, kFramesPerSecond));
  CHECK_FALSE(growth_is_sustainable(options, 0.5f, kFramesPerSecond));
}

TEST_CASE("the sustainable growth factor tracks decoder speed") {
  ChunkPolicyOptions options = default_options();
  options.pad_frames = 0;

  CHECK(max_sustainable_growth(options, 0.5f, kFramesPerSecond) ==
        doctest::Approx(2.0f));
  CHECK(max_sustainable_growth(options, 0.25f, kFramesPerSecond) ==
        doctest::Approx(4.0f));
  // Never advises shrinking chunks, however slow the decoder is.
  CHECK(max_sustainable_growth(options, 4.0f, kFramesPerSecond) ==
        doctest::Approx(1.0f));
}

TEST_CASE("degenerate inputs do not produce chunks") {
  ChunkPolicyOptions options = default_options();
  const std::vector<int> none =
      plan_boundaries({}, 0, kFramesPerSecond, options);
  CHECK(plan_spans(none, 0, kSamplesPerFrame, kSampleRate, options).empty());
  CHECK(boundary_cost({}, {}, 0).empty());
}

TEST_CASE("an utterance shorter than the first chunk stays whole") {
  ChunkPolicyOptions options = default_options();
  const int frames = 10;  // 0.25 s, under the 0.5 s first chunk
  const std::vector<int> boundaries =
      plan_boundaries({}, frames, kFramesPerSecond, options);
  CHECK(boundaries.size() == 2);
  CHECK(chunk_lengths(boundaries).front() == frames);
}
