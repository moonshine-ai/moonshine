// Unit tests for the VAD segmenter, using TFLM's micro_test.h.
//
// These cases exercise segment boundaries (look-behind pre-roll,
// trailing-segment flush, no-look-behind) on fixed probability sequences.
//
// Runs on the host (logic only -- no interpreter).

#include <cstddef>
#include <vector>

#include "tensorflow/lite/micro/testing/micro_test.h"
#include "vad/vad.h"

namespace {

constexpr int kHop = 512;
constexpr std::size_t kLookBehind = 8192;
constexpr std::size_t kMaxSeg = 15 * 16000;

std::vector<float> Repeat(float v, int n) { return std::vector<float>(n, v); }

std::vector<float> Concat(std::initializer_list<std::vector<float>> parts) {
  std::vector<float> out;
  for (const auto& p : parts) out.insert(out.end(), p.begin(), p.end());
  return out;
}

// Run a probability sequence through the segmenter and collect the closed
// segments as (start, end) sample-index pairs.
std::vector<std::pair<std::size_t, std::size_t>> Segments(
    const std::vector<float>& probs, std::size_t look_behind) {
  spelling::VadSegmenter seg(0.5f, 16, kHop, look_behind, kMaxSeg);
  seg.Start();
  std::vector<std::pair<std::size_t, std::size_t>> segs;
  for (float p : probs) {
    if (seg.ProcessFrame(p) == spelling::VadEvent::kSpeechEnd) {
      segs.emplace_back(seg.segment_start_sample(), seg.segment_end_sample());
    }
  }
  if (seg.Finish() == spelling::VadEvent::kSpeechEnd) {
    segs.emplace_back(seg.segment_start_sample(), seg.segment_end_sample());
  }
  return segs;
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SingleSegmentDetected) {
  const auto segs =
      Segments(Concat({Repeat(0.0f, 20), Repeat(0.9f, 40), Repeat(0.0f, 40)}),
               kLookBehind);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<int>(segs.size()), 1);
  // The segment should start before its end and the start should reflect the
  // look-behind pre-roll (i.e. start < first voiced frame's sample index).
  TF_LITE_MICRO_EXPECT_LT(segs[0].first, segs[0].second);
}

TF_LITE_MICRO_TEST(TwoSegmentsDetected) {
  const auto segs =
      Segments(Concat({Repeat(0.0f, 10), Repeat(0.95f, 30), Repeat(0.0f, 30),
                       Repeat(0.85f, 25), Repeat(0.0f, 20)}),
               kLookBehind);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<int>(segs.size()), 2);
}

TF_LITE_MICRO_TEST(TrailingSegmentFlushed) {
  // Speech that never falls back to silence must still be closed by Finish().
  const auto segs =
      Segments(Concat({Repeat(0.0f, 15), Repeat(0.9f, 50)}), kLookBehind);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<int>(segs.size()), 1);
}

TF_LITE_MICRO_TEST(NoLookBehindStartsLater) {
  const auto with_lb =
      Segments(Concat({Repeat(0.0f, 20), Repeat(0.9f, 40), Repeat(0.0f, 40)}),
               kLookBehind);
  const auto no_lb = Segments(
      Concat({Repeat(0.0f, 20), Repeat(0.9f, 40), Repeat(0.0f, 40)}), 0);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<int>(no_lb.size()), 1);
  // Without the pre-roll the segment starts at or after the look-behind case.
  TF_LITE_MICRO_EXPECT_GE(no_lb[0].first, with_lb[0].first);
}

TF_LITE_MICRO_TEST(ExtractClipFrontAlignedZeroPads) {
  const float src[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  float out[6] = {-1, -1, -1, -1, -1, -1};
  spelling::ExtractClipFrontAligned(src, 8, /*start=*/2, /*end=*/5, out, 6);
  TF_LITE_MICRO_EXPECT_NEAR(out[0], 3.0f, 1e-6f);
  TF_LITE_MICRO_EXPECT_NEAR(out[2], 5.0f, 1e-6f);
  TF_LITE_MICRO_EXPECT_NEAR(out[3], 0.0f, 1e-6f);  // tail zero-padded
  TF_LITE_MICRO_EXPECT_NEAR(out[5], 0.0f, 1e-6f);
}

TF_LITE_MICRO_TESTS_END
