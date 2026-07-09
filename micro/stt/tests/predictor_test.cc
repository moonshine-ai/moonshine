// Unit tests for the STT prediction helpers, using TFLM's micro_test.h.
//
// Argmax + stable-softmax are the post-processing the test harness and the live
// audio path use to turn logits into a labelled prediction. Runs on the host
// (logic only -- no interpreter).

#include <cmath>

#include "stt/stt.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ArgmaxPicksLargest) {
  const float logits[5] = {-1.0f, 0.5f, 3.2f, 3.1f, -4.0f};
  TF_LITE_MICRO_EXPECT_EQ(spelling::Argmax(logits, 5), 2);
}

TF_LITE_MICRO_TEST(ArgmaxTiesGoToLowestIndex) {
  const float logits[4] = {2.0f, 2.0f, 1.0f, 0.0f};
  TF_LITE_MICRO_EXPECT_EQ(spelling::Argmax(logits, 4), 0);
}

TF_LITE_MICRO_TEST(SoftmaxProbsSumToOne) {
  const float logits[3] = {1.0f, 2.0f, 3.0f};
  float sum = 0.0f;
  for (int i = 0; i < 3; ++i) sum += spelling::SoftmaxProb(logits, 3, i);
  TF_LITE_MICRO_EXPECT_NEAR(sum, 1.0f, 1e-5f);
}

TF_LITE_MICRO_TEST(SoftmaxProbMatchesHandComputed) {
  // Two equal logits -> 0.5 each.
  const float logits[2] = {5.0f, 5.0f};
  TF_LITE_MICRO_EXPECT_NEAR(spelling::SoftmaxProb(logits, 2, 0), 0.5f, 1e-5f);
}

TF_LITE_MICRO_TEST(SoftmaxStableOnLargeLogits) {
  // Subtracting the max keeps exp() from overflowing; the argmax should
  // dominate the probability mass.
  const float logits[3] = {1000.0f, 0.0f, -1000.0f};
  const float p = spelling::SoftmaxProb(logits, 3, 0);
  TF_LITE_MICRO_EXPECT_GT(p, 0.999f);
  TF_LITE_MICRO_EXPECT_LE(p, 1.0f);
}

TF_LITE_MICRO_TESTS_END
