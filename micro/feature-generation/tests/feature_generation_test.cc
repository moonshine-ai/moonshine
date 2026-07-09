// Unit tests for the feature-generation module, using TFLM's micro_test.h.
//
// Covers the shared building blocks (periodic Hann, Slaney mel scale) and the
// load-bearing invariant that MelStreamer (one FFT per non-overlapping block,
// ring buffer) produces the SAME normalised feature plane as the batch
// LogMelSpectrogram::Compute(center=false) over the same samples. That parity
// is what lets the streaming VAD front-end match the batch path the models expect.
//
// Runs on the host (logic only -- no interpreter): link against the module,
// kissfft, and the host TFLM stub.

#include "feature_generation/feature_generation.h"

#include <cmath>
#include <vector>

#include "kiss_fftr.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

// Build the CSR form of the Slaney filterbank from the dense matrix, matching
// what a flash-table generator emits.
void DenseToCsr(const std::vector<float>& dense, int n_mels, int n_freq,
                std::vector<int>* nz_off, std::vector<int>* nz_idx,
                std::vector<float>* nz_val) {
  nz_off->assign(n_mels + 1, 0);
  for (int m = 0; m < n_mels; ++m) {
    int cnt = 0;
    for (int k = 0; k < n_freq; ++k) {
      const float w = dense[static_cast<std::size_t>(m) * n_freq + k];
      if (w != 0.0f) {
        nz_idx->push_back(k);
        nz_val->push_back(w);
        ++cnt;
      }
    }
    (*nz_off)[m + 1] = (*nz_off)[m] + cnt;
  }
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(HannWindowPeriodicEndpoints) {
  const std::vector<float> w = spelling::HannWindowPeriodic(512);
  TF_LITE_MICRO_EXPECT_EQ(static_cast<int>(w.size()), 512);
  // Periodic Hann starts at 0 and is symmetric-ish about the centre.
  TF_LITE_MICRO_EXPECT_NEAR(w[0], 0.0f, 1e-6f);
  TF_LITE_MICRO_EXPECT_GT(w[256], 0.99f);  // near 1 at the centre
}

TF_LITE_MICRO_TEST(MelScaleRoundTrip) {
  for (float hz : {50.0f, 440.0f, 1000.0f, 4000.0f, 8000.0f}) {
    const float back = spelling::MelToHzSlaney(spelling::HzToMelSlaney(hz));
    TF_LITE_MICRO_EXPECT_NEAR(back, hz, hz * 1e-4f + 1e-3f);
  }
}

TF_LITE_MICRO_TEST(StreamerMatchesBatch) {
  const int n_mels = 32, window_frames = 24, n_fft = 512, hop = 512;
  const int sr = 16000;
  const float f_min = 20.0f, f_max = 8000.0f;
  const int n_freq = n_fft / 2 + 1;
  const int n_samples = window_frames * hop;

  // Deterministic pseudo-signal: a few tones plus a slow amplitude ramp so
  // different frames genuinely differ.
  std::vector<float> buf(n_samples);
  for (int i = 0; i < n_samples; ++i) {
    const double t = static_cast<double>(i) / sr;
    double v = 0.5 * std::sin(2 * M_PI * 220.0 * t) +
               0.3 * std::sin(2 * M_PI * 900.0 * t) +
               0.2 * std::sin(2 * M_PI * 3000.0 * t);
    v *= 0.2 + 0.8 * (static_cast<double>(i) / n_samples);
    buf[i] = static_cast<float>(v);
  }

  // Reference: batch LogMelSpectrogram, center=false, runtime-built tables.
  spelling::LogMelParams p{};
  p.sample_rate = sr;
  p.n_fft = n_fft;
  p.hop_length = hop;
  p.win_length = n_fft;
  p.n_mels = n_mels;
  p.f_min = f_min;
  p.f_max = f_max;
  p.target_frames = window_frames;
  p.eps = 1e-6f;
  p.center = false;
  spelling::LogMelSpectrogram ref(p);
  std::vector<float> out_ref(static_cast<std::size_t>(n_mels) * window_frames);
  ref.Compute(buf.data(), static_cast<std::size_t>(n_samples), out_ref.data());

  // Streamer: same Hann window + Slaney CSR filterbank, shared FFT state.
  std::vector<float> window = spelling::HannWindowPeriodic(n_fft);
  std::vector<float> dense =
      spelling::MakeMelFilterbank(n_freq, n_mels, sr, f_min, f_max);
  std::vector<int> nz_off, nz_idx;
  std::vector<float> nz_val;
  DenseToCsr(dense, n_mels, n_freq, &nz_off, &nz_idx, &nz_val);

  kiss_fftr_state* fft = kiss_fftr_alloc(n_fft, 0, nullptr, nullptr);
  spelling::MelStreamer streamer(n_mels, window_frames, n_fft, window.data(),
                                 nz_off.data(), nz_idx.data(), nz_val.data(),
                                 fft, 1e-6f);
  for (int k = 0; k < window_frames; ++k) {
    streamer.PushHop(&buf[static_cast<std::size_t>(k) * hop]);
  }
  std::vector<float> out_stream(static_cast<std::size_t>(n_mels) *
                                window_frames);
  streamer.BuildModelInput(out_stream.data());

  double max_abs = 0.0;
  for (std::size_t i = 0; i < out_ref.size(); ++i) {
    const double d = std::fabs(static_cast<double>(out_ref[i]) - out_stream[i]);
    if (d > max_abs) max_abs = d;
  }
  kiss_fftr_free(fft);

  // The two paths are the same maths in a different loop order; expect parity
  // to within float round-off.
  TF_LITE_MICRO_EXPECT_LT(max_abs, 1e-3);
}

TF_LITE_MICRO_TESTS_END
