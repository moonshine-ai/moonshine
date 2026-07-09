// libFuzzer harness for the audio resampler.
//
// The input is split into two float sample rates followed by float PCM samples.
// Sample rates are clamped to a finite, realistic range so the fuzzer exercises
// the resampling arithmetic (interpolation, indexing, boundary handling) rather
// than merely triggering an out-of-memory from an absurd output size.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "resampler.h"

namespace {
float sane_rate(float rate) {
  if (!std::isfinite(rate)) {
    return 16000.0f;
  }
  rate = std::fabs(rate);
  if (rate < 1.0f) {
    return 1.0f;
  }
  if (rate > 384000.0f) {
    return 384000.0f;
  }
  return rate;
}
}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // Layout: [4 bytes input rate][4 bytes output rate][float samples...].
  if (size < 8) {
    return 0;
  }
  float input_rate = 0.0f;
  float output_rate = 0.0f;
  std::memcpy(&input_rate, data, sizeof(float));
  std::memcpy(&output_rate, data + 4, sizeof(float));
  input_rate = sane_rate(input_rate);
  output_rate = sane_rate(output_rate);

  const size_t num_samples = (size - 8) / sizeof(float);
  std::vector<float> audio(num_samples);
  if (num_samples > 0) {
    std::memcpy(audio.data(), data + 8, num_samples * sizeof(float));
  }

  resample_audio(audio, input_rate, output_rate);
  return 0;
}
