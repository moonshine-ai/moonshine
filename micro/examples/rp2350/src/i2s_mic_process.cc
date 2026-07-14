#include "i2s_mic_process.h"

#include <algorithm>
#include <cmath>

namespace spelling {

int32_t I2sRawToInt32(uint32_t raw) {
  return static_cast<int32_t>(raw) >> kI2sSampleShift;
}

int16_t I2sRawToInt16(uint32_t raw) {
  const int32_t v = I2sRawToInt32(raw);
  return static_cast<int16_t>(
      std::clamp(v, static_cast<int32_t>(-32768), static_cast<int32_t>(32767)));
}

I2sDcBlocker::I2sDcBlocker(int sample_rate_hz, float cutoff_hz)
    : r_(std::exp(-2.f * 3.14159265f * cutoff_hz /
                   static_cast<float>(sample_rate_hz))) {}

int16_t I2sDcBlocker::Process(int16_t x) {
  const float y = static_cast<float>(x) - static_cast<float>(x_prev_) +
                  r_ * y_prev_;
  x_prev_ = x;
  y_prev_ = y;
  return static_cast<int16_t>(std::clamp(
      static_cast<int32_t>(std::lroundf(y)), static_cast<int32_t>(-32768),
      static_cast<int32_t>(32767)));
}

void I2sDcBlocker::Reset() {
  x_prev_ = 0;
  y_prev_ = 0.f;
}

void I2sRemoveBufferDc(int16_t* samples, int n) {
  if (n <= 0) return;
  int64_t sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += samples[i];
  }
  const int32_t mean = static_cast<int32_t>(sum / n);
  for (int i = 0; i < n; ++i) {
    samples[i] = static_cast<int16_t>(std::clamp(
        static_cast<int32_t>(samples[i]) - mean, static_cast<int32_t>(-32768),
        static_cast<int32_t>(32767)));
  }
}

}  // namespace spelling
