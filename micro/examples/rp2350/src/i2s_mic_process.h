// SPH0645 / Adafruit 3421 I2S sample conversion and DC removal.
//
// Matches arduino-pico I2S.read32(): each 32-bit FIFO word is a signed sample
// with the 24-bit mic payload in the upper bits (scale to int16 with >> 16).

#ifndef SPELLING_I2S_MIC_PROCESS_H_
#define SPELLING_I2S_MIC_PROCESS_H_

#include <cstdint>

namespace spelling {

// 24-bit payload left-justified in the 32-bit slot -> take the top 16 bits.
constexpr int kI2sSampleShift = 16;

int32_t I2sRawToInt32(uint32_t raw);
int16_t I2sRawToInt16(uint32_t raw);

// First-order DC blocker (~80 Hz default at 16 kHz).
class I2sDcBlocker {
 public:
  explicit I2sDcBlocker(int sample_rate_hz, float cutoff_hz = 80.0f);

  int16_t Process(int16_t x);
  void Reset();

 private:
  float r_;
  int16_t x_prev_ = 0;
  float y_prev_ = 0.f;
};

// Subtract the buffer mean (quick residual DC trim before host playback).
void I2sRemoveBufferDc(int16_t* samples, int n);

}  // namespace spelling

#endif  // SPELLING_I2S_MIC_PROCESS_H_
