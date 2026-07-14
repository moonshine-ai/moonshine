// I2S microphone input for the on-board echo service (SPH0645 / Adafruit 3421).
//
// Wiring (arduino-pico I2SInput.ino convention):
//   DOUT -> GPIO0   BCLK <- GPIO1   LRCL <- GPIO2   SEL -> GND (left channel)
//
// Capture is DMA-backed: a single DMA channel streams the PIO RX FIFO into a
// wrap-around ring buffer continuously, so audio is never dropped while the CPU
// is busy between hops (FFT + VAD inference). ReadHop just copies the next hop
// of already-captured samples out of the ring; it does not gate the PIO. See
// i2s_audio_io.cc for the ring sizing and L/R parity details.

#ifndef SPELLING_I2S_AUDIO_IO_H_
#define SPELLING_I2S_AUDIO_IO_H_

#include "audio_io.h"
#include "i2s_mic_process.h"

namespace spelling {

class I2sAudioInput : public AudioInput {
 public:
  // Starts the PIO I2S master receiver at `sample_rate` Hz (16000 for VAD).
  explicit I2sAudioInput(int sample_rate = 16000);

  bool ReadHop(int16_t* out, int n) override;
  void Drain() override;

 private:
  unsigned sm_;
  I2sDcBlocker dc_blocker_;
};

}  // namespace spelling

#endif  // SPELLING_I2S_AUDIO_IO_H_
