// I2S speaker output for the on-board echo service (MAX98357A / Adafruit 3006).
//
// Wiring (matches src/i2s_out.pio; LRC = BCLK + 1 for the sideset pair):
//   DIN -> GPIO10   BCLK <- GPIO11   LRC <- GPIO12   Vin -> VSYS   GND -> GND
//
// The MAX98357A has its own I2S DAC + reconstruction filter, so this backend
// just feeds int16 PCM straight to a PIO I2S transmitter -- no PWM carrier, no
// RC filter, and no output EQ. Each mono sample is written into BOTH I2S slots
// so the amp's default (L+R)/2 mix plays it at full level.
//
// A DMA-fed ring buffer decouples the (bursty) producer from the (steady) I2S
// consumer. The neural-TTS engine emits audio in bursts -- it goes silent for
// ~240 ms while it decodes a tile, then renders that tile out in a rush -- so
// feeding the 8-deep PIO FIFO directly with blocking writes starved it during
// every decode pause and chopped the speech into stuttering fragments (~36% of
// a sentence came out as gaps). Instead, Write() drops frames into a ~0.5 s
// ring that a free-running DMA drains into the PIO TX FIFO at exactly the sample
// rate (DREQ-paced, read-address auto-wrapped). Playback starts once the ring
// has pre-buffered enough to ride out a decode pause, and because synthesis is
// (just) faster than real-time the ring stays fed, so the audio plays smoothly.

#ifndef SPELLING_I2S_AUDIO_OUT_H_
#define SPELLING_I2S_AUDIO_OUT_H_

#include <cstdint>

#include "audio_io.h"
#include "hardware/pio.h"

namespace spelling {

// Peak-normalization gain so the loudest sample maps to `target` of full scale.
// Returns 1.0 for (near-)silent buffers, and is clamped to `max_gain` so a quiet
// clip isn't blown up into pure noise. Mono PCM clips captured from the I2S mic
// are often only a few percent of full scale -- normalizing first makes them use
// the whole output range.
float PeakNormalizeGain(const int16_t* samples, int n, float target = 0.9f,
                        float max_gain = 32.0f);
float PeakNormalizeGain(const float* samples, int n, float target = 0.9f,
                        float max_gain = 32.0f);

class I2sAudioOutput : public AudioOutput {
 public:
  // Defaults match the documented wiring: DIN=GP10, BCLK=GP11, LRC=GP12.
  explicit I2sAudioOutput(unsigned data_pin = 10, unsigned clock_base = 11,
                          int sample_rate = 16000);

  void Begin(int sample_rate, int num_samples,
             const char* kind = "AUDIO") override;
  void Write(const int16_t* samples, int n) override;
  void End() override;

  // Linear playback gain applied before I2S conversion. Use with
  // PeakNormalizeGain() to bring quiet clips up to a usable level. Reset to 1.0
  // between unrelated streams (it persists across Begin/Write/End).
  void SetGain(float gain) { gain_ = gain; }

 private:
  // (Re)configure the SM clock divider for the current sample rate.
  void ApplyClockDiv();
  // Push one packed stereo frame into the ring, blocking (spin) while the ring
  // is full so the DMA's real-time drain sets the pace.
  void PushFrame(uint32_t frame);
  // Kick the drain DMA (read from the ring base, DREQ-paced) once enough has
  // been pre-buffered.
  void StartDma();
  // DMA read cursor and buffered-frame count, as ring-word indices.
  unsigned ReadIdx() const;
  unsigned Used() const;

  PIO pio_;
  unsigned sm_;
  unsigned data_pin_;
  unsigned clock_base_;
  int sample_rate_ = 16000;
  float gain_ = 1.0f;

  int dma_chan_ = -1;       // drain DMA channel (ring -> PIO TX FIFO)
  unsigned wr_idx_ = 0;     // producer cursor into the ring (frames)
  bool dma_running_ = false;
};

}  // namespace spelling

#endif  // SPELLING_I2S_AUDIO_OUT_H_
