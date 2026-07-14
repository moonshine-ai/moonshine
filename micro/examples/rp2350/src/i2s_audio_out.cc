#include "i2s_audio_out.h"

#include <cmath>
#include <cstdint>
#include <cstring>

#include "hardware/dma.h"
#include "hardware/pio.h"
#include "i2s_out.pio.h"
#include "pico/stdlib.h"

namespace spelling {

namespace {

// Pack one mono int16 sample into a 32-bit stereo I2S frame: MSB-first shift
// means [31:16] is the ws=0 slot and [15:0] the ws=1 slot. Duplicate into both
// so the MAX98357A's default (L+R)/2 downmix reproduces the sample at full
// level (writing only one slot would be 6 dB quieter).
inline uint32_t StereoFrame(int16_t s) {
  const uint32_t u = static_cast<uint32_t>(static_cast<uint16_t>(s));
  return (u << 16) | u;
}

// Drain ring: the DMA streams packed stereo frames from here into the PIO TX
// FIFO, DREQ-paced, with the read address hardware-wrapped. Power-of-two byte
// size, aligned to it (DMA read-ring requirement). 8192 frames * 4 B = 32768 B
// (1<<15) == 0.512 s at 16 kHz -- the DMA read-ring size field caps a single
// channel at 2^15 bytes, and 0.5 s comfortably rides out the ~240 ms neural-TTS
// per-tile decode pause. Static (not a member) so it can be 32 KiB-aligned even
// when the I2sAudioOutput itself lives on the stack; there is only ever one.
constexpr unsigned kOutRingFrames = 8192;
constexpr unsigned kOutRingBytes = kOutRingFrames * 4u;  // 32768 == 1<<15
constexpr unsigned kOutRingBits = 15;                    // log2(kOutRingBytes)
// Pre-buffer nearly the whole ring before starting playback so the first decode
// pause after audio starts has a full ~0.45 s cushion to drain from.
constexpr unsigned kOutPrebufferFrames = kOutRingFrames - 1024;  // ~448 ms

alignas(kOutRingBytes) uint32_t g_out_ring[kOutRingFrames];

}  // namespace

float PeakNormalizeGain(const int16_t* samples, int n, float target,
                        float max_gain) {
  if (n <= 0) return 1.0f;
  int32_t peak = 0;
  for (int i = 0; i < n; ++i) {
    const int32_t a = samples[i] < 0 ? -static_cast<int32_t>(samples[i])
                                     : static_cast<int32_t>(samples[i]);
    if (a > peak) peak = a;
  }
  if (peak < 16) return 1.0f;  // ~ -66 dBFS: treat as silence, don't amplify.
  float gain = target * 32767.0f / static_cast<float>(peak);
  if (gain > max_gain) gain = max_gain;
  return gain;
}

float PeakNormalizeGain(const float* samples, int n, float target,
                        float max_gain) {
  if (n <= 0) return 1.0f;
  float peak = 0.0f;
  for (int i = 0; i < n; ++i) {
    const float a = std::fabs(samples[i]);
    if (a > peak) peak = a;
  }
  if (peak < 5e-4f) return 1.0f;  // (near-)silent: don't amplify noise.
  float gain = target / peak;
  if (gain > max_gain) gain = max_gain;
  return gain;
}

I2sAudioOutput::I2sAudioOutput(unsigned data_pin, unsigned clock_base,
                               int sample_rate)
    : pio_(pio0),
      sm_(0),
      data_pin_(data_pin),
      clock_base_(clock_base),
      sample_rate_(sample_rate > 0 ? sample_rate : 16000) {
  // Coexist with the I2S mic (also on pio0, SM0): claim a free SM and add the
  // TX program to the same PIO's instruction memory. The mic claims SM0 in
  // i2s_audio_io.cc, so this hands back a different SM in the combined apps.
  sm_ = pio_claim_unused_sm(pio_, true);
  const uint offset = pio_add_program(pio_, &i2s_out_program);
  i2s_out_program_init(pio_, sm_, offset, data_pin_, clock_base_,
                       static_cast<unsigned>(sample_rate_));

  // Drain DMA: ring -> PIO TX FIFO, DREQ-paced, read address auto-wrapped. Set
  // up once here; Begin()/StartDma() reset the read cursor and (re)start it per
  // utterance. read_addr steps through the ring, write_addr is the fixed FIFO.
  dma_chan_ = dma_claim_unused_channel(true);
  dma_channel_config c = dma_channel_get_default_config(dma_chan_);
  channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
  channel_config_set_read_increment(&c, true);            // step through ring
  channel_config_set_write_increment(&c, false);          // TX FIFO fixed addr
  channel_config_set_ring(&c, /*write=*/false, kOutRingBits);  // wrap read addr
  channel_config_set_dreq(&c, pio_get_dreq(pio_, sm_, /*is_tx=*/true));
  dma_channel_configure(dma_chan_, &c, &pio_->txf[sm_], g_out_ring,
                        0xFFFFFFFFu, /*trigger=*/false);
}

unsigned I2sAudioOutput::ReadIdx() const {
  const uintptr_t base = reinterpret_cast<uintptr_t>(g_out_ring);
  const uintptr_t rp = dma_channel_hw_addr(dma_chan_)->read_addr;
  return static_cast<unsigned>(((rp - base) >> 2) & (kOutRingFrames - 1));
}

unsigned I2sAudioOutput::Used() const {
  return (wr_idx_ - ReadIdx()) & (kOutRingFrames - 1);
}

void I2sAudioOutput::StartDma() {
  dma_channel_set_read_addr(dma_chan_, g_out_ring, false);
  dma_channel_set_trans_count(dma_chan_, 0xFFFFFFFFu, false);
  dma_channel_start(dma_chan_);
  dma_running_ = true;
}

void I2sAudioOutput::PushFrame(uint32_t frame) {
  // Back-pressure: block while the ring is full so the DMA's real-time drain
  // paces us. Leave one slot free so full/empty stay distinguishable.
  if (dma_running_) {
    while (Used() >= kOutRingFrames - 1) tight_loop_contents();
  }
  g_out_ring[wr_idx_] = frame;
  wr_idx_ = (wr_idx_ + 1) & (kOutRingFrames - 1);
  // Start playback once enough is buffered to ride out a decode pause. Short
  // clips that never reach the threshold get started from End().
  if (!dma_running_ && Used() >= kOutPrebufferFrames) StartDma();
}

void I2sAudioOutput::ApplyClockDiv() {
  // Stop, reprogram the divider, and restart cleanly so a rate change doesn't
  // glitch a partially-shifted frame.
  pio_sm_set_enabled(pio_, sm_, false);
  pio_sm_set_clkdiv(pio_, sm_, i2s_out_clkdiv(static_cast<unsigned>(sample_rate_)));
  pio_sm_clkdiv_restart(pio_, sm_);
  pio_sm_set_enabled(pio_, sm_, true);
}

void I2sAudioOutput::Begin(int sample_rate, int num_samples, const char* kind) {
  (void)kind;
  (void)num_samples;
  const int rate = sample_rate > 0 ? sample_rate : 16000;
  if (rate != sample_rate_) {
    sample_rate_ = rate;
    ApplyClockDiv();
  }
  // Fresh utterance: stop any prior drain, clear the ring to silence (so a
  // pre-start or slight over-read plays 0, not stale audio), and rewind both
  // cursors to the ring base.
  if (dma_running_) dma_channel_abort(dma_chan_);
  dma_running_ = false;
  std::memset(g_out_ring, 0, sizeof(g_out_ring));
  wr_idx_ = 0;
  dma_channel_set_read_addr(dma_chan_, g_out_ring, false);
}

void I2sAudioOutput::Write(const int16_t* samples, int n) {
  for (int i = 0; i < n; ++i) {
    int32_t v = static_cast<int32_t>(std::lrintf(samples[i] * gain_));
    if (v > 32767) v = 32767;
    if (v < -32768) v = -32768;
    PushFrame(StereoFrame(static_cast<int16_t>(v)));
  }
}

void I2sAudioOutput::End() {
  // Pad a little silence so the DMA settles the line at 0 (and any 1-frame
  // over-read past the tail is silent), then make sure playback actually runs
  // for clips too short to have hit the pre-buffer threshold.
  for (int i = 0; i < 16; ++i) PushFrame(StereoFrame(0));
  if (!dma_running_) StartDma();
  // Wait for the DMA to drain everything we wrote, so the tail isn't cut off.
  while (Used() > 0) tight_loop_contents();
  dma_channel_abort(dma_chan_);
  dma_running_ = false;
  dma_channel_set_read_addr(dma_chan_, g_out_ring, false);
  wr_idx_ = 0;
}

}  // namespace spelling
