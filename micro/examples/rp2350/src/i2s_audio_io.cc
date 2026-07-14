#include "i2s_audio_io.h"

#include <cstdint>

#include "hardware/dma.h"
#include "hardware/pio.h"
#include "i2s_mic.pio.h"
#include "i2s_mic_process.h"
#include "pico/stdlib.h"

namespace spelling {

namespace {

constexpr unsigned kDataPin = 0;
constexpr unsigned kClockBase = 1;
constexpr unsigned kBitsPerSample = 32;

PIO g_pio = pio0;
unsigned g_sm = 0;
bool g_pio_ready = false;

// Continuous DMA capture ring. The PIO autopushes one 32-bit word per channel,
// interleaved L,R,L,R,..., and the DMA streams them into this ring forever
// (DREQ-paced, write address auto-wrapped). This decouples capture from compute:
// while the CPU runs the FFT + VAD inference between hops, the DMA keeps filling
// the ring, so NO samples are dropped. (The old blocking pio_sm_get loop dropped
// every sample produced during compute, because the 8-deep RX FIFO overflowed --
// that gappy, discontinuity-laden audio is what flat-lined the VAD.)
//
// 2048 words = 1024 L/R frames = 64 ms at 16 kHz: comfortably covers one 32 ms
// hop plus the ~10-15 ms per-hop compute gap, so a normal hop never overruns,
// while staying small enough to fit alongside the 393 KB classifier arena. Must
// be a power-of-two byte size and aligned to it (DMA write-ring requirement).
constexpr unsigned kRingWords = 2048;
constexpr unsigned kRingBytes = kRingWords * 4;  // 8192 == 1<<13
constexpr unsigned kRingBits = 13;               // log2(kRingBytes)

alignas(kRingBytes) uint32_t g_capture_ring[kRingWords];

int g_dma_chan = -1;
// Read cursor into the ring, in words. Even-aligned so it always points at a
// LEFT word (the SPH0645 transmits on the left slot; SEL->GND). Advancing by 2
// per mono sample (skip the right word) preserves that parity across wraps,
// since kRingWords is even.
unsigned g_rd_idx = 0;

// Current DMA write position as a ring word index (0..kRingWords-1).
inline unsigned CaptureWriteIdx() {
  const uintptr_t base = reinterpret_cast<uintptr_t>(g_capture_ring);
  const uintptr_t wp = dma_channel_hw_addr(g_dma_chan)->write_addr;
  return static_cast<unsigned>(((wp - base) >> 2) & (kRingWords - 1));
}

void StartCaptureDma() {
  g_dma_chan = dma_claim_unused_channel(true);
  dma_channel_config c = dma_channel_get_default_config(g_dma_chan);
  channel_config_set_transfer_data_size(&c, DMA_SIZE_32);
  channel_config_set_read_increment(&c, false);   // RX FIFO is a fixed address
  channel_config_set_write_increment(&c, true);    // step through the ring
  channel_config_set_ring(&c, /*write=*/true, kRingBits);  // wrap the write addr
  channel_config_set_dreq(&c, pio_get_dreq(g_pio, g_sm, /*is_tx=*/false));

  // Huge transfer count: with the write-ring wrap this streams continuously into
  // the ring. 0xFFFFFFFF words at 32 kwords/s (16 kHz * 2 channels) is ~37 h of
  // uninterrupted capture -- effectively forever for a live echo/bring-up app.
  dma_channel_configure(g_dma_chan, &c,
                        g_capture_ring,        // write: ring base
                        &g_pio->rxf[g_sm],     // read: PIO RX FIFO
                        0xFFFFFFFFu,           // count
                        true);                 // start now
}

}  // namespace

I2sAudioInput::I2sAudioInput(int sample_rate)
    : sm_(0), dc_blocker_(sample_rate) {
  if (!g_pio_ready) {
    sm_ = 0;
    g_sm = sm_;
    // Mark SM0 as taken so a co-resident I2S *output* (I2sAudioOutput, which
    // pio_claim_unused_sm()s a slot on this same PIO) can't be handed SM0.
    pio_sm_claim(g_pio, sm_);
    const uint offset = pio_add_program(g_pio, &i2s_mic_in_program);
    i2s_mic_in_program_init(g_pio, sm_, offset, kDataPin, kClockBase,
                            kBitsPerSample,
                            static_cast<unsigned>(sample_rate));
    StartCaptureDma();
    g_rd_idx = CaptureWriteIdx() & ~1u;  // start reading from "now" (even)
    g_pio_ready = true;
  } else {
    sm_ = g_sm;
  }
}

bool I2sAudioInput::ReadHop(int16_t* out, int n) {
  const unsigned need = static_cast<unsigned>(2 * n);  // L+R words per sample
  // Block (real-time pacing) until a full hop has been captured by the DMA. The
  // ring is far deeper than one hop, so this just waits out the ~32 ms of audio,
  // it does not busy-burn for long.
  for (;;) {
    const unsigned avail = (CaptureWriteIdx() - g_rd_idx) & (kRingWords - 1);
    if (avail >= need) break;
    tight_loop_contents();
  }
  for (int i = 0; i < n; ++i) {
    const uint32_t left = g_capture_ring[g_rd_idx];  // even idx == left slot
    g_rd_idx = (g_rd_idx + 2) & (kRingWords - 1);     // skip the right word
    out[i] = dc_blocker_.Process(I2sRawToInt16(left));
  }
  return true;
}

void I2sAudioInput::Drain() {
  // Discard everything buffered so far: jump the read cursor to the current DMA
  // write position (kept even for L/R parity). Used while we play our own audio
  // so the mic backlog can't feed the VAD or back up the ring.
  g_rd_idx = CaptureWriteIdx() & ~1u;
}

}  // namespace spelling
