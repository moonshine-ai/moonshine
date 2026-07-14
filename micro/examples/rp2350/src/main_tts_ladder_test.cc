// Incremental bring-up ladder for the neural-TTS stack. Minimal app:
// heartbeat tick once per second, forever, plus exactly the components
// selected by TTS_LADDER_STAGE. A stage is "good" when ticks keep
// flowing; the first stage whose ticks stop (or that watchdog-reboots)
// names the guilty component.
//
//   stage 0: nothing -- banner + ticks only
//   stage 1: + WorldLiteSynth construction (kissfft heap plans) + one
//            forward/inverse FFT self-test per tick
//   stage 2: + PbDecoder construction (TFLM AllocateTensors)
//   stage 3: + one tile decode (one TFLM Invoke) per tick
//   stage 4: + one full utterance synthesis (PCM discarded)
//   stage 5: + PCM streamed over USB CDC
//
// Build: cmake -DTTS_LADDER_STAGE=N build && make moonshine_micro_tts_ladder_test

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "hardware/structs/watchdog.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"
#include "pico/time.h"

#ifndef TTS_LADDER_STAGE
#define TTS_LADDER_STAGE 0
#endif

// 0 = link the stage's code but never execute it (probes whether a boot
// failure is link-layout-level or runtime-level).
#ifndef TTS_LADDER_EXECUTE
#define TTS_LADDER_EXECUTE 1
#endif

#if TTS_LADDER_STAGE >= 1
#include "kiss_fftr.h"
#include "neural_tts/worldlite_synth.h"
#endif
#if TTS_LADDER_STAGE >= 2
#include "neural_tts/pb_decoder.h"
#include "neural_tts_demo_data.h"
#endif

// Crash forensics: hard faults stash PC/LR/CFSR in watchdog scratch and
// reboot; hangs trip the 8 s watchdog. Next boot reports which it was.
extern "C" void isr_hardfault_c(uint32_t* sp) {
  watchdog_hw->scratch[4] = 0xFA17FA17u;
  watchdog_hw->scratch[5] = sp[6];                             // PC
  watchdog_hw->scratch[6] = sp[5];                             // LR
  watchdog_hw->scratch[7] = *(volatile uint32_t*)0xE000ED28u;  // CFSR
  watchdog_reboot(0, 0, 10);
  while (true) {
  }
}

extern "C" __attribute__((naked)) void isr_hardfault(void) {
  __asm volatile(
      "movs r0, #0\n"
      "msr msplim, r0\n"
      "mrs r0, msp\n"
      "b isr_hardfault_c\n");
}

// Progress hook required by pb_decoder/worldlite_synth on PICO builds.
extern "C" void tts_checkpoint(uint32_t v) {
  watchdog_hw->scratch[3] = 0xC0DE0000u | (v & 0xFFFFu);
  watchdog_update();
}
extern "C" void tts_checkpoint2(uint32_t v) {
  watchdog_hw->scratch[2] = 0x53C00000u | (v & 0xFFFFu);
}
extern "C" void tts_trace(uint32_t, uint32_t) {}

namespace {

#define TRACE(...)       \
  do {                   \
    printf(__VA_ARGS__); \
    fflush(stdout);      \
    sleep_ms(10);        \
  } while (0)

#if TTS_LADDER_STAGE >= 2
constexpr size_t kArenaBytes = 300 * 1024;
alignas(16) uint8_t g_arena[kArenaBytes];
#endif

#if TTS_LADDER_STAGE >= 4
void EmitPcm(void* user, const int16_t* samples, int n) {
  int* count = static_cast<int*>(user);
#if TTS_LADDER_STAGE >= 5
  fwrite(samples, sizeof(int16_t), n, stdout);
  fflush(stdout);
#else
  (void)samples;
#endif
  *count += n;
}
#endif

}  // namespace

int main() {
  // Stack spill room below the 4 KiB SCRATCH_Y stack (see the TTS app).
  extern char __end__;
  __asm volatile("msr msplim, %0" ::"r"(&__end__ + PICO_HEAP_SIZE));

  stdio_init_all();
  const bool was_watchdog = watchdog_caused_reboot();
  watchdog_enable(8000, true);

  for (int i = 0; i < 5; ++i) {
    watchdog_update();
    TRACE("tts_ladder stage %d (boot %d)\n", TTS_LADDER_STAGE, i);
    if (watchdog_hw->scratch[4] == 0xFA17FA17u) {
      TRACE("!!! HARD FAULT pc=%08lx lr=%08lx cfsr=%08lx\n",
            watchdog_hw->scratch[5], watchdog_hw->scratch[6],
            watchdog_hw->scratch[7]);
    } else if (was_watchdog) {
      TRACE("!!! WATCHDOG (hang) ckpt=%08lx ckpt2=%08lx\n",
            watchdog_hw->scratch[3], watchdog_hw->scratch[2]);
    }
    sleep_ms(1000);
  }
  watchdog_hw->scratch[4] = 0;
  watchdog_hw->scratch[3] = 0;
  watchdog_hw->scratch[2] = 0;

#if TTS_LADDER_STAGE >= 1
  // Heap probe: malloc works (even a bogus 64 KiB "success" -- newlib's
  // sbrk clearly isn't enforcing PICO_HEAP_SIZE), but kiss_fftr_alloc in
  // the synth ctor returned NULL and the first banner-adjacent print
  // wedged. Probe kissfft's own alloc path directly, with sizes.
  {
    watchdog_update();
    size_t memneeded = 0;
    kiss_fftr_alloc(1024, 0, nullptr, &memneeded);
    TRACE("kiss_fftr_alloc(1024) wants %u bytes\n", (unsigned)memneeded);
    watchdog_update();
    kiss_fftr_cfg cfg_probe = kiss_fftr_alloc(1024, 0, nullptr, nullptr);
    TRACE("kiss_fftr_alloc(1024) heap = %p\n", (void*)cfg_probe);
    watchdog_update();
    if (cfg_probe) {
      static float tb[1024];
      static kiss_fft_cpx sp[513];
      for (int i = 0; i < 1024; ++i) tb[i] = (i == 3) ? 1.0f : 0.0f;
      TRACE("running kiss_fftr...\n");
      kiss_fftr(cfg_probe, tb, sp);
      TRACE("kiss_fftr done, sp[0]=(%f,%f)\n", sp[0].r, sp[0].i);
      kiss_fftr_free(cfg_probe);
    }
  }

  // Runtime switch (constant, but opaque to the optimizer via volatile):
  // with TTS_LADDER_EXECUTE=0 the binary layout is identical but the
  // synth is never constructed/used -- separates "the extra code/data
  // being LINKED breaks the boot" from "RUNNING it breaks the boot".
  volatile int execute_stage = TTS_LADDER_EXECUTE;
  neural_tts::WorldLiteSynth* synth = nullptr;
  if (execute_stage) {
    TRACE("init synth...\n");
    static neural_tts::WorldLiteSynth synth_obj;
    synth = &synth_obj;
    TRACE("synth plans fwd=%p inv=%p\n", synth->fwd_plan(),
          synth->inv_plan());
  }
#endif

#if TTS_LADDER_STAGE >= 2
  neural_tts::PbDecoder::Config cfg;
  cfg.model_data = g_pb_decoder_model;
  cfg.codebooks[0] = g_pb_codebook0;
  cfg.codebooks[1] = g_pb_codebook1;
  cfg.codebooks[2] = g_pb_codebook2;
  cfg.codebook_scales[0] = g_pb_codebook0_scale;
  cfg.codebook_scales[1] = g_pb_codebook1_scale;
  cfg.codebook_scales[2] = g_pb_codebook2_scale;
  cfg.n_stages = kPbStages;
  cfg.latent_dim = kPbLatentDim;
  cfg.tile_latents = kPbTileLatents;
  cfg.tile_hop = kPbTileHop;
  cfg.input_scale = kPbInputScale;
  cfg.output_scale = kPbOutputScale;
  TRACE("init decoder...\n");
  static neural_tts::PbDecoder decoder(cfg, g_arena, kArenaBytes);
  if (!decoder.ok()) {
    while (true) {
      watchdog_update();
      TRACE("FATAL: decoder init failed\n");
      sleep_ms(2000);
    }
  }
  TRACE("decoder ready, arena used %u\n",
        static_cast<unsigned>(decoder.arena_used_bytes()));
#endif

  for (unsigned tick = 0;; ++tick) {
    watchdog_update();

#if TTS_LADDER_STAGE >= 1
    // execute_stage: 1 = construct only, 2 = also run FFTs each tick
    const float fft_err =
        (synth && execute_stage >= 2) ? synth->FftSelfTest() : -1.0f;
#else
    const float fft_err = 0.0f;
#endif

#if TTS_LADDER_STAGE >= 3
    const PbDemoUtterance& utt = kPbUtterances[tick % kPbNumUtterances];
    neural_tts::PbCodedUtterance coded;
    coded.n_frames = utt.n_frames;
    coded.n_latents = utt.n_latents;
    coded.codes = utt.codes;
    coded.f0q = utt.f0q;
    decoder.BeginUtterance(&coded);
#endif

#if TTS_LADDER_STAGE >= 4
    int emitted = 0;
    const uint64_t t0 = time_us_64();
    synth.Synthesize(neural_tts::PbDecoder::GetFrameThunk, &decoder,
                     utt.n_frames, utt.gain, EmitPcm, &emitted);
    const uint32_t synth_ms =
        static_cast<uint32_t>((time_us_64() - t0) / 1000);
    TRACE("tick %u t=%lums fft_err=%.6f synth=%lums pcm=%d\n", tick,
          static_cast<unsigned long>(time_us_32() / 1000), fft_err,
          static_cast<unsigned long>(synth_ms), emitted);
#elif TTS_LADDER_STAGE >= 3
    neural_tts::WorldFrame frame;
    decoder.GetFrame(0, &frame);  // one tile decode = one TFLM Invoke
    TRACE("tick %u t=%lums fft_err=%.6f f0=%.1f benv0=%.5f\n", tick,
          static_cast<unsigned long>(time_us_32() / 1000), fft_err, frame.f0,
          frame.benv[0]);
#else
    TRACE("tick %u t=%lums fft_err=%.6f\n", tick,
          static_cast<unsigned long>(time_us_32() / 1000), fft_err);
#endif

    sleep_ms(1000);
  }
}
