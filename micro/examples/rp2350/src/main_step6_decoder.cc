// Step 6 of the minimal bring-up ladder: step 5 + the TFLM decoder.
// Constructs PbDecoder (AllocateTensors) and runs one real tile decode
// (one s16x8 TFLM Invoke on utterance 0's latents) per tick, printing
// the per-tile time. LED keeps blinking as the liveness indicator.

#include <cstdio>

#include "hardware/clocks.h"
#include "hardware/structs/qmi.h"
#include "hardware/vreg.h"
#include "neural_tts/pb_decoder.h"
#include "neural_tts/worldlite_synth.h"
#include "neural_tts_demo_data.h"
#include "pico/stdlib.h"
#include "pico/time.h"

#if defined(CYW43_WL_GPIO_LED_PIN)
#include "pico/cyw43_arch.h"
#endif

// Progress hooks required by neural_tts code on PICO builds; no-ops here.
extern "C" void tts_checkpoint(uint32_t) {}
extern "C" void tts_checkpoint2(uint32_t) {}
extern "C" void tts_trace(uint32_t, uint32_t) {}

namespace {

bool LedInit() {
#if defined(CYW43_WL_GPIO_LED_PIN)
  return cyw43_arch_init() == 0;
#else
  gpio_init(PICO_DEFAULT_LED_PIN);
  gpio_set_dir(PICO_DEFAULT_LED_PIN, GPIO_OUT);
  return true;
#endif
}

void LedPut(bool on) {
#if defined(CYW43_WL_GPIO_LED_PIN)
  cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, on);
#else
  gpio_put(PICO_DEFAULT_LED_PIN, on);
#endif
}

constexpr size_t kArenaBytes = 300 * 1024;
alignas(16) uint8_t g_arena[kArenaBytes];

}  // namespace

int main() {
  // Overclock experiment: 150 -> 300 MHz. Needs a core voltage bump, and
  // the QSPI flash divider raised first so XIP stays within the flash
  // chip's 133 MHz rating (div 3 @ 300 MHz = 100 MHz flash clock).
#ifndef STEP6_SYS_KHZ
#define STEP6_SYS_KHZ 300000
#endif
#ifndef STEP6_FLASH_DIV
#define STEP6_FLASH_DIV 3
#endif
#if STEP6_SYS_KHZ > 150000
  vreg_set_voltage(VREG_VOLTAGE_1_30);
  sleep_ms(10);
  hw_write_masked(&qmi_hw->m[0].timing,
                  STEP6_FLASH_DIV << QMI_M0_TIMING_CLKDIV_LSB,
                  QMI_M0_TIMING_CLKDIV_BITS);
  set_sys_clock_khz(STEP6_SYS_KHZ, true);
#endif

  stdio_init_all();
  const bool led_ok = LedInit();

  for (int i = 0; i < 6; ++i) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
  }
  printf("step6 boot, led_ok=%d sysclk=%lukHz\n", led_ok ? 1 : 0,
         (unsigned long)(clock_get_hz(clk_sys) / 1000));

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

  static neural_tts::PbDecoder decoder(cfg, g_arena, kArenaBytes);
  printf("decoder ok=%d arena_used=%u\n", decoder.ok() ? 1 : 0,
         (unsigned)decoder.arena_used_bytes());
  if (!decoder.ok()) {
    while (true) {
      LedPut(true);
      sleep_ms(100);
      LedPut(false);
      sleep_ms(100);
    }
  }

  const PbDemoUtterance& utt = kPbUtterances[0];
  neural_tts::PbCodedUtterance coded;
  coded.n_frames = utt.n_frames;
  coded.n_latents = utt.n_latents;
  coded.codes = utt.codes;
  coded.f0q = utt.f0q;

  for (unsigned tick = 0;; ++tick) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);

    decoder.BeginUtterance(&coded);  // resets the decoded-window cache
    neural_tts::WorldFrame frame;
    const uint64_t t0 = time_us_64();
    decoder.GetFrame(0, &frame);  // one tile decode = one TFLM Invoke
    const uint32_t ms = (uint32_t)((time_us_64() - t0) / 1000);
    printf("step6 tick %u invoke=%lums f0=%.1f benv0=%.5f\n", tick,
           (unsigned long)ms, frame.f0, frame.benv[0]);
  }
}
