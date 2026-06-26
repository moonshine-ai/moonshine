#include "app_common.h"

#include <cstdint>
#include <cstdio>

#include "classes.h"  // kNumClasses
#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "mel_tables.h"  // kMelTable* (consistency asserts)
#include "model_data.h"  // g_spelling_model_data_size
#include "pico/stdlib.h"

namespace spelling {

// The window + Slaney mel filterbank baked into flash MUST match the audio
// config the rest of the firmware uses; both are emitted from the same
// spelling_cnn_meta.json, so a mismatch only happens on a partial regenerate.
static_assert(kMelTableNMels == kNMels,
              "mel_tables.h n_mels != audio_config.h kNMels; regenerate");
static_assert(kMelTableNFft == kNFft,
              "mel_tables.h n_fft != audio_config.h kNFft; regenerate");
static_assert(
    kMelTableWinLength == kWinLength,
    "mel_tables.h win_length != audio_config.h kWinLength; regenerate");
static_assert(
    kMelTableSampleRate == kSampleRate,
    "mel_tables.h sample_rate != audio_config.h kSampleRate; regenerate");

// 16-byte-aligned so TFLM's planner has room for SIMD-aligned working buffers.
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];
alignas(16) float g_waveform[kClipNumSamples];

void LedPulse(unsigned pin, int count, int on_ms, int off_ms) {
  for (int i = 0; i < count; ++i) {
    gpio_put(pin, 1);
    sleep_ms(on_ms);
    gpio_put(pin, 0);
    sleep_ms(off_ms);
  }
}

unsigned BoardInit() {
  // Overclock to 250 MHz BEFORE stdio so clk_peri is reconfigured in one shot.
  // Inference is compute-bound on the M33 SIMD core, so latency scales
  // ~linearly with clk_sys. Raise the core voltage to 1.20 V first (1.10 V
  // won't sustain 250 MHz). USB's 48 MHz reference comes from pll_usb,
  // untouched.
  vreg_set_voltage(VREG_VOLTAGE_1_20);
  sleep_ms(10);  // let the regulator ramp before clocking faster
  set_sys_clock_khz(250000, true);

  // Initialize the LED immediately: 5 fast POST pulses prove we got past crt0 +
  // global ctors. PICO_DEFAULT_LED_PIN is undefined on Pico 2 W (LED is on the
  // cyw43 chip); fall back to GPIO25 (harmless if unwired).
#ifdef PICO_DEFAULT_LED_PIN
  const unsigned led_pin = PICO_DEFAULT_LED_PIN;
#else
  const unsigned led_pin = 25u;
#endif
  gpio_init(led_pin);
  gpio_set_dir(led_pin, GPIO_OUT);
  LedPulse(led_pin, 5, 80, 80);  // POST

  // Bring up USB stdio and wait up to ~30 s for the host to open the CDC port
  // (so a serial monitor attached after enumeration still catches the banner).
  // Slow blink during the wait distinguishes "waiting for monitor" from "hung".
  stdio_init_all();
  for (int i = 0; i < 1500 && !stdio_usb_connected(); ++i) {
    gpio_put(led_pin, (i & 0x10) ? 1 : 0);  // ~250 ms half-period
    sleep_ms(20);
  }
  gpio_put(led_pin, 0);
  sleep_ms(200);  // let the host's terminal finish its post-open setup

  // Do NOT call tflite::InitializeTarget(): on pico-tflmicro it re-invokes
  // stdio_init_all(), tearing down the live CDC enumeration (macOS then
  // re-mounts under a possibly-new /dev/cu.usbmodem*, stranding any monitor).
  printf("\n[boot] stdio up, host connected=%d\n",
         stdio_usb_connected() ? 1 : 0);
  fflush(stdout);
  return led_pin;
}

void PrintBootBanner() {
  // Print one line, flush, then breathe ~10 ms: back-to-back printfs can fill
  // TinyUSB's 256-byte CDC tx FIFO faster than the host polls it, which makes
  // stdio_usb_out_chars block on its 1 s timeout and looks like a hang.
  auto puts_drain = [](const char* s) {
    fputs(s, stdout);
    fflush(stdout);
    sleep_ms(10);
  };

  puts_drain("=== SpellingCNN tiny ===\n");

  char line[96];
  std::snprintf(line, sizeof(line), "Clock:   %lu MHz (clk_sys)\n",
                static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000000u));
  puts_drain(line);

#if defined(SPELLING_TINY_MULTICORE)
  puts_drain("Cores:   2 (dual-core SIMD GEMM + depthwise split enabled)\n");
#else
  puts_drain("Cores:   1 (single-core SIMD)\n");
#endif

  std::snprintf(line, sizeof(line),
                "Model:   %u bytes (int8 mel-mode classifier)\n",
                g_spelling_model_data_size);
  puts_drain(line);

  std::snprintf(line, sizeof(line), "Classes: %d\n", kNumClasses);
  puts_drain(line);

  std::snprintf(line, sizeof(line),
                "Mel:     n_mels=%d  target_frames=%d  hop=%d  n_fft=%d\n",
                kNMels, kTargetFrames, kHopLength, kNFft);
  puts_drain(line);

  std::snprintf(line, sizeof(line), "Arena:   %u bytes static\n",
                static_cast<unsigned>(kTensorArenaSize));
  puts_drain(line);
}

}  // namespace spelling
