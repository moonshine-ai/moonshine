// Step 5 of the minimal bring-up ladder: step 4 + the real WorldLiteSynth
// (constructor allocates two 1024-point kissfft plans from the heap;
// self-test runs a forward+inverse FFT pair per tick). This is where the
// wrong-board (pico2 on a Pico 2 W) builds reported NULL plans and then
// wedged. LED keeps blinking as the liveness indicator.

#include <cstdio>

#include "neural_tts/worldlite_synth.h"
#include "pico/stdlib.h"

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

}  // namespace

int main() {
  stdio_init_all();
  const bool led_ok = LedInit();

  for (int i = 0; i < 6; ++i) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
  }
  printf("step5 boot, led_ok=%d\n", led_ok ? 1 : 0);

  static neural_tts::WorldLiteSynth synth;
  printf("synth plans fwd=%p inv=%p ok=%d\n", synth.fwd_plan(),
         synth.inv_plan(), synth.ok() ? 1 : 0);

  for (unsigned tick = 0;; ++tick) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
    const float err = synth.ok() ? synth.FftSelfTest() : -1.0f;
    printf("step5 tick %u fft_self_test_err=%f\n", tick, err);
  }
}
