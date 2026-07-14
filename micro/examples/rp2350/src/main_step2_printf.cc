// Step 2 of the minimal bring-up ladder: blink the LED AND print one
// line per second over USB CDC. Nothing else -- no watchdog, no TTS.
// The LED is the liveness indicator: if it keeps blinking but text stops
// arriving, the firmware is alive and the USB/stdio/host side is what
// broke. Works on both Pico 2 (GPIO LED) and Pico 2 W (CYW43 LED).

#include <cstdio>

#include "pico/stdlib.h"

#if defined(CYW43_WL_GPIO_LED_PIN)
#include "pico/cyw43_arch.h"
#endif

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

  for (unsigned tick = 0;; ++tick) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
    printf("step2 tick %u led_ok=%d\n", tick, led_ok ? 1 : 0);
  }
}
