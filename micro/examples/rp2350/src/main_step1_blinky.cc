// Step 1 of the minimal bring-up ladder: blink the LED, nothing else.
// No USB, no stdio, no watchdog, no TTS code. If this blinks, the board,
// power, flash process, and toolchain are good.
//
// NOTE: on a plain Pico 2 the LED is GPIO 25. On a Pico 2 W GPIO 25 is
// the CYW43 radio chip-select and the LED is on the radio chip, so this
// will NOT blink there (that itself is useful information).

#include "pico/stdlib.h"

int main() {
  const uint kLed = PICO_DEFAULT_LED_PIN;  // 25 on pico2
  gpio_init(kLed);
  gpio_set_dir(kLed, GPIO_OUT);
  while (true) {
    gpio_put(kLed, 1);
    sleep_ms(250);
    gpio_put(kLed, 0);
    sleep_ms(250);
  }
}
