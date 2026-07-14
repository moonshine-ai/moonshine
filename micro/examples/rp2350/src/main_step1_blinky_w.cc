// Step 1b of the minimal bring-up ladder: blink the LED on a Pico 2 W.
// On the W the LED is wired to the CYW43 radio chip, not GPIO 25, so it
// needs the cyw43 driver. If step1 (GPIO 25) stays dark but this blinks,
// the board is a Pico 2 W and every firmware so far was built for the
// wrong board variant.

#include "pico/cyw43_arch.h"
#include "pico/stdlib.h"

int main() {
  if (cyw43_arch_init() != 0) {
    // Can't report anything without the LED; just park.
    while (true) {
      sleep_ms(1000);
    }
  }
  while (true) {
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 1);
    sleep_ms(250);
    cyw43_arch_gpio_put(CYW43_WL_GPIO_LED_PIN, 0);
    sleep_ms(250);
  }
}
