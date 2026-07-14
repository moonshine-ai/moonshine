// Minimal USB CDC stdio soak test (the `moonshine_micro_usb_banner_test`
// target). Prints one numbered line per second, forever. Nothing else:
// no TFLM, no kissfft, no GPIO, no audio framing.
//
// Purpose: the neural-TTS firmware's USB CDC output dies a few seconds
// after boot (device stays enumerated but goes mute and stops servicing
// picotool's vendor reset). This target isolates whether that wedge lives
// in the app/library code or in the stdio-USB/build-config layer itself.
// A hard-fault trap + watchdog make any crash report itself on the next
// boot instead of wedging the board beyond picotool's reach.

#include <cstdio>

#include "hardware/structs/watchdog.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"

extern "C" void isr_hardfault(void) {
  uint32_t* sp;
  __asm volatile("mrs %0, msp" : "=r"(sp));
  watchdog_hw->scratch[4] = 0xFA17FA17u;
  watchdog_hw->scratch[5] = sp[6];  // stacked PC
  watchdog_hw->scratch[6] = sp[5];  // stacked LR
  watchdog_reboot(0, 0, 10);
  while (true) {
  }
}

int main() {
  stdio_init_all();
  const bool was_watchdog = watchdog_caused_reboot();
  watchdog_enable(8000, true);

  for (unsigned tick = 0;; ++tick) {
    watchdog_update();
    printf("banner_test tick %u\n", tick);
    if (watchdog_hw->scratch[4] == 0xFA17FA17u) {
      printf("!!! PREVIOUS BOOT HARD FAULT pc=%08lx lr=%08lx\n",
             watchdog_hw->scratch[5], watchdog_hw->scratch[6]);
    } else if (was_watchdog && tick < 10) {
      printf("!!! previous boot ended in WATCHDOG reset\n");
    }
    fflush(stdout);
    sleep_ms(1000);
  }
}
