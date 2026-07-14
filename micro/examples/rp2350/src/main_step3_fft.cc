// Step 3 of the minimal bring-up ladder: step 2 + heap probe + one
// kissfft plan + one 1024-point FFT per tick. In the earlier (wrong-
// board) builds kiss_fftr_alloc returned NULL and the first FFT wedged
// the core, so this is the first real suspect being reintroduced.
// LED keeps blinking as the liveness indicator.

#include <cstdio>
#include <cstdlib>

#include "kiss_fftr.h"
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

constexpr int kFftSize = 1024;
float g_time_buf[kFftSize];
kiss_fft_cpx g_spec[kFftSize / 2 + 1];

}  // namespace

int main() {
  stdio_init_all();
  const bool led_ok = LedInit();

  // Give the host a few seconds to open the port before the one-time
  // reports, LED blinking all the while.
  for (int i = 0; i < 6; ++i) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
  }
  printf("step3 boot, led_ok=%d\n", led_ok ? 1 : 0);

  // Heap probe: report what malloc actually returns. PICO_HEAP_SIZE for
  // this target is the default; we just want truthful numbers.
  for (size_t sz = 4 * 1024; sz <= 128 * 1024; sz *= 2) {
    void* p = malloc(sz);
    printf("  malloc(%6u) = %p\n", (unsigned)sz, p);
    free(p);
  }

  size_t memneeded = 0;
  kiss_fftr_alloc(kFftSize, 0, nullptr, &memneeded);
  printf("kiss_fftr_alloc(%d) wants %u bytes\n", kFftSize,
         (unsigned)memneeded);
  kiss_fftr_cfg fwd = kiss_fftr_alloc(kFftSize, 0, nullptr, nullptr);
  kiss_fftr_cfg inv = kiss_fftr_alloc(kFftSize, 1, nullptr, nullptr);
  printf("plans fwd=%p inv=%p\n", (void*)fwd, (void*)inv);

  for (unsigned tick = 0;; ++tick) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);

    float max_err = -1.0f;
    if (fwd && inv) {
      // impulse -> forward -> inverse == kFftSize * impulse
      for (int i = 0; i < kFftSize; ++i) g_time_buf[i] = 0.0f;
      g_time_buf[3] = 1.0f;
      kiss_fftr(fwd, g_time_buf, g_spec);
      kiss_fftri(inv, g_spec, g_time_buf);
      max_err = 0.0f;
      for (int i = 0; i < kFftSize; ++i) {
        const float want = (i == 3) ? (float)kFftSize : 0.0f;
        const float e = g_time_buf[i] - want;
        const float ae = e < 0 ? -e : e;
        if (ae > max_err) max_err = ae;
      }
    }
    printf("step3 tick %u fft_max_err=%f\n", tick, max_err);
  }
}
