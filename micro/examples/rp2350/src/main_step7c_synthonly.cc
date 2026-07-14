// Step 7c of the minimal bring-up ladder: the mirror image of step 7b.
// Runs the FULL WorldLiteSynth::Synthesize loop (pulses, MinimumPhase,
// noise FFTs, overlap-add, FlushTo) but feeds it synthetic frames from a
// tiny generator instead of the TFLM decoder -- no PbDecoder, no arena,
// no Invoke. PCM is discarded. If this crashes, the bug is in the synth;
// if it survives, step 7's lockup comes from the synth+decoder combo.
//
// The synthetic frames alternate voiced (f0 sweeping 80..240 Hz, speechy
// tilted envelope) and unvoiced spans so both RenderPulse paths run.

#include <cmath>
#include <cstdio>

#include "neural_tts/worldlite_synth.h"
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

// 100-frame voiced spans alternating with 30-frame unvoiced spans.
void SynthFrame(void* /*user*/, int t, neural_tts::WorldFrame* f) {
  const int span = t % 130;
  const bool voiced = span < 100;
  f->f0 = voiced ? 80.0f + 160.0f * (span / 100.0f) : 0.0f;
  for (int i = 0; i < neural_tts::kWorldNumBenv; ++i) {
    // ~ -6 dB/octave tilt with a slow formant-ish wobble over time
    const float tilt = 0.05f * expf(-i * 0.06f);
    const float wobble =
        1.0f + 0.5f * sinf(0.13f * t + 0.4f * i);
    f->benv[i] = tilt * wobble;
  }
  for (int i = 0; i < neural_tts::kWorldNumBap; ++i) {
    f->bap[i] = voiced ? 0.1f + 0.05f * i : 0.95f;
  }
}

void DiscardPcm(void* user, const int16_t* samples, int n) {
  (void)samples;
  *static_cast<int*>(user) += n;
}

// Stack watermark: paint below the current SP down to the stack bottom,
// later scan for the deepest overwritten word. Free = untouched bytes.
extern "C" char __StackBottom;
constexpr uint32_t kStackPaint = 0xDEADBEEFu;

void PaintStack() {
  uint32_t msp;
  __asm volatile("mrs %0, msp" : "=r"(msp));
  uint32_t* lo = reinterpret_cast<uint32_t*>(&__StackBottom);
  uint32_t* hi = reinterpret_cast<uint32_t*>((msp - 64u) & ~3u);
  for (uint32_t* p = lo; p < hi; ++p) *p = kStackPaint;
}

uint32_t StackFreeBytes() {
  const uint32_t* p = reinterpret_cast<const uint32_t*>(&__StackBottom);
  uint32_t free_bytes = 0;
  while (*p++ == kStackPaint) free_bytes += 4;
  return free_bytes;
}

}  // namespace

int main() {
  PaintStack();
  stdio_init_all();
  const bool led_ok = LedInit();

  for (int i = 0; i < 6; ++i) {
    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
  }
  printf("step7c boot, led_ok=%d\n", led_ok ? 1 : 0);

  static neural_tts::WorldLiteSynth synth;
  printf("synth ok=%d selftest_err=%g\n", synth.ok() ? 1 : 0,
         (double)synth.FftSelfTest());

  constexpr int kFrames = 480;  // 2.4 s of audio, like the demo utterances
  for (int loop = 0;; ++loop) {
    int pcm_samples = 0;
    const uint64_t t0 = time_us_64();
    synth.Synthesize(SynthFrame, nullptr, kFrames, 0.5f, DiscardPcm,
                     &pcm_samples);
    const uint64_t el = time_us_64() - t0;
    printf("step7c loop %d: %d frames -> %d samples in %.0fms (rtf=%.2f) "
           "stack_free=%lu\n",
           loop, kFrames, pcm_samples, el / 1000.0f,
           (el / 1e6f) / (pcm_samples / 16000.0f),
           (unsigned long)StackFreeBytes());

    LedPut(true);
    sleep_ms(250);
    LedPut(false);
    sleep_ms(250);
  }
}
