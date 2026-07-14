// Step 7 of the minimal bring-up ladder: step 6 + full WORLD-lite
// synthesis of each demo utterance (decoder -> synth -> PCM), with the
// PCM discarded rather than streamed, printing per-utterance timing.
// LED blinks between utterances as the liveness indicator.

#include <cstdio>

#include "hardware/watchdog.h"
#include "neural_tts/pb_decoder.h"
#include "neural_tts/worldlite_synth.h"
#include "neural_tts_demo_data.h"
#include "pico/stdlib.h"
#include "pico/time.h"

#if defined(CYW43_WL_GPIO_LED_PIN)
#include "pico/cyw43_arch.h"
#endif

// Progress hooks called by neural_tts code on PICO builds. Here they feed
// the hardware watchdog and stash their argument in watchdog scratch
// registers (which survive a watchdog reboot), so a lockup auto-reboots
// after 8 s and the boot banner reports the last checkpoint reached
// instead of requiring a manual BOOTSEL recovery.
extern "C" void tts_checkpoint(uint32_t v) {
  watchdog_hw->scratch[4] = v;
  watchdog_update();
}
extern "C" void tts_checkpoint2(uint32_t v) { watchdog_hw->scratch[5] = v; }
extern "C" void tts_trace(uint32_t tag, uint32_t val) {
  watchdog_hw->scratch[6] = (tag << 24) | (val & 0xFFFFFFu);
}

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
  if (watchdog_caused_reboot()) {
    printf("step7 boot AFTER WATCHDOG: ckpt=%lu ckpt2=%lu trace=%08lx\n",
           (unsigned long)watchdog_hw->scratch[4],
           (unsigned long)watchdog_hw->scratch[5],
           (unsigned long)watchdog_hw->scratch[6]);
  }
  printf("step7 boot, led_ok=%d\n", led_ok ? 1 : 0);
  watchdog_hw->scratch[4] = 0;
  watchdog_hw->scratch[5] = 0;
  watchdog_hw->scratch[6] = 0;
  watchdog_enable(8000, true);  // pause_on_debug

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

  static neural_tts::WorldLiteSynth synth;
  static neural_tts::PbDecoder decoder(cfg, g_arena, kArenaBytes);
  printf("decoder ok=%d arena_used=%u synth ok=%d\n", decoder.ok() ? 1 : 0,
         (unsigned)decoder.arena_used_bytes(), synth.ok() ? 1 : 0);
  if (!synth.ok() || !decoder.ok()) {
    while (true) {
      printf("step7 INIT FAILED synth_ok=%d decoder_ok=%d, halting\n",
             synth.ok() ? 1 : 0, decoder.ok() ? 1 : 0);
      watchdog_update();
      LedPut(true);
      sleep_ms(400);
      LedPut(false);
      sleep_ms(400);
    }
  }

  for (int loop = 0;; ++loop) {
    for (int u = 0; u < kPbNumUtterances; ++u) {
      const PbDemoUtterance& utt = kPbUtterances[u];
      printf("step7 SAY %d \"%s\" (%d frames)\n", u, utt.text, utt.n_frames);

      neural_tts::PbCodedUtterance coded;
      coded.n_frames = utt.n_frames;
      coded.n_latents = utt.n_latents;
      coded.codes = utt.codes;
      coded.f0q = utt.f0q;
      decoder.BeginUtterance(&coded);

      int pcm_samples = 0;
      const uint64_t t0 = time_us_64();
      synth.Synthesize(neural_tts::PbDecoder::GetFrameThunk, &decoder,
                       utt.n_frames, utt.gain, DiscardPcm, &pcm_samples);
      const uint64_t el = time_us_64() - t0;

      const float audio_s = pcm_samples / 16000.0f;
      printf("step7 STATS utt=%d audio=%.2fs total=%.0fms net=%.0fms "
             "(%d tiles) rtf=%.2f stack_free=%lu\n",
             u, audio_s, el / 1000.0f, decoder.decode_us() / 1000.0f,
             decoder.tiles_decoded(), (el / 1e6f) / audio_s,
             (unsigned long)StackFreeBytes());

      LedPut(true);
      sleep_ms(250);
      LedPut(false);
      sleep_ms(250);
    }
    printf("step7 loop %d done\n", loop);
  }
}
