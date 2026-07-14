// Step 7b of the minimal bring-up ladder: step 6 + sweep GetFrame through
// EVERY frame of each utterance (exercises the tile-advance path:
// DecodeTileAt at latent 24, 48, 72, ... including the final partial
// tile). No synthesis. Step 6 only ever decoded tile 0, so this is the
// first test of the decoder's window logic. LED blinks per utterance.

#include <cstdio>

#include "neural_tts/pb_decoder.h"
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
  printf("step7b boot, led_ok=%d\n", led_ok ? 1 : 0);

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

  for (int loop = 0;; ++loop) {
    for (int u = 0; u < kPbNumUtterances; ++u) {
      const PbDemoUtterance& utt = kPbUtterances[u];
      printf("step7b utt %d \"%s\" (%d frames)...\n", u, utt.text,
             utt.n_frames);

      neural_tts::PbCodedUtterance coded;
      coded.n_frames = utt.n_frames;
      coded.n_latents = utt.n_latents;
      coded.codes = utt.codes;
      coded.f0q = utt.f0q;
      decoder.BeginUtterance(&coded);

      // Same access pattern as WorldLiteSynth::Synthesize: frames in
      // order, plus the t+1 lookahead it uses for interpolation.
      float benv_sum = 0.0f;
      const uint64_t t0 = time_us_64();
      neural_tts::WorldFrame frame;
      for (int t = 0; t < utt.n_frames; ++t) {
        decoder.GetFrame(t, &frame);
        benv_sum += frame.benv[0];
        if (t + 1 < utt.n_frames) {
          decoder.GetFrame(t + 1, &frame);
        }
      }
      const uint64_t el = time_us_64() - t0;
      printf("step7b utt %d done: %.0fms, %d tiles, benv_sum=%.4f, "
             "stack_free=%lu\n",
             u, el / 1000.0f, decoder.tiles_decoded(), benv_sum,
             (unsigned long)StackFreeBytes());

      LedPut(true);
      sleep_ms(250);
      LedPut(false);
      sleep_ms(250);
    }
    printf("step7b loop %d done\n", loop);
  }
}
