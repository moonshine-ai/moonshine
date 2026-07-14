// Host harness for WorldLiteSynth: reads raw [T,61] float32 WORLD-lite
// controls (f0, benv[48], bap[12]) from stdin, writes raw int16 16 kHz PCM
// to stdout. argv[1] = optional gain (default 1.0).
//
// Used by scripts/test_worldlite_c.py to validate the float32/kissfft port
// against pyworld before it runs on the RP2350.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "neural_tts/worldlite_synth.h"

int main(int argc, char** argv) {
  const float gain = argc > 1 ? strtof(argv[1], nullptr) : 1.0f;

  std::vector<float> raw;
  float buf[4096];
  size_t n;
  while ((n = fread(buf, sizeof(float), 4096, stdin)) > 0)
    raw.insert(raw.end(), buf, buf + n);
  const int ctrl_dim = 1 + neural_tts::kWorldNumBenv + neural_tts::kWorldNumBap;
  const int num_frames = static_cast<int>(raw.size()) / ctrl_dim;
  if (num_frames <= 0) {
    fprintf(stderr, "no frames on stdin\n");
    return 1;
  }

  std::vector<neural_tts::WorldFrame> frames(num_frames);
  for (int t = 0; t < num_frames; ++t) {
    const float* r = raw.data() + t * ctrl_dim;
    frames[t].f0 = r[0];
    memcpy(frames[t].benv, r + 1, sizeof(float) * neural_tts::kWorldNumBenv);
    memcpy(frames[t].bap, r + 1 + neural_tts::kWorldNumBenv,
           sizeof(float) * neural_tts::kWorldNumBap);
  }

  struct Ctx {
    const neural_tts::WorldFrame* frames;
  } ctx{frames.data()};

  neural_tts::WorldLiteSynth synth;
  synth.Synthesize(
      [](void* user, int t, neural_tts::WorldFrame* frame) {
        *frame = static_cast<Ctx*>(user)->frames[t];
      },
      &ctx, num_frames, gain,
      [](void*, const int16_t* samples, int count) {
        fwrite(samples, sizeof(int16_t), count, stdout);
      },
      nullptr);
  return 0;
}
