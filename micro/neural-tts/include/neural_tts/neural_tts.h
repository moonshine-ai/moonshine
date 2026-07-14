// neural_tts -- black-box text-to-speech for the RP2350.
//
// Text in, 16 kHz mono int16 PCM chunks out. Everything else -- G2P,
// duration rules, diphone/word unit selection, RVQ decode through the
// s16x8 TFLM graph, prosody shaping, and WORLD-lite vocoding -- happens
// behind this interface, driven by a single flash-resident data pack
// (g_neural_tts_pack, built by scripts/export_neural_tts_pack.py).
//
// The synthesis pipeline mirrors the Phase A/B host reference
// (scripts/synth_diphone_world.py with the shipping flag set: cap 32,
// w_join 5, xfade 3, gain_eq, dur_mode=unit, f0_mode=unit_decl,
// prosody_shape, timbre_norm 0.3, f0_norm, prosody model), with these
// deliberate on-device simplifications:
//   * word-unit continuity costs skip the log-envelope term (edge
//     vectors are stored for diphones only),
//   * diphone candidate pools are capped at 8 (the pack stores <= 6
//     usage-ranked candidates per type anyway; only class-fallback pools
//     were ever larger),
//   * long inputs are split into <= ~5 s chunks at silence (then word-gap)
//     boundaries and synthesized sequentially, so RAM stays bounded.
//
// Typical use (tts_service.cc):
//   neural_tts::NeuralTts tts(cfg, arena, arena_size);   // once
//   tts.Synthesize("hello world", emit_fn, user);        // per utterance
//
// The arena is only used inside Synthesize() calls; between calls it can
// be lent to other subsystems (VAD / STT), matching the app's phase-
// disjoint arena sharing.

#ifndef NEURAL_TTS_NEURAL_TTS_H_
#define NEURAL_TTS_NEURAL_TTS_H_

#include <cstddef>
#include <cstdint>

#include "neural_tts/pack_format.h"

namespace neural_tts {

class NeuralTts {
 public:
  // `pack` is the flash pack (must outlive the object). The arena must be
  // at least kMinArenaBytes; ~340 KiB is comfortable.
  NeuralTts(const uint8_t* pack, uint8_t* arena, size_t arena_size);

  static constexpr size_t kMinArenaBytes = 300 * 1024;
  static constexpr int kSampleRate = 16000;

  bool ok() const { return ok_; }

  // emit(user, samples, n): consecutive 16 kHz mono int16 PCM.
  typedef void (*EmitFn)(void* user, const int16_t* samples, int n);

  // Synthesize plain English text (on-device G2P). Returns total samples
  // emitted, or a negative error code. Blocking; audio is emitted in
  // chunks as it is rendered.
  int Synthesize(const char* text, EmitFn emit, void* user);

  // Same, from an IPA string (bypasses the G2P word front end).
  int SynthesizeIpa(const char* ipa, EmitFn emit, void* user);

  // Exact sample count Synthesize(text) will produce, without decoding or
  // rendering (runs the deterministic planning passes only). Useful for
  // protocols that announce the length before streaming.
  int EstimateSamples(const char* text);
  int EstimateSamplesIpa(const char* ipa);

  // Wall-clock breakdown of the most recent Synthesize*/EstimateSamples*
  // call (microseconds). Sums over all chunks of the utterance.
  struct Stats {
    uint32_t g2p_us;      // text -> phone tokens
    uint32_t runs_us;     // Klatt rule segments -> phone runs
    uint32_t plan_us;     // prosody buckets + unit selection + parts
    uint32_t stream_us;   // RVQ code unpack into the contiguous stream
    uint32_t alloc_us;    // PbDecoder construction (AllocateTensors)
    uint32_t decode_us;   // Materialize: ReadRows + f0 + warp/assemble
    uint32_t invoke_us;   // TFLM Invoke portion of decode_us
    uint32_t post_us;     // gain EQ + join smoothing + f0 pass
    uint32_t render_us;   // WORLD-lite vocoder + emit
    uint32_t first_pcm_us;  // start of call -> first emit() of chunk 0
    int chunks;
    int tiles;
  };
  const Stats& stats() const { return stats_; }

 private:
  int SynthesizeTokens(void* tokens_vec, EmitFn emit, void* user,
                       bool plan_only);

  Pack pack_;
  uint8_t* arena_;
  size_t arena_size_;
  bool ok_ = false;
  Stats stats_ = {};
};

}  // namespace neural_tts

#endif  // NEURAL_TTS_NEURAL_TTS_H_
