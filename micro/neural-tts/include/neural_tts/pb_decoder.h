// TFLM wrapper for the Phase B RVQ decoder (s16x8: int16 activations,
// int8 weights). Streams WORLD-lite control frames out of a coded
// utterance: RVQ codes -> int8-codebook latent sums -> tiled TFLM decode
// (TL latents in, 4*TL frames out, HOP-latent steps with discarded
// margins) -> benv/bap frames, plus a u16 f0 side stream.
//
// Designed to plug straight into WorldLiteSynth::GetFrameFn: frames are
// requested (near-)sequentially and tiles are decoded lazily.

#ifndef NEURAL_TTS_PB_DECODER_H_
#define NEURAL_TTS_PB_DECODER_H_

#include <cstddef>
#include <cstdint>

#include "neural_tts/worldlite_synth.h"

namespace tflite {
class MicroInterpreter;
struct Model;
}  // namespace tflite

namespace neural_tts {

// Static geometry of the shipped graph (must match the export;
// see scripts/export_pb_decoder_litert.py and export_pb_demo_data.py).
constexpr int kPbLatentDimMax = 64;
constexpr int kPbStagesMax = 3;
constexpr int kPbTileLatentsMax = 32;

struct PbCodedUtterance {
  int n_frames;
  int n_latents;
  const uint16_t* codes;  // [n_latents][n_stages]
  const uint16_t* f0q;    // [n_frames], Hz * 16, 0 = unvoiced
};

class PbDecoder {
 public:
  struct Config {
    const unsigned char* model_data;
    const int8_t* codebooks[kPbStagesMax];      // [k][dim] row-major
    const float* codebook_scales[kPbStagesMax];  // [dim]
    int n_stages;
    int latent_dim;
    int tile_latents;  // TL
    int tile_hop;      // latents kept per step
    float input_scale;   // graph int16 input quantization
    float output_scale;  // graph int16 output quantization
  };

  // Arena is caller-owned (e.g. the shared app arena).
  PbDecoder(const Config& config, uint8_t* arena, size_t arena_bytes);

  bool ok() const { return ok_; }
  size_t arena_used_bytes() const;

  void BeginUtterance(const PbCodedUtterance* utt);

  // WorldLiteSynth::GetFrameFn-compatible; `user` is the PbDecoder*.
  static void GetFrameThunk(void* user, int t, WorldFrame* frame);
  void GetFrame(int t, WorldFrame* frame);

  // Batch-read rows [t0, t0+n) of the current utterance's RAW int16 graph
  // output ([60] = benv dB/20 then bap, quantized by output_scale) into
  // `out`, decoding tiles lazily. Calls must be (near-)monotonic in t0:
  // forward skips of any size are fine (the reader jumps the tile grid),
  // but backward seeks are limited to the tile overlap margin. Used by
  // the unit-selection synthesizer, which concatenates all selected
  // units' codes into one stream so tile compute is amortized across
  // units instead of being padded per unit.
  void ReadRows(int t0, int n, int16_t* out);

  // cumulative microseconds spent inside TFLM Invoke() + latent prep
  uint64_t decode_us() const { return decode_us_; }
  int tiles_decoded() const { return tiles_decoded_; }

 private:
  // false = Invoke failed (window state unchanged); callers must not retry
  // in a loop.
  bool DecodeTileAt(int latent_start);

  Config cfg_;
  bool ok_ = false;
  const tflite::Model* model_ = nullptr;
  tflite::MicroInterpreter* interp_ = nullptr;
  int16_t* in_data_ = nullptr;
  const int16_t* out_data_ = nullptr;

  const PbCodedUtterance* utt_ = nullptr;
  // decoded window: frames [win_frame0_, win_frame0_ + win_frames_)
  int win_frame0_ = 0;
  int win_frames_ = 0;
  int next_latent_ = 0;
  // int16 model output for the kept window, [tile_hop*4][60]
  int16_t win_buf_[kPbTileLatentsMax * 4 * 60];
  float q_[kPbLatentDimMax];  // DecodeTileAt scratch (stack is 4 KB)

  uint64_t decode_us_ = 0;
  int tiles_decoded_ = 0;
};

}  // namespace neural_tts

#endif  // NEURAL_TTS_PB_DECODER_H_
