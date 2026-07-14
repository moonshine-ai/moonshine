// Shared internals for the batch (synth.cc) and streaming (synth_stream.cc)
// drivers. Factoring these out keeps the two paths bit-identical through
// segment building, frame rasterization, smoothing, and the F0 contour -- the
// only intentional difference is the final loudness stage (the batch path
// peak-normalizes the finished buffer; the streaming path can't see it and uses
// a fixed gain + limiter).

#ifndef TTS_SYNTH_INTERNAL_H_
#define TTS_SYNTH_INTERNAL_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "config.h"
#include "klatt.h"

namespace tts {
namespace synth_detail {

constexpr float kFrameMs = 5.0f;
// A silence longer than this (intrinsic, pre-scale) is a phrase boundary; word
// gaps (~60 ms) are not, so pitch flows across them but resets at sentence
// ends.
constexpr float kPhraseBreakMs = 120.0f;

// Intermediate target segment, before rasterization into frames.
struct Segment {
  float dur_ms = 0.0f;
  float f1 = 0, f2 = 0, f3 = 0, b1 = 0, b2 = 0, b3 = 0;
  float av = 0, af = 0, ah = 0;
  float nasal = 0;
  float fnp = 0, fnz = 0, fric_cf = 0;
  bool is_vowel = false;
  bool is_silence = false;
  bool major_pause = false;
  float accent = 0.0f;
  // Index of the input phone token this segment came from (-1 for synthetic
  // lead/tail silence).
  int src_token = -1;
};

// Forward+backward (zero-phase) / forward-only / asymmetric one-pole smoothing
// over a per-frame track stored as a raw array.
void SmoothBidir(float* v, size_t n, float tau_ms);
void SmoothFwd(float* v, size_t n, float tau_ms);
void SmoothAsym(float* v, size_t n, float attack_ms, float release_ms);

// Step 1 (+1b): phone tokens -> segment list (stop expansion, lead/tail
// silence, stress/accent assignment with downstep, context-dependent duration).
std::vector<Segment> BuildSegments(const std::vector<std::string>& phones,
                                   const VoiceParams& vp);

// Heap-free variant: writes into caller-owned `out` (e.g. tensor-arena bump).
// Returns segment count, or -1 on overflow / lookup failure.
int BuildSegments(const char* const* phones, int n_phones, const VoiceParams& vp,
                  Segment* out, int max_out);

// Number of frames a segment list rasterizes to at a given duration scale.
size_t CountFrames(const std::vector<Segment>& segs, float dur_scale);

// Per-frame parameter tracks, addressed by frame index in [0, n). The arrays
// are owned by the caller (std::vector data on desktop, an arena slab on the
// MCU); this is just a bundle of pointers so the fill code is storage-agnostic.
struct ParamTracks {
  float* f1 = nullptr;
  float* f2 = nullptr;
  float* f3 = nullptr;
  float* b1 = nullptr;
  float* b2 = nullptr;
  float* b3 = nullptr;
  float* av = nullptr;
  float* af = nullptr;
  float* ah = nullptr;
  float* nasal = nullptr;
  float* fnp = nullptr;
  float* fnz = nullptr;
  float* fric_cf = nullptr;
  float* accent = nullptr;
  float* f0 = nullptr;
  uint8_t* major = nullptr;
  size_t n = 0;
};

// Steps 2-4b: rasterize segments into `t` (which must already point at arrays
// of length t.n == CountFrames(segs, dur_scale)), smooth the tracks, lay down
// the F0 contour, and apply the voice-identity (formant/F0) scaling.
void FillParamTracks(const std::vector<Segment>& segs, const VoiceParams& vp,
                     float dur_scale, bool question, ParamTracks& t);

// Materialize the SynthFrame for frame index `i`.
SynthFrame FrameAt(const ParamTracks& t, size_t i);

// Translate the tunable voice into the Klatt core's parameter block (includes
// the formant_scale applied to the fixed high formants).
KlattParams MakeKlattParams(const VoiceParams& vp);

}  // namespace synth_detail
}  // namespace tts

#endif  // TTS_SYNTH_INTERNAL_H_
