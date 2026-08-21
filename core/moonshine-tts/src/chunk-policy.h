#ifndef MOONSHINE_TTS_CHUNK_POLICY_H
#define MOONSHINE_TTS_CHUNK_POLICY_H

#include <cstddef>
#include <vector>

namespace moonshine_tts {

/// How an utterance is cut into sub-sentence pieces for streaming.
///
/// Only engines with a two-stage graph can use this: a prosody stage that runs
/// once over the whole utterance, then a decoder that can be asked for a slice.
/// Everything here works in prosody frames, so it is engine-agnostic as long as
/// the engine reports how many audio samples a frame is worth.
struct ChunkPolicyOptions {
  /// Length of the first chunk. This is the only one that sets
  /// time-to-first-audio; the rest are produced while it plays.
  float first_chunk_seconds = 0.6f;

  /// How far a boundary may move from its nominal position to find a quiet
  /// frame. Cutting inside a stop closure is what keeps a join inaudible.
  float tolerance_seconds = 0.2f;

  /// Equal-power crossfade across each join.
  float crossfade_seconds = 0.025f;

  /// How much longer each chunk is than the one before it.
  ///
  /// Sub-sentence chunks are normalised over their own time span by the
  /// decoder, so a short chunk is normalised against statistics that do not
  /// represent the sentence and comes back at the wrong level. Fewer and longer
  /// chunks are the only effective remedy: correcting the level afterwards can
  /// only reach the part of the error that is a constant offset per chunk,
  /// which measures at about a third of it. Growing costs no latency because
  /// only the first chunk is decoded before playback starts, and it lowers
  /// decoder work as well, because padding is paid per chunk.
  ///
  /// The ceiling is decode speed, not quality. See `growth_is_sustainable`.
  float growth = 2.0f;

  /// Frames decoded either side of a chunk and then discarded, so the decoder's
  /// convolutions see their receptive field across the join.
  int pad_frames = 8;
};

/// One chunk: which frames to decode, and which samples of the result to keep.
///
/// `decode_first`/`decode_last` include the padding and the crossfade overhang,
/// so they overlap between neighbours. `keep_first`/`keep_last` are absolute
/// sample positions in the finished utterance, so successive chunks overlap by
/// exactly one crossfade and the joined length matches an unchunked render.
struct ChunkSpan {
  int decode_first_frame = 0;
  int decode_last_frame = 0;
  size_t keep_first_sample = 0;
  size_t keep_last_sample = 0;
};

/// Per-frame cost of cutting at each frame; lower is a better place to cut.
///
/// Quiet and unvoiced frames are cheapest. `f0` and `energy` arrive at twice
/// the frame rate in Kokoro, so each is averaged down in pairs when its length
/// allows; anything else is used as-is.
std::vector<float> boundary_cost(const std::vector<float>& f0,
                                 const std::vector<float>& energy, int frames);

/// Choose boundaries, growing each chunk by `options.growth` and snapping each
/// to the cheapest frame within tolerance.
///
/// Returns frame indices starting at 0 and ending at `frames`, so there is one
/// more entry than there are chunks. A `cost` of the wrong length disables
/// snapping and the boundaries land on the nominal grid.
std::vector<int> plan_boundaries(const std::vector<float>& cost, int frames,
                                 int frames_per_second,
                                 const ChunkPolicyOptions& options);

/// Turn boundaries into decode and keep spans.
std::vector<ChunkSpan> plan_spans(const std::vector<int>& boundaries,
                                  int frames, int samples_per_frame,
                                  int sample_rate,
                                  const ChunkPolicyOptions& options);

/// Append `piece` to `accumulated` with a sine-law crossfade over `overlap`
/// samples, which holds power constant through the join.
void crossfade_append(std::vector<float>& accumulated,
                      const std::vector<float>& piece, size_t overlap);

/// Whether a growth factor can be sustained by a decoder of this speed.
///
/// A chunk has to finish decoding before its predecessor finishes playing.
/// Growing multiplies the decode time and the buffer hiding it by the same
/// factor, so the test reduces to `growth * realtime_cost <= 1` and the first
/// join is the one that decides it. `realtime_cost` is decoder seconds per
/// second of audio, so 0.25 means four times faster than realtime.
///
/// Padding is charged on top, which is why this takes the policy rather than
/// just the two numbers: a fixed extra span per chunk weighs most on the short
/// early chunks, where the margin is already tightest.
bool growth_is_sustainable(const ChunkPolicyOptions& options,
                           float realtime_cost, int frames_per_second);

/// Largest growth factor this decoder can sustain, at or above 1.0.
float max_sustainable_growth(const ChunkPolicyOptions& options,
                             float realtime_cost, int frames_per_second);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_CHUNK_POLICY_H
