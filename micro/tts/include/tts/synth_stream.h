// Streaming, caller-arena formant synthesizer for the RP2350 (and desktop).
//
// Unlike the batch Synthesize() (synth.h), which materializes the whole PCM
// waveform in a std::vector and peak-normalizes it, StreamSynth produces audio
// in arbitrary-length chunks on demand and never holds the full utterance:
//
//   * the second-largest buffer -- the per-frame parameter tracks -- lives in a
//     caller-supplied arena (on the MCU this is the otherwise-idle TFLM tensor
//     arena, since recognition and synthesis never run at the same time);
//   * the largest buffer -- the PCM -- is never materialized; frames are
//     rendered one at a time into a tiny scratch and copied into the caller's
//     output buffer as Read() drains them.
//
// Because the global peak isn't known while streaming, the final loudness stage
// differs from batch: a fixed VoiceParams::output_gain plus a soft limiter
// replace peak-normalization. Everything upstream (segments, smoothing, F0) is
// the shared, bit-identical code in synth_internal.*.
//
// Lifecycle: construct, begin an utterance, pull audio chunks until done:

#ifndef TTS_SYNTH_STREAM_H_
#define TTS_SYNTH_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "config.h"
#include "g2p_dict.h"  // Lexicon
#include "klatt.h"
#include "synth_internal.h"

namespace tts {

// Per-utterance runtime options (the same shape as SynthOptions).
struct StreamOptions {
  float sample_rate = 22050.0f;
  float speed = 1.0f;     // >1 faster, <1 slower
  bool question = false;  // final rising boundary tone
};

// Return / status codes (0 == success, negative == error).
enum StreamStatus {
  kStreamOk = 0,
  kStreamErrBadArg = -1,     // null/empty argument
  kStreamErrArenaFull = -2,  // utterance needs more arena than provided
  kStreamErrNoPhones = -3,   // text/IPA produced no usable phones
};

class StreamSynth {
 public:
  // `vp` must outlive the synth (not copied). `arena`/`arena_size` is all the
  // working memory the audio path will use; nothing is heap-allocated per
  // sample. A few KiB of transient heap is used by the front-end (tokenizer /
  // segment list) inside Begin*, bounded by the input length.
  StreamSynth(const VoiceParams& vp, uint8_t* arena, size_t arena_size);

  // Begin a new utterance from plain text (on-device G2P). `overrides`, if
  // non-null, is consulted first (proper nouns etc.). Resets all stream state.
  int BeginText(const char* text, const StreamOptions& opts,
                const Lexicon* overrides = nullptr);

  // Begin a new utterance from an IPA string (bypasses the G2P; for isolating
  // the synthesizer). Resets all stream state.
  int BeginIpa(const char* ipa, const StreamOptions& opts);

  // Pull up to `max_samples` mono float samples (roughly [-1, 1]) into `out`.
  // Returns the number of samples written; 0 means the utterance is fully
  // drained (done() is then true). Chunks may be any size.
  int Read(float* out, int max_samples);

  bool done() const {
    return frame_idx_ >= nframes_ && frame_buf_pos_ >= frame_buf_len_;
  }
  int sample_rate() const { return sr_hz_; }
  int total_samples() const {
    return static_cast<int>(nframes_ * static_cast<size_t>(spf_));
  }

 private:
  int BeginPhones(const std::vector<std::string>& phones,
                  const StreamOptions& opts);
  void ArenaReset() { arena_used_ = 0; }
  float* ArenaFloats(size_t count);
  uint8_t* ArenaBytes(size_t count, size_t align);
  void RenderNextFrame();

  const VoiceParams& vp_;
  uint8_t* arena_;
  size_t arena_size_;
  size_t arena_used_ = 0;

  synth_detail::ParamTracks tracks_;
  size_t nframes_ = 0;
  int spf_ = 0;
  int sr_hz_ = 0;

  std::optional<KlattSynth> synth_;

  // Frame-at-a-time render scratch + drain cursor.
  float* frame_buf_ = nullptr;  // spf_ samples, in the arena
  int frame_buf_pos_ = 0;
  int frame_buf_len_ = 0;
  size_t frame_idx_ = 0;  // next frame to render

  // Global emitted-sample index, for edge fades.
  size_t out_idx_ = 0;
  size_t total_samples_ = 0;
  int fade_ = 0;
  float gain_ = 1.0f;
};

}  // namespace tts

#endif  // TTS_SYNTH_STREAM_H_
