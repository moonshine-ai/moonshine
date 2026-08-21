#ifndef MOONSHINE_TTS_TTS_STREAM_H
#define MOONSHINE_TTS_TTS_STREAM_H

#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "chunk-policy.h"
#include "sentence-splitter.h"

namespace moonshine_tts {

/// What `MoonshineTTS::next_chunk` found.
enum class TtsStreamStatus {
  /// A chunk was produced.
  kChunk,
  /// No complete unit is buffered. Push more text, or flush.
  kNeedText,
  /// Input ended and everything buffered has been synthesized.
  kEndOfStream,
  /// The generation was abandoned by `cancel_stream`, so the reply the caller
  /// was playing will not be finished. Reported once and only when there was
  /// something to abandon, which lets a consumer tell an interruption apart
  /// from a reply that simply ran out of text.
  kCancelled,
};

/// One piece of synthesized audio, small enough to start playing before the
/// rest of the utterance exists.
struct TtsChunk {
  std::vector<float> audio{};
  int sample_rate = 24000;
  /// The text this chunk covers, when the engine can attribute it. Sources that
  /// cut on acoustic frames rather than word boundaries leave this empty for
  /// every chunk but the first.
  std::string text{};
  /// Which queued utterance this came from. Increments per utterance so a
  /// consumer can tell where one reply ends and the next begins.
  uint64_t utterance_id = 0;
  /// Set on the last chunk of an utterance.
  bool is_final = false;
};

/// Turns one utterance of text into a sequence of audio chunks.
///
/// Engines that can decode a slice at a time implement this directly; anything
/// else uses `WholeUtteranceChunkSource` and streams at sentence granularity,
/// which is what makes streaming available on every voice we ship rather than
/// only the ones with a split graph.
class ChunkSource {
 public:
  virtual ~ChunkSource() = default;

  /// Start a new utterance. Whatever this does is paid before the caller hears
  /// anything, so engines should defer as much as they can to `next`.
  virtual void begin(std::string_view text) = 0;

  /// The next piece of audio, or an empty vector once the utterance is done.
  virtual std::vector<float> next() = 0;

  /// Whether another non-empty chunk is available. Must be answerable without
  /// synthesizing it: the stream uses this to set `is_final`, and computing one
  /// chunk ahead would give back the latency streaming is here to win.
  virtual bool has_more() const = 0;

  virtual int sample_rate() const = 0;
};

/// A source for engines whose graph is split into a prosody stage and a
/// decoder that can be asked for a range of frames.
///
/// The prosody stage runs once in `begin`, which is what the caller waits on
/// before the first chunk; the decoder then runs per chunk in `next`. Chunk
/// lengths grow, so the first one is short enough to start playback quickly
/// while later ones are long enough that the decoder's own normalisation sees
/// a representative span. See `ChunkPolicyOptions::growth`.
class SlicedDecodeChunkSource : public ChunkSource {
 public:
  /// What the prosody stage produced for one utterance. `f0` and `energy` pick
  /// the cut points; the engine keeps whatever else it needs itself.
  struct Prosody {
    int frames = 0;
    std::vector<float> f0{};
    std::vector<float> energy{};
  };

  /// Run the prosody stage. Called once per utterance. Reporting zero frames
  /// means this utterance cannot be sliced, and the fallback is used instead.
  using AnalyzeFn = std::function<Prosody(std::string_view)>;
  /// Decode frames `[first, last)` of the utterance most recently analyzed.
  using DecodeFn = std::function<std::vector<float>(int first, int last)>;
  /// Synthesize a whole utterance, for input the stages cannot take. Kokoro's
  /// prosody stage is capped at 512 tokens, and a caller who pushes a sentence
  /// longer than that must still hear it.
  using FallbackFn = std::function<std::vector<float>(std::string_view)>;

  SlicedDecodeChunkSource(AnalyzeFn analyze, DecodeFn decode,
                          FallbackFn fallback, ChunkPolicyOptions policy,
                          int samples_per_frame, int sample_rate);

  void begin(std::string_view text) override;
  std::vector<float> next() override;
  bool has_more() const override;
  int sample_rate() const override;

 private:
  AnalyzeFn analyze_;
  DecodeFn decode_;
  FallbackFn fallback_;
  ChunkPolicyOptions policy_;
  int samples_per_frame_;
  int sample_rate_;
  std::vector<ChunkSpan> spans_{};
  size_t next_span_ = 0;
  /// Text held for the fallback path, non-empty only when slicing was declined
  /// for this utterance.
  std::string whole_text_{};
  /// The tail of the previous chunk that still has to be blended with the next
  /// one. Held back rather than emitted, so what callers receive never overlaps
  /// and can be played end to end.
  std::vector<float> carry_{};
};

/// A source for engines whose stages reproduce the whole render exactly, so
/// chunks can be played back to back with nothing done at the joins.
///
/// Piper is one. Given the same latent frames with padding either side, its
/// generator returns the samples the whole utterance would have held, to about
/// a millionth. That removes both of the things the Kokoro path needs: the
/// crossfade that hides the step at a join, and choosing cut points where the
/// audio is quiet enough for that step to be missed. What is left is the
/// growing chunk schedule, which is here for latency rather than for artifacts.
///
/// Asking the engine for a frame range, rather than working out where that
/// range starts in samples, is also what lets a voice stream whose frames are
/// not a whole number of output samples. Piper's are not: a 22.05 kHz voice
/// resampled to 24 kHz puts 278.6 samples in a frame.
///
/// `ChunkPolicyOptions::crossfade_seconds` and `tolerance_seconds` are unused.
class ExactSliceChunkSource : public ChunkSource {
 public:
  /// Prepare `text` and report how many frames long it is. Reporting zero
  /// means this utterance cannot be sliced, and the fallback is used instead.
  using AnalyzeFn = std::function<int(std::string_view)>;
  /// The audio for frames `[first, last)` of the utterance most recently
  /// analyzed, already at `sample_rate`.
  using DecodeFn = std::function<std::vector<float>(int first, int last)>;
  /// Synthesize a whole utterance, for input the stages cannot take.
  using FallbackFn = std::function<std::vector<float>(std::string_view)>;

  ExactSliceChunkSource(AnalyzeFn analyze, DecodeFn decode, FallbackFn fallback,
                        ChunkPolicyOptions policy, int frames_per_second,
                        int sample_rate);

  void begin(std::string_view text) override;
  std::vector<float> next() override;
  bool has_more() const override;
  int sample_rate() const override;

 private:
  AnalyzeFn analyze_;
  DecodeFn decode_;
  FallbackFn fallback_;
  ChunkPolicyOptions policy_;
  int frames_per_second_;
  int sample_rate_;
  /// Frame indices, one more than the number of chunks.
  std::vector<int> boundaries_{};
  size_t next_chunk_ = 0;
  /// Text held for the fallback path, non-empty only when slicing was declined.
  std::string whole_text_{};
};

/// The fallback source: synthesize the whole utterance and hand it back as a
/// single chunk.
class WholeUtteranceChunkSource : public ChunkSource {
 public:
  using SynthesizeFn = std::function<std::vector<float>(std::string_view)>;

  WholeUtteranceChunkSource(SynthesizeFn synthesize, int sample_rate);

  void begin(std::string_view text) override;
  std::vector<float> next() override;
  bool has_more() const override;
  int sample_rate() const override;

 private:
  SynthesizeFn synthesize_;
  int sample_rate_;
  std::string text_{};
  bool pending_ = false;
};

/// The state of one streaming generation: buffered text, the queue of
/// utterances it has revealed, and how far through the current one we are.
///
/// Internal. `MoonshineTTS` owns at most one of these and exposes its
/// operations as its own methods, so callers never see this type and there is
/// no object whose lifetime has to be kept inside the synthesizer's.
///
/// Deliberately single-threaded and synchronous. `next_chunk` blocks while it
/// computes and never waits on anyone else, so bindings can put it on whatever
/// worker thread suits their platform without the core owning a thread policy.
class TtsStream {
 public:
  using Status = TtsStreamStatus;

  TtsStream(std::unique_ptr<ChunkSource> source,
            SentenceSplitOptions split_options);
  TtsStream(const TtsStream&) = delete;
  TtsStream& operator=(const TtsStream&) = delete;
  ~TtsStream();

  /// Append text. Pieces are concatenated verbatim, so a token-by-token feed
  /// from an LLM reassembles correctly. Any complete sentence this reveals is
  /// queued for synthesis; a trailing fragment waits for more text.
  void push_text(std::string_view text);

  /// Queue whatever is buffered even if it does not look like a complete
  /// sentence. Use this at a point where the caller knows the thought is
  /// finished but the punctuation does not say so.
  void flush();

  /// No more text is coming. Flushes, then makes `next_chunk` report
  /// `kEndOfStream` once the queue drains.
  void end_input();

  /// Produce the next chunk, synthesizing if needed.
  Status next_chunk(TtsChunk& out);

  /// Drop queued text and abandon the utterance in progress. Used when a
  /// conversation is interrupted and the pending reply is no longer wanted.
  void cancel();

  /// Whether `end_input` has been called.
  bool input_ended() const { return input_ended_; }

 private:
  bool start_next_unit();

  std::unique_ptr<ChunkSource> source_;
  SentenceSplitOptions split_options_;
  std::string buffer_{};
  std::deque<std::string> units_{};
  std::string current_text_{};
  uint64_t next_utterance_id_ = 1;
  uint64_t current_utterance_id_ = 0;
  bool source_active_ = false;
  bool input_ended_ = false;
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_TTS_STREAM_H
