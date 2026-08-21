#ifndef MOONSHINE_TTS_MOONSHINE_TTS_H
#define MOONSHINE_TTS_MOONSHINE_TTS_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "moonshine-g2p-options.h"
#include "moonshine-tts-options.h"
#include "tts-stream.h"

namespace moonshine_tts {

bool kokoro_tts_lang_supported(std::string_view lang_cli,
                               const MoonshineG2POptions& g2p_opt = {});

/// Unified TTS: **Kokoro** and **Piper** ONNX backends; shared ``MoonshineG2P``
/// where applicable.
class MoonshineTTS {
 public:
  MoonshineTTS(std::string_view language, const MoonshineTTSOptions& opt);
  MoonshineTTS(const MoonshineTTS&) = delete;
  MoonshineTTS& operator=(const MoonshineTTS&) = delete;
  MoonshineTTS(MoonshineTTS&&) noexcept;
  MoonshineTTS& operator=(MoonshineTTS&&) noexcept;
  ~MoonshineTTS();

  static constexpr int kSampleRateHz = 24000;

  std::vector<float> synthesize(std::string_view text);

  /// Synthesize with per-call option overrides (same key names as
  /// ``MoonshineTTSOptions::parse_options``). Recognized keys: ``speed``,
  /// ``normalize_audio`` (legacy alias ``piper_normalize_audio``),
  /// ``output_volume`` (legacy alias ``piper_output_volume``). Other entries
  /// are ignored.
  std::vector<float> synthesize(
      std::string_view text,
      const std::vector<std::pair<std::string, std::string>>& option_overrides);

  /// Synthesize from an existing IPA phoneme string, skipping grapheme-to-
  /// phoneme conversion. ``phonemes`` is the same International Phonetic
  /// Alphabet representation produced by ``MoonshineG2P::text_to_ipa`` (i.e.
  /// the ``moonshine_text_to_phonemes`` C API), so callers can inspect or edit
  /// the phonemes between G2P and vocoding. The string is normalized to the
  /// active engine's phoneme inventory before synthesis.
  std::vector<float> synthesize_from_phonemes(std::string_view phonemes);

  /// ``synthesize_from_phonemes`` with per-call option overrides (same keys as
  /// the ``synthesize`` overload: ``speed``, ``normalize_audio``,
  /// ``output_volume``).
  std::vector<float> synthesize_from_phonemes(
      std::string_view phonemes,
      const std::vector<std::pair<std::string, std::string>>& option_overrides);

  /// Streaming synthesis: push text in as it becomes available, pull audio out
  /// a chunk at a time.
  ///
  /// A synthesizer runs one generation at a time. `push_text` starts one if
  /// none is running, `end_input` finishes it, and `cancel_stream` abandons it;
  /// `synthesize` throws while one is in flight rather than quietly competing
  /// for the model. There is no session object to own or outlive anything.
  ///
  /// Pull-based and synchronous. `next_chunk` blocks while it computes and
  /// never waits on anyone else, so a binding can drive it from whatever worker
  /// suits its platform without the core owning a thread policy.
  ///
  /// Chunk sizing follows the ``stream_*`` fields of the options this was
  /// constructed with. Engines that can decode part of an utterance cut inside
  /// a sentence; the rest fall back to one chunk per sentence, which still
  /// starts playback on the first clause rather than the last.

  /// Append text. Pieces are concatenated verbatim, so a token-by-token feed
  /// from a language model reassembles correctly. Any complete sentence this
  /// reveals is queued; a trailing fragment waits for more text.
  void push_text(std::string_view text);

  /// Queue whatever is buffered even if it does not look like a complete
  /// sentence, for when the caller knows the thought is finished but the
  /// punctuation does not say so.
  void flush();

  /// No more text is coming. Flushes, then makes `next_chunk` report
  /// `kEndOfStream` once the queue drains.
  void end_input();

  /// Produce the next chunk, synthesizing during the call.
  TtsStreamStatus next_chunk(TtsChunk& out);

  /// Drop queued text and abandon the generation in progress, returning the
  /// synthesizer to idle. Used when a conversation is interrupted and the
  /// pending reply is no longer wanted.
  void cancel_stream();

  /// Whether a streaming generation is in flight.
  bool is_streaming() const;

  /// Split text the way streaming does, for callers that want to queue
  /// utterances themselves. Uses the synthesizer's language for its
  /// abbreviation and terminator rules.
  std::vector<std::string> split_utterances(std::string_view text) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void write_wav_mono_pcm16(const std::filesystem::path& path,
                          const std::vector<float>& samples);

/// Kokoro or Piper vocoder asset keys only (no G2P), relative to ``g2p_root``.
/// With default ``MoonshineTTSOptions{}``, uses ``vocoder_engine=auto`` and
/// default voice layout.
std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(
    std::string_view language_cli);
/// Uses ``voice`` (optional ``kokoro_`` / ``piper_`` prefix sets vocoder),
/// Piper/Kokoro file map entries, and
/// ``g2p_options`` (e.g. Spanish narrow obstruents for auto engine) like
/// ``MoonshineTTS``.
std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(
    std::string_view language_cli, const MoonshineTTSOptions& options);

/// Union of vocoder keys across all languages in
/// ``moonshine_asset_catalog_all_registered_language_tags``.
std::vector<std::string>
moonshine_catalog_all_tts_vocoder_dependency_keys_union();

/// One Kokoro or Piper voice id and whether the asset is available (on disk or
/// in-memory file map).
struct MoonshineTtsVoiceAvailability {
  std::string id;
  bool available = false;
};

/// All known voices for ``language_cli`` with availability, using the same path
/// layout rules as
/// ``moonshine_catalog_tts_vocoder_only_dependency_keys``. Returned ``id``
/// values are prefixed with ``kokoro_`` or
/// ``piper_``. When vocoder is ``auto``, Kokoro and Piper catalogs are merged
/// (both prefixes). Kokoro uses the upstream Kokoro-82M voice catalog
/// (VOICES.md) plus any extra ``*.kokorovoice`` under the resolved voices
/// directory. Piper uses the language default ONNX stem plus any ``*.onnx`` in
/// the resolved voices directory. The ``voice`` field in ``options`` does not
/// filter the list.
std::vector<MoonshineTtsVoiceAvailability>
moonshine_list_tts_voices_with_availability(std::string_view language_cli,
                                            const MoonshineTTSOptions& options);

}  // namespace moonshine_tts

#endif
