#ifndef MOONSHINE_TTS_PIPER_VOICE_LEVELS_H
#define MOONSHINE_TTS_PIPER_VOICE_LEVELS_H

#include <string_view>

namespace moonshine_tts {

/// How loud one Piper voice comes out, before any level shaping.
struct PiperVoiceLevel {
  std::string_view stem;
  /// The peak it reaches across a range of sentences in its own language,
  /// measured with normalization off.
  float peak;
};

/// The peak a streamed utterance aims for.
///
/// The same target Kokoro streaming uses, and for the same reason: short of 1
/// so an utterance louder than its voice's stored level has somewhere to go,
/// but only just short, because the gap is loudness given away against the
/// one-shot path. See kokoro-voice-levels.h for the measurements behind it.
inline constexpr float kPiperStreamingPeakTarget = 0.95F;

/// The peak `stem` is expected to reach, or 0 if it was never measured.
///
/// One-shot synthesis peak-normalizes each call, which needs the finished
/// waveform. Streaming has to commit to a gain before the generator runs, and
/// measurement showed that nothing available at that point predicts the level
/// except which voice is speaking. Piper's latent is the generator's own input,
/// so it looked like a better predictor than Kokoro's prosody curve, but it is
/// not: across a dozen utterances its magnitude correlates with the rendered
/// peak at roughly zero, and the error left after fitting it matches what a
/// plain per-voice constant leaves. So the levels are measured offline and
/// looked up here.
///
/// A voice with no entry returns 0 and leaves the caller to fall back.
float piper_voice_reference_peak(std::string_view stem);

/// What every unmeasured voice assumes, the median across those measured.
float piper_default_reference_peak();

/// The gain every chunk of a streamed utterance in `stem` is multiplied by.
///
/// One value for the whole utterance rather than one per chunk, so the level
/// cannot lurch at a join.
float piper_streaming_gain(std::string_view stem);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_PIPER_VOICE_LEVELS_H
