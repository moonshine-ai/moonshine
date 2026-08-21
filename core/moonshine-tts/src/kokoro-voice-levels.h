#ifndef MOONSHINE_TTS_KOKORO_VOICE_LEVELS_H
#define MOONSHINE_TTS_KOKORO_VOICE_LEVELS_H

#include <string_view>

namespace moonshine_tts {

/// How loud one voice comes out, before any level shaping.
struct KokoroVoiceLevel {
  std::string_view id;
  /// The peak it reaches across a range of sentences, measured with
  /// normalization off.
  float peak;
};

/// The peak a streamed utterance aims for.
///
/// Short of 1 so the utterances that come out louder than their voice's stored
/// level have somewhere to go, but only just short, because the gap between
/// this and 1 is loudness given away against the one-shot path. Over 622
/// measured utterances 0.95 costs a median of 1.6 dB and clamps 16 samples of
/// the single loudest, under a millisecond; raising it to 1.0 would buy back
/// 0.45 dB, which is below what anyone can hear, and clamp four times as many
/// utterances.
///
/// This only centers the level. It cannot narrow the spread around it, because
/// one-shot normalization scales each utterance by its own peak and utterances
/// of one voice vary by about 4 dB peak to peak. A quiet one still arrives
/// quieter than the one-shot path would have made it.
inline constexpr float kStreamingPeakTarget = 0.95F;

/// The peak `voice_id` is expected to reach, or 0 if it was never measured.
///
/// One-shot synthesis peak-normalizes, which needs the finished waveform.
/// Streaming has to commit to a gain before the decoder runs, and measurement
/// showed nothing available at that point predicts the level except which voice
/// is speaking: the prosody stage's energy curve tracks the shape of an
/// utterance but not its scale, and the style vector predicts nothing at all.
/// So the levels are measured offline by tts-voice-level-calibrate and looked
/// up here.
///
/// A voice with no entry, which includes any cloned one, returns 0 and leaves
/// the caller to fall back.
float kokoro_voice_reference_peak(std::string_view voice_id);

/// What every unmeasured voice assumes, the median across those measured.
float kokoro_default_reference_peak();

/// The gain every chunk of a streamed utterance in `voice_id` is multiplied by.
///
/// One value for the whole utterance rather than one per chunk. A chunk's own
/// peak says more about which syllables happen to fall inside it than about how
/// loud the voice is, so normalizing chunk by chunk would make the level lurch
/// at every join, which is the artifact the growing chunk schedule exists to
/// avoid.
float kokoro_streaming_gain(std::string_view voice_id);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_KOKORO_VOICE_LEVELS_H
