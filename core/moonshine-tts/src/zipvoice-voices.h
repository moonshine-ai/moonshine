#ifndef MOONSHINE_TTS_ZIPVOICE_VOICES_H
#define MOONSHINE_TTS_ZIPVOICE_VOICES_H

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace moonshine_tts {

/// One built-in ZipVoice reference voice to clone, sourced from the VCTK corpus and compiled into
/// the library as a 4-second, 24 kHz, mono 16-bit PCM clip plus its exact transcript. Selected as one
/// masculine and one feminine speaker per accent where both are available. The ``id`` is the accent +
/// gender slug (e.g. ``american_female``); callers select it via ``voice = "zipvoice_<id>"``.
struct ZipVoiceBuiltinVoice {
  const char* id;
  const char* accent;
  const char* gender;
  const char* vctk_speaker;
  const char* clone_transcript;
  const int16_t* pcm;      // mono 16-bit PCM
  uint32_t num_samples;
  uint32_t sample_rate;    // always 24000
};

/// The compiled-in table of built-in voices. ``count`` receives the number of entries.
const ZipVoiceBuiltinVoice* zipvoice_builtin_voices(size_t* count);

/// Looks up a built-in voice by ``id`` (without the ``zipvoice_`` prefix), or nullptr if not found.
const ZipVoiceBuiltinVoice* zipvoice_find_builtin_voice(std::string_view id);

/// Convenience: decode a built-in voice's PCM to normalized float samples in ``[-1, 1]``.
std::vector<float> zipvoice_builtin_voice_pcm_to_float(const ZipVoiceBuiltinVoice& voice);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_ZIPVOICE_VOICES_H
