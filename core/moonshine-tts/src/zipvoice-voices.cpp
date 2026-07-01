#include "zipvoice-voices.h"

#include <cstring>

namespace moonshine_tts {

const ZipVoiceBuiltinVoice* zipvoice_find_builtin_voice(std::string_view id) {
  size_t count = 0;
  const ZipVoiceBuiltinVoice* voices = zipvoice_builtin_voices(&count);
  for (size_t i = 0; i < count; ++i) {
    if (id == voices[i].id) {
      return &voices[i];
    }
  }
  return nullptr;
}

std::vector<float> zipvoice_builtin_voice_pcm_to_float(const ZipVoiceBuiltinVoice& voice) {
  std::vector<float> out(voice.num_samples);
  for (uint32_t i = 0; i < voice.num_samples; ++i) {
    out[i] = static_cast<float>(voice.pcm[i]) / 32768.F;
  }
  return out;
}

}  // namespace moonshine_tts
