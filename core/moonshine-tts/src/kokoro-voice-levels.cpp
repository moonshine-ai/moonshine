#include "kokoro-voice-levels.h"

#include <algorithm>
#include <array>
#include <cstddef>

#include "kokoro-voice-levels-data.h"

namespace moonshine_tts {

float kokoro_voice_reference_peak(std::string_view voice_id) {
  const auto* const begin = std::begin(kKokoroVoiceLevels);
  const auto* const end = std::end(kKokoroVoiceLevels);
  const auto* found =
      std::lower_bound(begin, end, voice_id,
                       [](const KokoroVoiceLevel& level, std::string_view id) {
                         return level.id < id;
                       });
  if (found == end || found->id != voice_id) {
    return 0.F;
  }
  return found->peak;
}

float kokoro_streaming_gain(std::string_view voice_id) {
  float peak = kokoro_voice_reference_peak(voice_id);
  if (peak <= 0.F) {
    peak = kokoro_default_reference_peak();
  }
  if (peak <= 0.F) {
    return 1.F;
  }
  return kStreamingPeakTarget / peak;
}

float kokoro_default_reference_peak() {
  constexpr size_t kCount = std::size(kKokoroVoiceLevels);
  if constexpr (kCount == 0) {
    return 0.F;
  } else {
    // The table is sorted by id rather than by level, so the median has to be
    // found rather than indexed. Small and fixed, so a copy is cheaper than
    // keeping a second sorted table around.
    std::array<float, kCount> peaks{};
    for (size_t i = 0; i < kCount; ++i) {
      peaks[i] = kKokoroVoiceLevels[i].peak;
    }
    std::sort(peaks.begin(), peaks.end());
    return peaks[kCount / 2];
  }
}

}  // namespace moonshine_tts
