#include "piper-voice-levels.h"

#include <algorithm>
#include <array>
#include <cstddef>

#include "piper-voice-levels-data.h"

namespace moonshine_tts {

float piper_voice_reference_peak(std::string_view stem) {
  const auto end = kPiperVoiceLevels.end();
  const auto found =
      std::lower_bound(kPiperVoiceLevels.begin(), end, stem,
                       [](const PiperVoiceLevel& level, std::string_view id) {
                         return level.stem < id;
                       });
  if (found == end || found->stem != stem) {
    return 0.F;
  }
  return found->peak;
}

float piper_streaming_gain(std::string_view stem) {
  float peak = piper_voice_reference_peak(stem);
  if (peak <= 0.F) {
    peak = piper_default_reference_peak();
  }
  if (peak <= 0.F) {
    return 1.F;
  }
  return kPiperStreamingPeakTarget / peak;
}

float piper_default_reference_peak() {
  constexpr size_t kCount = kPiperVoiceLevels.size();
  if constexpr (kCount == 0) {
    return 0.F;
  } else {
    // The table is sorted by stem rather than by level, so the median has to
    // be found rather than indexed. Once, since the table never changes.
    static const float median = [] {
      std::array<float, kCount> peaks{};
      for (size_t i = 0; i < kCount; ++i) {
        peaks[i] = kPiperVoiceLevels[i].peak;
      }
      std::sort(peaks.begin(), peaks.end());
      return peaks[kCount / 2];
    }();
    return median;
  }
}

}  // namespace moonshine_tts
