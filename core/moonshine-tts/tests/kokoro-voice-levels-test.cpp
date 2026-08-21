#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "kokoro-voice-levels.h"

#include <doctest/doctest.h>

#include "kokoro-voice-levels-data.h"

using namespace moonshine_tts;

TEST_CASE("a measured voice has its own level") {
  const float peak = kokoro_voice_reference_peak("af_heart");
  CHECK(peak > 0.F);
  CHECK(peak <= 1.F);
}

TEST_CASE("an unknown voice has no level of its own") {
  CHECK(kokoro_voice_reference_peak("not_a_voice") == 0.F);
  // Cloned voices are named by the caller, so they will never be in the table.
  CHECK(kokoro_voice_reference_peak("") == 0.F);
}

TEST_CASE("lookup finds every entry") {
  // The lookup binary searches, so a table that fell out of sorted order would
  // silently start missing voices rather than failing to build.
  for (const KokoroVoiceLevel& level : kKokoroVoiceLevels) {
    CHECK(kokoro_voice_reference_peak(level.id) == level.peak);
  }
}

TEST_CASE("the table is sorted by id") {
  for (size_t i = 1; i < std::size(kKokoroVoiceLevels); ++i) {
    CHECK(kKokoroVoiceLevels[i - 1].id < kKokoroVoiceLevels[i].id);
  }
}

TEST_CASE("gain brings a voice to the streaming target") {
  const float peak = kokoro_voice_reference_peak("af_heart");
  CHECK(peak * kokoro_streaming_gain("af_heart") ==
        doctest::Approx(kStreamingPeakTarget));
}

TEST_CASE("an unknown voice falls back to the median level") {
  const float fallback = kokoro_default_reference_peak();
  CHECK(fallback > 0.F);
  CHECK(fallback * kokoro_streaming_gain("not_a_voice") ==
        doctest::Approx(kStreamingPeakTarget));
}

TEST_CASE("every gain stays within a sensible range") {
  // A gain far from 1 would mean a voice was measured against the wrong
  // language, or that the table predates a model change.
  for (const KokoroVoiceLevel& level : kKokoroVoiceLevels) {
    const float gain = kokoro_streaming_gain(level.id);
    CHECK(gain > 0.5F);
    CHECK(gain < 4.F);
  }
}
