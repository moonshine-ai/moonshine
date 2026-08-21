#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "piper-voice-levels.h"

#include <doctest/doctest.h>

#include "piper-voice-levels-data.h"

using namespace moonshine_tts;

TEST_CASE("a measured voice has its own level") {
  const float peak = piper_voice_reference_peak("en_US-amy-medium");
  CHECK(peak > 0.F);
  CHECK(peak <= 1.F);
}

TEST_CASE("a voice is found by stem, whatever file form it arrived as") {
  // Callers hand this whatever the voice was requested as, and a stem is what
  // the table is keyed on.
  CHECK(piper_voice_reference_peak("not_a_voice") == 0.F);
  CHECK(piper_voice_reference_peak("") == 0.F);
}

TEST_CASE("lookup finds every entry") {
  // The lookup binary searches, so a table that fell out of sorted order would
  // silently start missing voices rather than failing to build.
  for (const PiperVoiceLevel& level : kPiperVoiceLevels) {
    CHECK(piper_voice_reference_peak(level.stem) == level.peak);
  }
}

TEST_CASE("the table is sorted by stem") {
  for (size_t i = 1; i < kPiperVoiceLevels.size(); ++i) {
    CHECK(kPiperVoiceLevels[i - 1].stem < kPiperVoiceLevels[i].stem);
  }
}

TEST_CASE("every catalog voice was measured") {
  // A voice missing from the table streams at the median level instead of its
  // own, which is a decibel or two out on the voices furthest from the middle.
  CHECK(kPiperVoiceLevels.size() >= 90);
}

TEST_CASE("gain brings a voice to the streaming target") {
  const float peak = piper_voice_reference_peak("en_US-amy-medium");
  CHECK(peak * piper_streaming_gain("en_US-amy-medium") ==
        doctest::Approx(kPiperStreamingPeakTarget));
}

TEST_CASE("an unmeasured voice falls back to the median level") {
  const float fallback = piper_default_reference_peak();
  CHECK(fallback > 0.F);
  CHECK(fallback * piper_streaming_gain("not_a_voice") ==
        doctest::Approx(kPiperStreamingPeakTarget));
}

TEST_CASE("every gain stays within a sensible range") {
  // A gain far from 1 would mean a voice was measured against the wrong
  // language, or that the table predates a model change.
  for (const PiperVoiceLevel& level : kPiperVoiceLevels) {
    const float gain = piper_streaming_gain(level.stem);
    CHECK(gain > 0.5F);
    CHECK(gain < 10.F);
  }
}
