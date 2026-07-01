#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "zipvoice-mel.h"
#include "zipvoice-voices.h"

#include <cmath>
#include <string>
#include <vector>

using namespace moonshine_tts;

TEST_CASE("zipvoice-builtin-voices") {
  size_t count = 0;
  const ZipVoiceBuiltinVoice* voices = zipvoice_builtin_voices(&count);
  REQUIRE(voices != nullptr);
  // One masculine + one feminine per accent where both exist, minus voices excluded after review.
  CHECK(count >= 12);

  const ZipVoiceBuiltinVoice* v = zipvoice_find_builtin_voice("american_female");
  REQUIRE(v != nullptr);
  CHECK(std::string(v->accent) == "American");
  CHECK(std::string(v->gender) == "female");
  CHECK(v->sample_rate == 24000u);
  CHECK(v->num_samples == 24000u * 4u);  // 4-second clip
  CHECK(v->pcm != nullptr);
  CHECK(std::string(v->prompt_text).size() > 0);

  CHECK(zipvoice_find_builtin_voice("indian_male") != nullptr);
  CHECK(zipvoice_find_builtin_voice("not_a_real_voice") == nullptr);
  // Removed after review — must no longer be present.
  CHECK(zipvoice_find_builtin_voice("scottish_male") == nullptr);
  CHECK(zipvoice_find_builtin_voice("british_female") == nullptr);

  const std::vector<float> f = zipvoice_builtin_voice_pcm_to_float(*v);
  REQUIRE(f.size() == v->num_samples);
  for (float x : f) {
    CHECK(x >= -1.0f);
    CHECK(x <= 1.0f);
  }
}

TEST_CASE("zipvoice-vocos-fbank") {
  VocosFbank fbank;
  // 1 second of a 440 Hz tone at 24 kHz.
  const int sr = VocosFbank::kSampleRate;
  std::vector<float> tone(static_cast<size_t>(sr));
  for (size_t i = 0; i < tone.size(); ++i) {
    tone[i] = 0.5f * std::sin(2.0 * 3.14159265358979 * 440.0 * static_cast<double>(i) / sr);
  }
  int frames = 0;
  const std::vector<float> mel = fbank.extract(tone, &frames);
  CHECK(frames == VocosFbank::num_frames_for(tone.size()));
  CHECK(frames == 1 + sr / VocosFbank::kHop);
  REQUIRE(mel.size() == static_cast<size_t>(frames) * VocosFbank::kNMels);
  for (float x : mel) {
    CHECK(std::isfinite(x));
    // log(clamp(min=1e-7)) >= log(1e-7)
    CHECK(x >= std::log(1e-7f) - 1e-3f);
  }
}
