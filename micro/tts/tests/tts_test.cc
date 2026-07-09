// Unit tests for the TTS synth core, using TFLM's micro_test.h.
//
// Covers the English G2P front-end (text -> phone tokens, number normalization)
// and an end-to-end StreamSynth smoke test (text -> non-empty PCM that stays in
// range). The synth core has no platform dependency, so this runs on the host.

#include "tts/tts.h"

#include <string>
#include <vector>

#include "g2p.h"  // private header (TextToPhones) -- reached via the src path
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace {

bool Contains(const std::vector<std::string>& toks, const char* needle) {
  for (const auto& t : toks) {
    if (t == needle) return true;
  }
  return false;
}

}  // namespace

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(G2PProducesPhones) {
  const std::vector<std::string> toks = tts::TextToPhones("hello world");
  TF_LITE_MICRO_EXPECT_GT(static_cast<int>(toks.size()), 0);
  // Two words -> a word-gap token between them.
  TF_LITE_MICRO_EXPECT_TRUE(Contains(toks, " "));
}

TF_LITE_MICRO_TEST(G2PNumberNormalization) {
  // "21" should expand to words ("twenty one"), i.e. produce several phones.
  const std::vector<std::string> toks = tts::TextToPhones("21");
  TF_LITE_MICRO_EXPECT_GT(static_cast<int>(toks.size()), 2);
}

TF_LITE_MICRO_TEST(StreamSynthProducesAudioInRange) {
  static uint8_t arena[64 * 1024];
  tts::VoiceParams voice = tts::DefaultVoiceParams();
  tts::StreamSynth synth(voice, arena, sizeof(arena));
  tts::StreamOptions opts;
  opts.sample_rate = 16000.0f;
  TF_LITE_MICRO_EXPECT_EQ(synth.BeginText("a", opts), tts::kStreamOk);

  int total = 0;
  float peak = 0.0f;
  float buf[256];
  for (int n; (n = synth.Read(buf, 256)) > 0;) {
    total += n;
    for (int i = 0; i < n; ++i) {
      const float a = buf[i] < 0 ? -buf[i] : buf[i];
      if (a > peak) peak = a;
    }
  }
  TF_LITE_MICRO_EXPECT_GT(total, 0);
  TF_LITE_MICRO_EXPECT_TRUE(synth.done());
  // The soft limiter keeps the stream within a sane range.
  TF_LITE_MICRO_EXPECT_LE(peak, 1.5f);
}

TF_LITE_MICRO_TESTS_END
