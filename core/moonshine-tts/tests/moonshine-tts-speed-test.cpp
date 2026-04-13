#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "moonshine-tts.h"
#include "rule-g2p-test-support.h"

#include <filesystem>

using namespace moonshine_tts;
namespace r = moonshine_tts::rule_g2p_test;

namespace {

bool bundled_tts_data_present(const std::filesystem::path& root) {
  namespace fs = std::filesystem;
  return fs::is_regular_file(root / "kokoro" / "model.onnx") &&
         fs::is_regular_file(root / "kokoro" / "config.json") &&
         fs::is_regular_file(root / "en_us" / "piper-voices" / "en_US-lessac-medium.onnx");
}

}  // namespace

TEST_CASE("MoonshineTTS Kokoro: per-call speed changes duration and restores default") {
  const std::filesystem::path root = r::moonshine_tts_bundled_data_dir_relative();
  if (!bundled_tts_data_present(root)) {
    return;
  }
  MoonshineTTSOptions opt;
  opt.g2p_options.g2p_root = root;
  opt.voice = "kokoro_af_heart";
  opt.speed = 1.0;
  MoonshineTTS tts("en_us", opt);
  const std::string text = "Hello world. This exercises Kokoro speech synthesis speed.";
  const std::vector<float> baseline = tts.synthesize(text);
  REQUIRE(baseline.size() > 2000u);
  const std::vector<float> fast = tts.synthesize(text, {{"speed", "2.0"}});
  CHECK(fast.size() < baseline.size());
  const std::vector<float> after = tts.synthesize(text);
  CHECK(after.size() == baseline.size());
}

TEST_CASE("MoonshineTTS Piper: per-call speed reduces duration vs baseline") {
  const std::filesystem::path root = r::moonshine_tts_bundled_data_dir_relative();
  if (!bundled_tts_data_present(root)) {
    return;
  }
  MoonshineTTSOptions opt;
  opt.g2p_options.g2p_root = root;
  opt.voice = "piper_en_US-lessac-medium";
  opt.speed = 1.0;
  MoonshineTTS tts("en_us", opt);
  const std::string text = "Hello world. This exercises Piper speech synthesis speed.";
  const std::vector<float> baseline = tts.synthesize(text);
  REQUIRE(baseline.size() > 2000u);
  const std::vector<float> fast = tts.synthesize(text, {{"speed", "2.0"}});
  CHECK(fast.size() < baseline.size());
  // Piper uses stochastic noise in the vocoder; sample counts are not stable run-to-run at fixed speed.
  (void)tts.synthesize(text);
}
