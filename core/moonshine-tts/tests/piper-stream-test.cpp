#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "moonshine-tts.h"
#include "rule-g2p-test-support.h"

using namespace moonshine_tts;
namespace r = moonshine_tts::rule_g2p_test;

namespace {

constexpr const char* kStem = "en_US-amy-medium";
/// A voice with more than one speaker. Its body hands the generator a speaker
/// embedding beside the latent, and that embedding has to reach every chunk.
constexpr const char* kMultiSpeakerStem = "en_US-libritts_r-medium";

/// Whether a voice is installed with its two stages.
bool staged_voice_present(const std::filesystem::path& root,
                          const std::string& stem) {
  namespace fs = std::filesystem;
  const fs::path voices = root / "en_us" / "piper-voices";
  const auto stage = [&](const std::string& name) {
    return fs::is_regular_file(voices / (stem + name + ".model.ort")) ||
           fs::is_regular_file(voices / (stem + name + ".ort"));
  };
  return stage(".upstream") && stage(".generator");
}

/// A synthesizer whose output does not change between identical calls.
///
/// Piper draws noise for its durations and for the flow, so two renders of one
/// sentence differ in both length and samples. Zeroing both scales makes the
/// model deterministic, which is what lets a test compare one path against
/// another sample by sample. Nothing about chunking depends on the noise.
MoonshineTTS make_tts(const std::filesystem::path& root,
                      const std::string& stem = kStem) {
  MoonshineTTSOptions options;
  options.g2p_options.g2p_root = root;
  // The ``piper_`` prefix is what picks the engine; without it the name is
  // read as a Kokoro voice and the request quietly lands somewhere else.
  options.voice = "piper_" + stem;
  options.piper_noise_scale_override = 0.f;
  options.piper_noise_w_override = 0.f;
  // Peak normalization scales by the finished utterance's own peak, which
  // streaming never sees, so leave it off and compare the raw levels.
  options.normalize_audio = false;
  return MoonshineTTS("en_us", options);
}

std::vector<float> stream_all(MoonshineTTS& tts, const std::string& text,
                              int* chunks_out) {
  tts.push_text(text);
  tts.end_input();
  std::vector<float> joined;
  int chunks = 0;
  TtsChunk chunk;
  while (tts.next_chunk(chunk) == TtsStreamStatus::kChunk) {
    joined.insert(joined.end(), chunk.audio.begin(), chunk.audio.end());
    ++chunks;
  }
  if (chunks_out != nullptr) {
    *chunks_out = chunks;
  }
  return joined;
}

/// Error between two renders as a fraction of the signal, ignoring any length
/// difference of a sample or two at the very end.
double relative_error(const std::vector<float>& a,
                      const std::vector<float>& b) {
  const size_t count = std::min(a.size(), b.size());
  if (count == 0) {
    return 1.0;
  }
  double error = 0.0;
  double signal = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double difference = static_cast<double>(a[i]) - b[i];
    error += difference * difference;
    signal += static_cast<double>(b[i]) * b[i];
  }
  if (signal <= 0.0) {
    return 1.0;
  }
  return std::sqrt(error / signal);
}

}  // namespace

// The point of the Piper split: chunks joined end to end are the utterance the
// one-shot path produces. Kokoro can only manage this approximately, and needs
// a crossfade to cover the difference. Piper's generator, handed the same
// frames with padding around them, returns the same samples, so this is an
// equality test rather than a similarity one.
TEST_CASE("Piper streamed chunks reconstruct the one-shot render") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!staged_voice_present(root, kStem)) {
    WARN_MESSAGE(false, "skip: staged Piper voice not installed");
    return;
  }
  MoonshineTTS tts = make_tts(root);
  const std::string text =
      "I have moved your meeting to three o'clock on Thursday, and let "
      "everyone know about the change.";

  const std::vector<float> whole = tts.synthesize(text);
  REQUIRE(whole.size() > 24000u);

  int chunks = 0;
  const std::vector<float> streamed = stream_all(tts, text, &chunks);
  CHECK(chunks > 1);
  // Both paths resample the same native audio, so they can disagree by at most
  // the one sample that rounding at the end can add or drop.
  CHECK(streamed.size() >= whole.size() - 2);
  CHECK(streamed.size() <= whole.size() + 2);
  // What is left is the padding residual, about a millionth, and it does not
  // grow along the utterance: a drifting resample grid would show up here as
  // an error that climbs chunk by chunk.
  CHECK(relative_error(streamed, whole) < 1e-5);
}

// Each chunk has to be right where it lands, not merely right on average. A
// chunk placed a fraction of a sample off still sums to something close to the
// whole render, so checking them one at a time is what catches it.
TEST_CASE("Piper each chunk matches its own span of the one-shot render") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!staged_voice_present(root, kStem)) {
    return;
  }
  MoonshineTTS tts = make_tts(root);
  const std::string text =
      "The delivery is scheduled for Friday morning, some time between eight "
      "and eleven, and someone will need to sign for it.";
  const std::vector<float> whole = tts.synthesize(text);
  REQUIRE(whole.size() > 24000u);

  tts.push_text(text);
  tts.end_input();
  TtsChunk chunk;
  size_t at = 0;
  int checked = 0;
  while (tts.next_chunk(chunk) == TtsStreamStatus::kChunk) {
    const size_t begin = std::min(at, whole.size());
    const size_t end = std::min(at + chunk.audio.size(), whole.size());
    const std::vector<float> want(whole.begin() + static_cast<long>(begin),
                                  whole.begin() + static_cast<long>(end));
    INFO("chunk starting at sample " << at);
    CHECK(relative_error(chunk.audio, want) < 1e-4);
    at += chunk.audio.size();
    ++checked;
  }
  CHECK(checked > 1);
}

// A multi-speaker voice crosses two tensors between the stages rather than
// one, and the second is fixed for the utterance while the latent is not.
// Confusing the two is easy and quiet: the voice still loads, and the wrong
// tensor is the right shape often enough to render something.
TEST_CASE("Piper streams a multi-speaker voice") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!staged_voice_present(root, kMultiSpeakerStem)) {
    WARN_MESSAGE(false, "skip: staged multi-speaker Piper voice not installed");
    return;
  }
  MoonshineTTS tts = make_tts(root, kMultiSpeakerStem);
  const std::string text =
      "The library closes at six today, so anything you want to borrow needs "
      "to be checked out before then.";

  const std::vector<float> whole = tts.synthesize(text);
  REQUIRE(whole.size() > 24000u);
  int chunks = 0;
  const std::vector<float> streamed = stream_all(tts, text, &chunks);
  CHECK(chunks > 1);
  CHECK(streamed.size() >= whole.size() - 2);
  CHECK(streamed.size() <= whole.size() + 2);
  CHECK(relative_error(streamed, whole) < 1e-5);
}

// Every chunk after the first is produced while its predecessor plays, so what
// decides whether streaming holds up is that the first one is short. It also
// has to be long enough to be worth playing, which the policy's minimum
// guarantees.
TEST_CASE("Piper first chunk is short and later chunks grow") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!staged_voice_present(root, kStem)) {
    return;
  }
  MoonshineTTS tts = make_tts(root);
  tts.push_text(
      "The quarterly figures came in this morning, and revenue is up by "
      "about twelve percent on the same period last year.");
  tts.end_input();

  std::vector<size_t> sizes;
  TtsChunk chunk;
  while (tts.next_chunk(chunk) == TtsStreamStatus::kChunk) {
    sizes.push_back(chunk.audio.size());
  }
  REQUIRE(sizes.size() > 2);
  const double first_seconds =
      static_cast<double>(sizes.front()) / MoonshineTTS::kSampleRateHz;
  CHECK(first_seconds < 0.9);
  CHECK(first_seconds > 0.3);
  // Growing, except for the last chunk, which is whatever is left over.
  for (size_t i = 1; i + 1 < sizes.size(); ++i) {
    CHECK(sizes[i] > sizes[i - 1]);
  }
}

// A voice whose latent frames are not a whole number of output samples is the
// case that would silently drift: 22.05 kHz resampled to 24 kHz puts 278.6
// samples in a frame. Streaming a long utterance and checking the total length
// is what catches an accumulating rounding error.
TEST_CASE("Piper chunk lengths do not drift against the whole render") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!staged_voice_present(root, kStem)) {
    return;
  }
  MoonshineTTS tts = make_tts(root);
  const std::string text =
      "There were three of them waiting by the door, and none of them said a "
      "single word to us, which was strange, because we had been expecting an "
      "argument about the delivery times and about who was going to pay for "
      "the extra storage.";
  const std::vector<float> whole = tts.synthesize(text);
  const std::vector<float> streamed = stream_all(tts, text, nullptr);
  REQUIRE(whole.size() > 24000u);
  CHECK(streamed.size() >= whole.size() - 2);
  CHECK(streamed.size() <= whole.size() + 2);
}
