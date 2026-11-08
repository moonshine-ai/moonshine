#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "phonon-tts.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "moonshine-tts-options.h"
#include "moonshine-tts.h"
#include "rule-g2p-test-support.h"

using namespace moonshine_tts;
namespace r = moonshine_tts::rule_g2p_test;

namespace {

bool phonon_data_present(const std::filesystem::path& root) {
  namespace fs = std::filesystem;
  for (const std::string_view key :
       {kTtsPhononEmbedPromptKey, kTtsPhononEmbedLatentKey,
        kTtsPhononBackboneKey, kTtsPhononFlowHeadKey,
        kTtsPhononMimiTransformerKey, kTtsPhononMimiSeanetKey,
        kTtsPhononModelJsonKey, kTtsPhononTokenizerKey,
        kTtsPhononVoicesJsonKey}) {
    if (!fs::is_regular_file(root / fs::path(key))) {
      return false;
    }
  }
  return true;
}

PhononTTSOptions base_options(const std::filesystem::path& root) {
  PhononTTSOptions opt;
  opt.g2p_options.g2p_root = root;
  opt.voice_id = "zoey";
  opt.temperature = 0.F;
  return opt;
}

float rms(const std::vector<float>& x) {
  double ss = 0.0;
  for (const float v : x) {
    ss += static_cast<double>(v) * static_cast<double>(v);
  }
  return x.empty() ? 0.F : static_cast<float>(std::sqrt(ss / x.size()));
}

/// Little-endian float32 samples, as ``scripts/export-phonon-model.py`` writes
/// its ``reference.f32`` next to the graphs.
std::vector<float> read_f32(const std::filesystem::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
  std::vector<float> out(bytes.size() / sizeof(float));
  std::memcpy(out.data(), bytes.data(), out.size() * sizeof(float));
  return out;
}

}  // namespace

TEST_CASE("PhononTTS lists the catalog voices") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  const std::vector<PhononVoiceInfo> voices =
      PhononTTS::list_voices(root, FileInformationMap{});
  REQUIRE(!voices.empty());
  bool found = false;
  for (const PhononVoiceInfo& v : voices) {
    CHECK(!v.id.empty());
    CHECK(!v.file.empty());
    CHECK(std::filesystem::is_regular_file(root / "phonon" / v.file));
    found = found || v.id == "zoey";
  }
  CHECK(found);

  // The compiled-in id list stands in for voices.json where that file is not
  // available (nothing fetched yet, or the WASM binding), so it has to agree
  // with the shipped file exactly.
  std::vector<std::string> from_json;
  for (const PhononVoiceInfo& v : voices) {
    from_json.push_back(v.id);
    CHECK(v.file == "voices/" + v.id + ".safetensors");
  }
  std::vector<std::string> builtin = PhononTTS::builtin_voice_ids();
  std::sort(from_json.begin(), from_json.end());
  std::sort(builtin.begin(), builtin.end());
  CHECK(builtin == from_json);
}

TEST_CASE("PhononTTS takes a catalog voice latent supplied in memory") {
  // The WASM binding hands every asset over as bytes under its canonical key
  // and has no filesystem to fall back to, so a catalog voice has to resolve
  // through the in-memory map rather than a path beside voices.json.
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  std::ifstream f(root / "phonon" / "voices" / "zoey.safetensors",
                  std::ios::binary);
  const std::vector<uint8_t> latent((std::istreambuf_iterator<char>(f)),
                                    std::istreambuf_iterator<char>());
  REQUIRE(!latent.empty());

  PhononTTSOptions opt = base_options(root);
  // Point the disk lookup somewhere empty so only the in-memory copy can
  // satisfy it; the graphs still come from the data directory by path.
  for (const std::string_view key :
       {kTtsPhononEmbedPromptKey, kTtsPhononEmbedLatentKey,
        kTtsPhononBackboneKey, kTtsPhononFlowHeadKey,
        kTtsPhononMimiTransformerKey, kTtsPhononMimiSeanetKey,
        kTtsPhononModelJsonKey, kTtsPhononTokenizerKey,
        kTtsPhononVoicesJsonKey}) {
    opt.tts_asset_files.set_path(
        key, std::filesystem::absolute(root / std::filesystem::path(key)));
  }
  opt.tts_asset_files.set_memory("phonon/voices/zoey.safetensors",
                                 latent.data(), latent.size());
  opt.g2p_options.g2p_root = std::filesystem::temp_directory_path();

  PhononTTS tts(opt);
  const std::vector<float> a = tts.synthesize("Hello from memory.");
  CHECK(a.size() > 0.5 * PhononTTS::kSampleRateHz);
  CHECK(rms(a) > 0.01F);
}

TEST_CASE(
    "PhononTTS synthesizes speech deterministically and matches the "
    "reference render") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  PhononTTS tts(base_options(root));
  const std::string text = "The birch canoe slid on the smooth planks.";
  const std::vector<float> a = tts.synthesize(text);
  REQUIRE(a.size() % static_cast<size_t>(PhononTTS::kFrameSamples) == 0);
  // About 2.7 s for this sentence; anything from 1.5 to 5 s is sane.
  CHECK(a.size() > 1.5 * PhononTTS::kSampleRateHz);
  CHECK(a.size() < 5.0 * PhononTTS::kSampleRateHz);
  CHECK(rms(a) > 0.01F);
  for (const float v : a) {
    CHECK(std::isfinite(v));
    CHECK(v >= -1.F);
    CHECK(v <= 1.F);
  }
  // Temperature 0 is deterministic.
  const std::vector<float> b = tts.synthesize(text);
  REQUIRE(b.size() == a.size());
  CHECK(b == a);

  // The exporter leaves the PyTorch render of this sentence at temperature 0
  // beside the graphs. ORT and PyTorch differ at the 1e-4 level per frame and
  // the autoregression compounds that, so compare loosely.
  const std::filesystem::path ref = root / "phonon" / "reference.f32";
  if (std::filesystem::is_regular_file(ref)) {
    const std::vector<float> want = read_f32(ref);
    CHECK(want.size() == a.size());
    if (want.size() == a.size()) {
      double err = 0.0;
      double scale = 0.0;
      for (size_t i = 0; i < a.size(); ++i) {
        err += std::fabs(static_cast<double>(a[i]) - want[i]);
        scale += std::fabs(static_cast<double>(want[i]));
      }
      CHECK(err / scale < 0.05);
    }
  }
}

TEST_CASE("PhononTTS streams frame by frame to the same samples") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  PhononTTS tts(base_options(root));
  const std::string text = "Streaming should sound like the whole render.";
  const std::vector<float> whole = tts.synthesize(text);

  std::vector<float> streamed;
  for (const std::string& prompt : tts.plan_prompts(text)) {
    std::unique_ptr<PhononTTS::Generation> gen = tts.begin(prompt);
    while (!gen->done()) {
      const std::vector<float> frame = gen->next_frame();
      REQUIRE(frame.size() == static_cast<size_t>(PhononTTS::kFrameSamples));
      streamed.insert(streamed.end(), frame.begin(), frame.end());
    }
    CHECK(gen->next_frame().empty());
  }
  REQUIRE(streamed.size() == whole.size());
  CHECK(streamed == whole);
}

TEST_CASE("PhononTTS duration conditioning moves the length") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  PhononTTS tts(base_options(root));
  const std::string text =
      "Speaking rate is conditioned rather than stretched, so the model "
      "takes its time or hurries.";
  const std::vector<float> normal = tts.synthesize(text);
  tts.set_speed(1.3);
  const std::vector<float> fast = tts.synthesize(text);
  tts.set_speed(0.75);
  const std::vector<float> slow = tts.synthesize(text);
  CHECK(fast.size() < normal.size());
  CHECK(slow.size() > normal.size());
}

TEST_CASE("PhononTTS groups long text into several prompts") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  PhononTTS tts(base_options(root));
  std::string text;
  for (int i = 0; i < 12; ++i) {
    text += "This is sentence number " + std::to_string(i + 1) +
            " of a longer passage. ";
  }
  const std::vector<std::string> prompts = tts.plan_prompts(text);
  CHECK(prompts.size() >= 2);
  // Normalization applied: numbers under a thousand stay, typography folds.
  const std::vector<std::string> folded =
      tts.plan_prompts("“Quoted” text — with 1500 items.");
  REQUIRE(folded.size() == 1);
  CHECK(folded[0] == "\"Quoted\" text with 1 thousand 500 items.");
}

TEST_CASE("MoonshineTTS selects Phonon from the voice prefix and streams it") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  MoonshineTTSOptions opt;
  opt.g2p_options.g2p_root = root;
  opt.voice = "phonon_zoey";
  opt.normalize_audio = false;
  opt.phonon_temperature = 0.F;
  opt.stream_first_chunk_seconds = 0.4F;
  MoonshineTTS tts("en_us", opt);
  const std::string text =
      "First a short sentence. Then a second one that runs a little longer "
      "so the growing chunks have something to grow into.";
  // The stream speaks a sentence at a time, so the render to match is each
  // sentence on its own (one prompt would group them and come out differently).
  std::vector<float> whole;
  for (const std::string& sentence : tts.split_utterances(text)) {
    const std::vector<float> part = tts.synthesize(sentence);
    whole.insert(whole.end(), part.begin(), part.end());
  }
  REQUIRE(whole.size() > static_cast<size_t>(PhononTTS::kSampleRateHz));

  tts.push_text(text);
  tts.end_input();
  std::vector<float> streamed;
  int chunks = 0;
  int finals = 0;
  size_t first_chunk = 0;
  for (;;) {
    TtsChunk chunk;
    const TtsStreamStatus status = tts.next_chunk(chunk);
    if (status == TtsStreamStatus::kEndOfStream) {
      break;
    }
    REQUIRE(status == TtsStreamStatus::kChunk);
    CHECK(chunk.sample_rate == PhononTTS::kSampleRateHz);
    CHECK(!chunk.audio.empty());
    if (chunks == 0) {
      first_chunk = chunk.audio.size();
    }
    streamed.insert(streamed.end(), chunk.audio.begin(), chunk.audio.end());
    ++chunks;
    finals += chunk.is_final ? 1 : 0;
  }
  // Two sentences, each in more than one chunk; the first chunk is short.
  CHECK(chunks >= 4);
  CHECK(finals == 2);
  CHECK(first_chunk <= 0.5 * PhononTTS::kSampleRateHz);
  // Frame-exact streaming reproduces the one-shot render.
  REQUIRE(streamed.size() == whole.size());
  CHECK(streamed == whole);

  const std::vector<MoonshineTtsVoiceAvailability> voices =
      moonshine_list_tts_voices_with_availability("en_us", opt);
  bool listed = false;
  for (const MoonshineTtsVoiceAvailability& v : voices) {
    if (v.id == "phonon_zoey") {
      listed = true;
      CHECK(v.available);
    }
  }
  CHECK(listed);
}

TEST_CASE("PhononTTS rejects a missing voice and a bad speed") {
  const std::filesystem::path root =
      r::moonshine_tts_bundled_data_dir_relative();
  if (!phonon_data_present(root)) {
    return;
  }
  PhononTTSOptions opt = base_options(root);
  opt.voice_id = "not_a_voice";
  CHECK_THROWS(PhononTTS{opt});
  opt = base_options(root);
  opt.speed = 0.0;
  CHECK_THROWS(PhononTTS{opt});
}
