#include "spelling-model.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "debug-utils.h"
#include "spelling-fusion.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

// Locate the bundled spelling model. CMake copies the test runner into
// the build dir alongside ``test-assets/spelling_cnn.ort`` (see
// ``copy_test_assets`` in ``CMakeLists.txt``); we also accept paths
// relative to the workspace root for ad-hoc invocation outside ctest.
std::string find_model_path() {
  static const std::vector<std::string> candidates = {
      "spelling_cnn.ort",
      "test-assets/spelling_cnn.ort",
      "../../test-assets/spelling_cnn.ort",
  };
  for (const auto &p : candidates) {
    if (std::filesystem::exists(p)) return p;
  }
  return {};
}

std::string find_wav(const std::string &label, const std::string &filename) {
  static const std::vector<std::string> roots = {
      "alphanumeric",
      "test-assets/alphanumeric",
      "../../test-assets/alphanumeric",
  };
  for (const auto &root : roots) {
    std::string path = root + "/" + label + "/" + filename;
    if (std::filesystem::exists(path)) return path;
  }
  return {};
}

// Read entire file into a byte buffer.
std::vector<uint8_t> read_file(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return {};
  std::streamsize size = stream.tellg();
  if (size <= 0) return {};
  stream.seekg(0, std::ios::beg);
  std::vector<uint8_t> buffer(static_cast<size_t>(size));
  if (!stream.read(reinterpret_cast<char *>(buffer.data()), size)) return {};
  return buffer;
}

}  // namespace

TEST_CASE("spelling-model: load from path") {
  std::string path = find_model_path();
  if (path.empty()) {
    MESSAGE("spelling_cnn.ort not found; skipping");
    return;
  }
  SpellingModel model;
  REQUIRE(model.load(path.c_str()) == 0);
  CHECK(model.sample_rate() == 16000);
  CHECK(model.clip_seconds() == doctest::Approx(1.0f));
  CHECK(model.classes().size() == 36);
}

TEST_CASE("spelling-model: load from memory") {
  std::string path = find_model_path();
  if (path.empty()) {
    MESSAGE("spelling_cnn.ort not found; skipping");
    return;
  }
  std::vector<uint8_t> data = read_file(path);
  REQUIRE_FALSE(data.empty());
  SpellingModel model;
  REQUIRE(model.load_from_memory(data.data(), data.size()) == 0);
  CHECK(model.sample_rate() == 16000);
}

TEST_CASE("spelling-model: predict on bundled clips") {
  std::string path = find_model_path();
  if (path.empty()) {
    MESSAGE("spelling_cnn.ort not found; skipping");
    return;
  }
  SpellingModel model;
  REQUIRE(model.load(path.c_str()) == 0);

  // (label_dir, expected canonical char). Each label dir contains a
  // handful of speaker recordings; we only need the model to top-1 the
  // expected character on at least the petewarden clip, which is the
  // cleanest reference recording.
  struct Clip {
    const char *label;
    const char *expected_char;
  };
  const std::vector<Clip> clips = {
      {"a", "a"},  {"b", "b"},     {"c", "c"},   {"five", "5"},
      {"nine", "9"}, {"zero", "0"},
  };
  size_t correct = 0;
  size_t evaluated = 0;
  for (const auto &clip : clips) {
    std::string wav = find_wav(clip.label, "petewarden_nohash_0.wav");
    if (wav.empty()) continue;
    float *audio = nullptr;
    size_t audio_size = 0;
    int32_t sr = 0;
    REQUIRE(load_wav_data(wav.c_str(), &audio, &audio_size, &sr) == true);
    REQUIRE(audio != nullptr);
    REQUIRE(sr == 16000);
    SpellingPrediction prediction;
    REQUIRE(model.predict(audio, audio_size, sr, &prediction) == 0);
    free(audio);
    ++evaluated;
    if (prediction.character == clip.expected_char) ++correct;
    INFO("clip=" << clip.label << " predicted=" << prediction.character
                 << " p=" << prediction.probability);
  }
  REQUIRE(evaluated >= 3);
  // Allow at most one miss in the smoke test — the model is well
  // above 95 % top-1 on the People's Speech eval, but speaker
  // recordings can occasionally trip it up on a single clip.
  CHECK(correct + 1 >= evaluated);
}

TEST_CASE("spelling-model: invalid arguments are rejected") {
  std::string path = find_model_path();
  if (path.empty()) {
    MESSAGE("spelling_cnn.ort not found; skipping");
    return;
  }
  SpellingModel model;
  REQUIRE(model.load(path.c_str()) == 0);
  SpellingPrediction prediction;
  // Wrong sample rate.
  std::vector<float> dummy(16000, 0.0f);
  CHECK(model.predict(dummy.data(), dummy.size(), 8000, &prediction) != 0);
  // Null pointer / zero size.
  CHECK(model.predict(nullptr, 0, 16000, &prediction) != 0);
  CHECK(model.predict(dummy.data(), dummy.size(), 16000, nullptr) != 0);
}
