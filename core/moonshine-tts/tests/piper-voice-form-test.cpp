#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "piper-voice-catalog.h"

using namespace moonshine_tts;

namespace {

std::filesystem::path piper_data_root() {
  // Tests run from the monorepo root (see moonshine_tts_add_test).
  const std::filesystem::path root("core/moonshine-tts/data");
  return std::filesystem::is_directory(root) ? root : std::filesystem::path{};
}

std::vector<std::filesystem::path> voice_dirs(
    const std::filesystem::path& data_root) {
  std::vector<std::filesystem::path> dirs;
  for (const auto& lang : std::filesystem::directory_iterator(data_root)) {
    const auto voices = lang.path() / "piper-voices";
    if (std::filesystem::is_directory(voices)) {
      dirs.push_back(voices);
    }
  }
  return dirs;
}

}  // namespace

// The catalog records what a voice ships as, because a client with no files
// yet has to know what to download and cannot look at its own disk. That makes
// the catalog a duplicate of the data tree, so this test fails when the two
// drift apart -- for instance after ``scripts/convert-models-to-ort.py`` picks
// a different form for a new voice.
TEST_CASE("catalog voice form matches the shipped files") {
  const auto data_root = piper_data_root();
  if (data_root.empty()) {
    WARN_MESSAGE(false, "skip: core/moonshine-tts/data not found");
    return;
  }

  size_t checked = 0;
  for (const auto& voices : voice_dirs(data_root)) {
    // A voice is named by its config, which keeps the same name whatever form
    // the models take.
    for (const auto& ent : std::filesystem::directory_iterator(voices)) {
      const std::string name = ent.path().filename().string();
      static constexpr std::string_view kConfig = ".onnx.json";
      if (name.size() <= kConfig.size() ||
          name.compare(name.size() - kConfig.size(), kConfig.size(), kConfig) !=
              0) {
        continue;
      }
      const std::string stem = name.substr(0, name.size() - kConfig.size());
      const bool split_on_disk = std::filesystem::is_regular_file(
          voices / (stem + ".upstream.model.ort"));
      const bool single_on_disk =
          std::filesystem::is_regular_file(voices / (stem + ".upstream.ort"));
      if (!split_on_disk && !single_on_disk) {
        // Not built yet; the catalog says nothing useful about it.
        continue;
      }
      INFO("voice " << stem << " in " << voices.string());
      CHECK(piper_voice_ships_split(stem) == split_on_disk);
      for (const std::string& filename : piper_voice_model_filenames(stem)) {
        INFO("expected file " << filename);
        CHECK(std::filesystem::is_regular_file(voices / filename));
      }
      ++checked;
    }
  }
  CHECK(checked > 0);
}

// A voice used to ship as one model, and clients cache what they download. The
// stages replace it rather than joining it, so nothing should still be asking
// for the old files.
TEST_CASE("no voice still asks for a whole-utterance model") {
  const auto data_root = piper_data_root();
  if (data_root.empty()) {
    return;
  }
  for (const auto& voices : voice_dirs(data_root)) {
    for (const auto& ent : std::filesystem::directory_iterator(voices)) {
      const std::string name = ent.path().filename().string();
      const bool belongs_to_a_stage =
          name.find(".upstream.") != std::string::npos ||
          name.find(".generator.") != std::string::npos ||
          name.find(".onnx.json") != std::string::npos;
      INFO("file " << name);
      CHECK(belongs_to_a_stage);
    }
  }
}
