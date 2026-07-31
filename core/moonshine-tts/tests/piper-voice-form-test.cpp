#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "piper-voice-catalog.h"

#include <doctest/doctest.h>

#include <filesystem>
#include <set>
#include <string>
#include <vector>

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

// The catalog records whether a voice ships as a split ORT pair or a single
// ``.ort``, because a client with no files yet has to know what to download and
// cannot look at its own disk. That makes the catalog a duplicate of the data
// tree, so this test fails when the two drift apart -- for instance after
// ``scripts/convert-models-to-ort.py`` picks a different form for a new voice.
TEST_CASE("catalog voice form matches the shipped files") {
  const auto data_root = piper_data_root();
  if (data_root.empty()) {
    WARN_MESSAGE(false, "skip: core/moonshine-tts/data not found");
    return;
  }

  size_t checked = 0;
  for (const auto& voices : voice_dirs(data_root)) {
    std::set<std::string> stems;
    for (const auto& ent : std::filesystem::directory_iterator(voices)) {
      const std::string name = ent.path().filename().string();
      // Longest first: ".ort" would otherwise swallow the ".weights" in
      // "<stem>.weights.ort" and invent a voice named "<stem>.weights".
      for (const std::string suffix :
           {".weights.ort", ".model.ort", ".onnx", ".ort"}) {
        if (name.size() > suffix.size() &&
            name.compare(name.size() - suffix.size(), suffix.size(), suffix) ==
                0) {
          stems.insert(name.substr(0, name.size() - suffix.size()));
          break;
        }
      }
    }
    for (const std::string& stem : stems) {
      const bool split_on_disk =
          std::filesystem::is_regular_file(voices / (stem + ".model.ort")) &&
          std::filesystem::is_regular_file(voices / (stem + ".weights.ort"));
      const bool single_on_disk =
          std::filesystem::is_regular_file(voices / (stem + ".ort"));
      if (!split_on_disk && !single_on_disk) {
        // Still ONNX-only; the catalog says nothing useful about it yet.
        continue;
      }
      INFO("voice " << stem << " in " << voices.string());
      CHECK(piper_voice_ships_split(stem) == split_on_disk);
      for (const std::string& name : piper_voice_model_filenames(stem)) {
        INFO("expected file " << name);
        CHECK(std::filesystem::is_regular_file(voices / name));
      }
      ++checked;
    }
  }
  CHECK(checked > 0);
}
