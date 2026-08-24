// Loads every published tiny streaming speech-to-text model and runs a short
// transcription. This is the test that catches a language whose tokenizer,
// streaming_config, or quantized graphs cannot actually decode, which the
// catalog and download-manifest tests never exercise.
//
// Fixtures live under test-assets/<model>/, populated by
// scripts/fetch-voice-assets.sh. Adding a streaming language to the catalog
// without fetching its tiny weights fails here, which is the point.

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "debug-utils.h"
#include "moonshine-c-api.h"
#include "moonshine-model-catalog.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

bool is_tiny_streaming(int32_t model_arch) {
  return model_arch == MOONSHINE_MODEL_ARCH_TINY_STREAMING;
}

// Catalog download URLs are ``.../model/<dir>/<quantized_...>``. The local
// fixture is the ``<dir>`` segment, matching fetch-voice-assets.sh.
std::string fixture_dir_from_download_url(const std::string &url) {
  const std::string marker = "/model/";
  const auto pos = url.find(marker);
  if (pos == std::string::npos) {
    return {};
  }
  const auto start = pos + marker.size();
  const auto slash = url.find('/', start);
  if (slash == std::string::npos) {
    return url.substr(start);
  }
  return url.substr(start, slash - start);
}

struct StreamingSmokeCase {
  std::string label;
  std::string language;
  std::string model_dir;
  int32_t model_arch;
};

std::vector<StreamingSmokeCase> tiny_streaming_cases() {
  std::vector<StreamingSmokeCase> cases;
  for (const moonshine::SttCatalogLanguage &lang :
       moonshine::stt_catalog_listing()) {
    for (const moonshine::SttCatalogModel &model : lang.models) {
      if (!is_tiny_streaming(model.model_arch)) {
        continue;
      }
      StreamingSmokeCase smoke;
      smoke.language = lang.code;
      smoke.model_dir = fixture_dir_from_download_url(model.download_url);
      smoke.model_arch = model.model_arch;
      smoke.label = smoke.model_dir.empty() ? lang.code : smoke.model_dir;
      cases.push_back(std::move(smoke));
    }
  }
  return cases;
}

std::string transcript_text(const transcript_t *transcript) {
  if (transcript == nullptr) {
    return {};
  }
  std::string text;
  for (uint64_t i = 0; i < transcript->line_count; ++i) {
    const char *line = transcript->lines[i].text;
    if (line == nullptr || line[0] == '\0') {
      continue;
    }
    if (!text.empty()) {
      text += ' ';
    }
    text += line;
  }
  return text;
}

bool load_short_clip(std::vector<float> *audio, int32_t *sample_rate) {
  const char *candidates[] = {"two_cities_16k.wav", "beckett.wav",
                              "two_cities.wav"};
  for (const char *path : candidates) {
    if (!std::filesystem::exists(path)) {
      continue;
    }
    float *data = nullptr;
    size_t count = 0;
    int32_t rate = 0;
    if (!load_wav_data(path, &data, &count, &rate)) {
      continue;
    }
    // load_wav_data hands back a raw C-allocated buffer; adopt it so every
    // path releases it without a bare deallocation call (see STYLE_GUIDE.md).
    const std::unique_ptr<float, decltype(&std::free)> owned(data, &std::free);
    if (data == nullptr || count == 0 || rate <= 0) {
      continue;
    }
    // Two seconds is enough to exercise encoder + decoder without turning
    // this into a WER test.
    const size_t max_samples = static_cast<size_t>(rate) * 2;
    const size_t used = count < max_samples ? count : max_samples;
    audio->assign(data, data + used);
    *sample_rate = rate;
    return true;
  }
  return false;
}

}  // namespace

TEST_CASE("streaming-language-smoke") {
  const std::vector<StreamingSmokeCase> cases = tiny_streaming_cases();
  // ar, de, en, es, ja, tl, vi, zh. A smaller list means a language was
  // dropped from the catalog or the filter above is wrong.
  REQUIRE(cases.size() >= 8);

  std::vector<float> audio;
  int32_t sample_rate = 0;
  REQUIRE(load_short_clip(&audio, &sample_rate));
  REQUIRE_FALSE(audio.empty());

  for (const StreamingSmokeCase &smoke : cases) {
    SUBCASE(smoke.label.c_str()) {
      REQUIRE_FALSE(smoke.model_dir.empty());
      REQUIRE(std::filesystem::exists(smoke.model_dir));
      REQUIRE(std::filesystem::exists(smoke.model_dir + "/encoder.ort"));
      REQUIRE(std::filesystem::exists(smoke.model_dir + "/tokenizer.bin"));
      REQUIRE(
          std::filesystem::exists(smoke.model_dir + "/streaming_config.json"));

      const moonshine_option_t options[] = {
          {"vad_threshold", "0"},
          {"return_audio_data", "false"},
          {"max_tokens_per_second", "13"},
      };
      const int32_t handle = moonshine_load_transcriber_from_files(
          smoke.model_dir.c_str(), static_cast<uint32_t>(smoke.model_arch),
          options, sizeof(options) / sizeof(options[0]),
          MOONSHINE_HEADER_VERSION);
      REQUIRE(handle >= 0);

      transcript_t *transcript = nullptr;
      const int32_t err = moonshine_transcribe_without_streaming(
          handle, audio.data(), audio.size(), sample_rate, 0, &transcript);
      const std::string text = transcript_text(transcript);
      LOGF("%s (%s): \"%s\"", smoke.label.c_str(), smoke.language.c_str(),
           text.c_str());
      CHECK(err == MOONSHINE_ERROR_NONE);
      REQUIRE(transcript != nullptr);
      REQUIRE(transcript->line_count > 0);
      if (smoke.language == "en") {
        // English audio through the English model must produce some text,
        // so a silent no-op decode is a failure rather than a pass.
        CHECK_FALSE(text.empty());
      }
      moonshine_free_transcriber(handle);
    }
  }
}
