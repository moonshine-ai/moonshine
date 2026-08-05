#include "moonshine-cpp.h"

#include <cinttypes>
#include <filesystem>
#include <fstream>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {
// Duplicate of load_wav_data in debug-utils.cpp to avoid depending on
// internal library code.
bool load_wav_data(const char *path, float **out_float_data,
                   size_t *out_num_samples, int32_t *out_sample_rate) {
  *out_float_data = nullptr;
  *out_num_samples = 0;

  // Open the file in binary mode
  FILE *file = std::fopen(path, "rb");
  if (!file) {
    std::perror("Failed to open WAV file");
    return false;
  }

  // Read the RIFF header
  char riff_header[4];
  if (std::fread(riff_header, 1, 4, file) != 4 ||
      std::strncmp(riff_header, "RIFF", 4) != 0) {
    std::fclose(file);
    std::fprintf(stderr, "Not a RIFF file\n");
    return false;
  }

  // Skip chunk size and check WAVE
  std::fseek(file, 4, SEEK_CUR);
  char wave_header[4];
  if (std::fread(wave_header, 1, 4, file) != 4 ||
      std::strncmp(wave_header, "WAVE", 4) != 0) {
    std::fclose(file);
    std::fprintf(stderr, "Not a WAVE file\n");
    return false;
  }

  // Find the "fmt " chunk
  char chunk_id[4];
  uint32_t chunk_size = 0;
  bool found_fmt = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
      found_fmt = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_fmt) {
    std::fclose(file);
    std::fprintf(stderr, "No fmt chunk found\n");
    return false;
  }

  // Read fmt chunk
  uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0, byte_rate = 0;
  uint16_t block_align = 0;
  if (chunk_size < 16) {
    std::fclose(file);
    std::fprintf(stderr, "fmt chunk too small\n");
    return false;
  }
  if (std::fread(&audio_format, sizeof(uint16_t), 1, file) != 1 ||
      std::fread(&num_channels, sizeof(uint16_t), 1, file) != 1 ||
      std::fread(&sample_rate, sizeof(uint32_t), 1, file) != 1 ||
      std::fread(&byte_rate, sizeof(uint32_t), 1, file) != 1 ||
      std::fread(&block_align, sizeof(uint16_t), 1, file) != 1 ||
      std::fread(&bits_per_sample, sizeof(uint16_t), 1, file) != 1) {
    std::fclose(file);
    std::fprintf(stderr, "Truncated fmt chunk\n");
    return false;
  }
  // Skip any extra fmt bytes
  if (chunk_size > 16) std::fseek(file, chunk_size - 16, SEEK_CUR);

  if (audio_format != 1 || bits_per_sample != 16) {
    std::fclose(file);
    std::fprintf(stderr, "Only 16-bit PCM WAV files are supported\n");
    return false;
  }

  // Find the "data" chunk
  bool found_data = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "data", 4) == 0) {
      found_data = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_data) {
    std::fclose(file);
    std::fprintf(stderr, "No data chunk found\n");
    return false;
  }

  // Read PCM data
  size_t num_samples = chunk_size / (bits_per_sample / 8);
  if (num_samples == 0) {
    std::fclose(file);
    std::fprintf(stderr, "No samples found\n");
    return false;
  }
  float *result_data = (float *)malloc(num_samples * sizeof(float));
  for (size_t i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    if (std::fread(&sample, sizeof(int16_t), 1, file) != 1) {
      num_samples = i;
      break;
    }
    result_data[i] = static_cast<float>(sample) / 32768.0f;
  }
  std::fclose(file);
  *out_float_data = result_data;
  *out_num_samples = num_samples;
  if (out_sample_rate != nullptr) {
    *out_sample_rate = sample_rate;
  }
  return true;
}

// Would use std::filesystem::exists, but it's not available in C++11.
bool file_exists(const std::string &path) {
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) {
    return false;
  }
  std::fclose(file);
  return true;
}

class TestListener : public moonshine::TranscriptEventListener {
 public:
  int started_count = 0;
  int updated_count = 0;
  int text_changed_count = 0;
  int completed_count = 0;
  void onLineStarted(const moonshine::LineStarted &) override {
    started_count++;
  }
  void onLineUpdated(const moonshine::LineUpdated &) override {
    updated_count++;
  }
  void onLineTextChanged(const moonshine::LineTextChanged &) override {
    text_changed_count++;
  }
  void onLineCompleted(const moonshine::LineCompleted &) override {
    completed_count++;
  }
};

}  // namespace

TEST_CASE("moonshine-cpp-test") {
  SUBCASE("transcribe-without-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(file_exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    moonshine::Transcriber transcriber(root_model_path,
                                       moonshine::ModelArch::TINY);
    moonshine::Transcript transcript = transcriber.transcribeWithoutStreaming(
        std::vector<float>(wav_data, wav_data + wav_data_size), wav_sample_rate,
        0);
    REQUIRE(transcript.lines.size() > 0);
    for (const auto &line : transcript.lines) {
      REQUIRE(line.audioData.size() > 0);
      REQUIRE(line.startTime >= 0.0f);
      REQUIRE(line.duration > 0.0f);
      REQUIRE(line.isComplete);
      REQUIRE(line.isUpdated);
      REQUIRE(line.isNew);
      REQUIRE(line.hasTextChanged);
      // Speaker identification is opt-in, so no spans by default.
      REQUIRE(line.speakerSpans.empty());
    }
  }
  SUBCASE("transcribe-with-streaming") {
    std::string wav_path = "two_cities.wav";
    REQUIRE(file_exists(wav_path));
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    REQUIRE(wav_data != nullptr);
    REQUIRE(wav_data_size > 0);
    std::string root_model_path = "tiny-en";
    moonshine::Transcriber transcriber(root_model_path,
                                       moonshine::ModelArch::TINY);

    TestListener listener;
    transcriber.addListener(&listener);

    transcriber.start();

    const float chunk_duration_seconds = 0.0451f;
    const size_t chunk_size =
        (size_t)(chunk_duration_seconds * wav_sample_rate);
    size_t samples_since_last_transcription = 0;
    const size_t samples_between_transcriptions =
        (size_t)(wav_sample_rate * 0.481f);
    size_t line_count = 0;
    std::set<uint64_t> existing_line_ids;
    for (size_t i = 0; i < wav_data_size; i += chunk_size) {
      const float *chunk_data = wav_data + i;
      const size_t chunk_data_size = std::min(chunk_size, wav_data_size - i);
      transcriber.addAudio(
          std::vector<float>(chunk_data, chunk_data + chunk_data_size),
          wav_sample_rate);
      samples_since_last_transcription += chunk_data_size;
      if (samples_since_last_transcription < samples_between_transcriptions) {
        continue;
      }
      samples_since_last_transcription = 0;
      moonshine::Transcript transcript = transcriber.updateTranscription(0);
      line_count = std::max(line_count, transcript.lines.size());
      bool any_updated_lines = false;
      size_t line_index = 0;
      size_t lines_size = transcript.lines.size();
      for (const auto &line : transcript.lines) {
        REQUIRE(line.audioData.size() > 0);
        REQUIRE(line.startTime >= 0.0f);
        REQUIRE(line.duration > 0.0f);

        // Make sure the line ID is unique and stable.
        const bool seen_id_before =
            existing_line_ids.find(line.lineId) != existing_line_ids.end();
        if (!seen_id_before) {
          existing_line_ids.insert(line.lineId);
        }
        REQUIRE(existing_line_ids.size() <= lines_size);

        // There should be at most one incomplete line at the end of the
        // transcript.
        if (!line.isComplete) {
          const bool is_last_line = (line_index == (lines_size - 1));
          if (!is_last_line) {
            fprintf(stderr,
                    "Incomplete line %" PRIu64
                    " ('%s', %.2fs) is not the last line "
                    "%zu\n",
                    line.lineId, line.text.c_str(), line.startTime,
                    lines_size - 1);
          }
          REQUIRE(is_last_line);
        }
        line_index++;

        if (line.isUpdated) {
          any_updated_lines = true;
        } else {
          // If an earlier line has been updated, then all later lines should
          // have been updated as well.
          REQUIRE(!any_updated_lines);
        }
        if (!line.isUpdated) {
          continue;
        }
        fprintf(stderr, "%.1f (#%" PRId64 "): %s\n", line.startTime,
                line.lineId, line.text.c_str());
      }
    }
    transcriber.stop();
    REQUIRE(line_count > 0);
    REQUIRE(listener.started_count > 0);
    REQUIRE(listener.updated_count > 0);
    REQUIRE(listener.text_changed_count > 0);
    REQUIRE(listener.completed_count > 0);
    REQUIRE(listener.started_count == listener.completed_count);
    REQUIRE(listener.updated_count >= listener.started_count);
  }
  SUBCASE("g2p") {
    std::string root_model_path = "../core/moonshine-tts/data/";
    // The English G2P lexicon lives under the large (git-LFS) TTS data tree,
    // which isn't always present -- e.g. the reliability box intentionally
    // skips syncing moonshine-tts/data. Match the sibling data-dependent
    // subcases and skip rather than throw when the lexicon is absent.
    if (!file_exists(root_model_path + "en_us/dict_filtered_heteronyms.tsv")) {
      MESSAGE("skip: en_us G2P lexicon not in moonshine-tts/data");
      return;
    }
    std::string text = "Hello! This is a test of the Moonshine text to speech.";
    moonshine::GraphemeToPhonemizer g2p(
        "en_us", {
                     {"g2p_root", root_model_path.c_str()},
                 });
    std::string ipa = g2p.toIpa(text);
    REQUIRE(ipa.size() > 10);
  }
  SUBCASE("embedding model invalid model path throws") {
    REQUIRE_THROWS_AS((void)moonshine::EmbeddingModel(
                          "/nonexistent/moonshine/embedding/model",
                          moonshine::EmbeddingModelArch::GEMMA_300M),
                      moonshine::MoonshineException);
  }
  SUBCASE("spelling-mode-replaces-line-text-via-cpp-ctor") {
    // Mirrors the C-API "spelling-mode-replaces-line-text" subcase but
    // exercised through the C++ wrapper's options-aware constructor +
    // ``Transcriber::FLAG_SPELLING_MODE`` flag, so we don't regress the
    // language binding.
    std::string spelling_path = "spelling_cnn.ort";
    std::string wav_path = "alphanumeric/a/petewarden_nohash_0.wav";
    std::string root_model_path = "tiny-en";
    if (!file_exists(spelling_path)) {
      MESSAGE("skip: spelling_cnn.ort not in test-assets");
      return;
    }
    if (!file_exists(wav_path)) {
      MESSAGE("skip: alphanumeric clip not in test-assets");
      return;
    }
    if (!file_exists(root_model_path + "/encoder_model.ort")) {
      MESSAGE("skip: tiny-en model not in test-assets");
      return;
    }
    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));
    moonshine::Transcriber transcriber(root_model_path,
                                       moonshine::ModelArch::TINY,
                                       /*updateInterval=*/0.5, spelling_path);
    moonshine::Transcript transcript = transcriber.transcribeWithoutStreaming(
        std::vector<float>(wav_data, wav_data + wav_data_size), wav_sample_rate,
        moonshine::Transcriber::FLAG_SPELLING_MODE);
    REQUIRE(transcript.lines.size() >= 1);
    CHECK(transcript.lines[0].text == "a");
    free(wav_data);
  }
  SUBCASE("loadFromMemory-with-spelling-buffer") {
    // Verifies the static factory signature and end-to-end behaviour:
    // we slurp the on-disk model + spelling .ort into vectors and hand
    // them to ``Transcriber::loadFromMemory``. The resulting
    // transcriber should still resolve the "alpha" clip to ``"a"`` when
    // ``FLAG_SPELLING_MODE`` is set.
    std::string root_model_path = "tiny-en";
    std::string wav_path = "alphanumeric/a/petewarden_nohash_0.wav";
    std::string spelling_path = "spelling_cnn.ort";
    if (!file_exists(spelling_path) ||
        !file_exists(root_model_path + "/encoder_model.ort") ||
        !file_exists(wav_path)) {
      MESSAGE("skip: spelling/alphanumeric assets missing");
      return;
    }
    auto slurp = [](const std::string &path) {
      std::ifstream f(path, std::ios::binary);
      return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
    };
    std::vector<uint8_t> encoder =
        slurp(root_model_path + "/encoder_model.ort");
    std::vector<uint8_t> decoder =
        slurp(root_model_path + "/decoder_model_merged.ort");
    std::vector<uint8_t> tokenizer = slurp(root_model_path + "/tokenizer.bin");
    std::vector<uint8_t> spelling = slurp(spelling_path);
    REQUIRE(encoder.size() > 0);
    REQUIRE(decoder.size() > 0);
    REQUIRE(tokenizer.size() > 0);
    REQUIRE(spelling.size() > 0);

    float *wav_data = nullptr;
    size_t wav_data_size = 0;
    int32_t wav_sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                          &wav_sample_rate));

    moonshine::Transcriber transcriber = moonshine::Transcriber::loadFromMemory(
        encoder.data(), encoder.size(), decoder.data(), decoder.size(),
        tokenizer.data(), tokenizer.size(), moonshine::ModelArch::TINY,
        /*updateInterval=*/0.5, spelling.data(), spelling.size());
    moonshine::Transcript transcript = transcriber.transcribeWithoutStreaming(
        std::vector<float>(wav_data, wav_data + wav_data_size), wav_sample_rate,
        moonshine::Transcriber::FLAG_SPELLING_MODE);
    REQUIRE(transcript.lines.size() >= 1);
    CHECK(transcript.lines[0].text == "a");
    free(wav_data);
  }
  SUBCASE("embedding model scores phrases when model present") {
    const std::string dir = "embeddinggemma-300m-ONNX";
    if (!file_exists(dir + "/model_q4.ort")) {
      return;
    }
    moonshine::EmbeddingModel model(dir,
                                    moonshine::EmbeddingModelArch::GEMMA_300M);
    const std::vector<float> phrase =
        model.calculateEmbedding("turn on the lights");
    REQUIRE(!phrase.empty());
    const std::vector<float> utterance =
        model.calculateEmbedding("switch on the lights");
    const std::vector<float> unrelated =
        model.calculateEmbedding("the stock market crashed");
    REQUIRE(model.distance(phrase, phrase) > 0.99f);
    REQUIRE(model.distance(phrase, utterance) >
            model.distance(phrase, unrelated));
  }
  SUBCASE("manifests name the files a caller has to download") {
    // The C++ library downloads nothing, so these manifests are the only way a
    // caller learns what to fetch. They are pure catalog lookups, so unlike
    // most of this file they need no model files present.
    const std::string stt = moonshine::Transcriber::getDependencies("en");
    CHECK(stt.find("\"groups\"") != std::string::npos);
    CHECK(stt.find("tokenizer.bin") != std::string::npos);

    const std::string diarization =
        moonshine::Transcriber::getDiarizationDependencies();
    CHECK(diarization.find("segmentation.ort") != std::string::npos);
    CHECK(diarization.find("embedding.ort") != std::string::npos);

    CHECK(moonshine::Transcriber::getCatalog().find("\"languages\"") !=
          std::string::npos);
    CHECK(moonshine::EmbeddingModel::getCatalog().find("\"models\"") !=
          std::string::npos);
    CHECK(moonshine::EmbeddingModel::getDependencies("embeddinggemma-300m")
              .find("\"groups\"") != std::string::npos);

    // An unknown language is an error rather than a silent empty manifest.
    REQUIRE_THROWS_AS(
        (void)moonshine::Transcriber::getDependencies("not-a-language"),
        moonshine::MoonshineException);
  }
  SUBCASE("speech clip extraction finds speech and rejects silence") {
    // The voice-activity detector is compiled in, so this needs no models.
    const std::string wav_path = "beckett.wav";
    if (!file_exists(wav_path)) {
      MESSAGE("skip: beckett.wav not in test-assets");
      return;
    }
    float *wav_data = nullptr;
    size_t sample_count = 0;
    int32_t sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &sample_count,
                          &sample_rate));
    const std::vector<float> audio(wav_data, wav_data + sample_count);
    free(wav_data);

    const std::string g2p_root = "../core/moonshine-tts/data/";
    if (!file_exists(g2p_root + "zipvoice/vocoder.ort")) {
      MESSAGE("skip: zipvoice assets not in moonshine-tts/data");
      return;
    }
    moonshine::TextToSpeech tts("en_us", {{"g2p_root", g2p_root}});
    const int32_t tts_handle = tts.getHandle();

    const moonshine::SpeechClip clip =
        moonshine::extractSpeechClip(audio, sample_rate, tts_handle);
    REQUIRE(clip.isComplete);
    CHECK(clip.speechDuration > 1.0f);
    CHECK(clip.audio.size() ==
          static_cast<size_t>(4 * moonshine::VoiceClone::CLIP_SAMPLE_RATE));

    const std::vector<float> silence(sample_rate * 5, 0.0f);
    CHECK_FALSE(moonshine::extractSpeechClip(silence, sample_rate, tts_handle)
                    .isComplete);
  }
  SUBCASE("VoiceClone becomes ready as audio arrives") {
    const std::string wav_path = "beckett.wav";
    if (!file_exists(wav_path)) {
      MESSAGE("skip: beckett.wav not in test-assets");
      return;
    }
    float *wav_data = nullptr;
    size_t sample_count = 0;
    int32_t sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &sample_count,
                          &sample_rate));
    const std::vector<float> audio(wav_data, wav_data + sample_count);
    free(wav_data);

    const std::string g2p_root = "../core/moonshine-tts/data/";
    if (!file_exists(g2p_root + "zipvoice/vocoder.ort")) {
      MESSAGE("skip: zipvoice assets not in moonshine-tts/data");
      return;
    }
    moonshine::TextToSpeech tts("en_us", {{"g2p_root", g2p_root}});

    moonshine::VoiceClone clone(tts.getHandle());
    int ready_calls = 0;
    clone.onReady([&ready_calls] { ready_calls++; });
    CHECK_FALSE(clone.isReady());

    // Feed it the way an audio callback would, a chunk at a time.
    const size_t chunk = static_cast<size_t>(sample_rate) / 10;
    for (size_t i = 0; i < audio.size() && !clone.isReady(); i += chunk) {
      const size_t end = std::min(i + chunk, audio.size());
      clone.addAudio(std::vector<float>(audio.begin() + i, audio.begin() + end),
                     sample_rate);
    }

    REQUIRE(clone.isReady());
    CHECK(ready_calls == 1);
    CHECK(clone.audio().size() ==
          static_cast<size_t>(4 * moonshine::VoiceClone::CLIP_SAMPLE_RATE));
    CHECK(clone.speechSeconds() > 0.0f);

    // A handler attached after the fact still fires, and reset undoes it all.
    int late_calls = 0;
    clone.onReady([&late_calls] { late_calls++; });
    CHECK(late_calls == 1);
    clone.reset();
    CHECK_FALSE(clone.isReady());
    CHECK(clone.audio().empty());
  }
  SUBCASE("cloneFrom rebuilds the synthesizer with a captured voice") {
    const std::string g2p_root = "../core/moonshine-tts/data/";
    const std::string wav_path = "beckett.wav";
    if (!file_exists(g2p_root + "zipvoice/vocoder.ort")) {
      MESSAGE("skip: zipvoice assets not in moonshine-tts/data");
      return;
    }
    if (!file_exists(wav_path)) {
      MESSAGE("skip: beckett.wav not in test-assets");
      return;
    }
    float *wav_data = nullptr;
    size_t sample_count = 0;
    int32_t sample_rate = 0;
    REQUIRE(load_wav_data(wav_path.c_str(), &wav_data, &sample_count,
                          &sample_rate));
    const std::vector<float> audio(wav_data, wav_data + sample_count);
    free(wav_data);

    moonshine::TextToSpeech tts("en_us", {{"g2p_root", g2p_root}});
    CHECK_FALSE(tts.isCloned());

    // The transcript is supplied here so the test needs no speech-to-text model.
    tts.cloneFrom(audio, sample_rate, "Ever tried. Ever failed. No matter.");
    CHECK(tts.isCloned());

    const moonshine::TtsSynthesisResult result =
        tts.synthesize("Cloning a custom voice.");
    CHECK(result.sampleRateHz > 0);
    CHECK(result.samples.size() > static_cast<size_t>(result.sampleRateHz) / 4);
  }
  SUBCASE("cloning from silence explains itself") {
    const std::string g2p_root = "../core/moonshine-tts/data/";
    if (!file_exists(g2p_root + "zipvoice/vocoder.ort")) {
      MESSAGE("skip: zipvoice assets not in moonshine-tts/data");
      return;
    }
    moonshine::TextToSpeech tts("en_us", {{"g2p_root", g2p_root}});
    const std::vector<float> silence(16000 * 5, 0.0f);
    REQUIRE_THROWS_AS(tts.cloneFrom(silence, 16000, "nothing was said"),
                      moonshine::MoonshineException);
    // The failed clone left the original synthesizer usable.
    CHECK_FALSE(tts.isCloned());
  }
  SUBCASE("VoiceClone copies share one capture") {
    // VoiceClone is a handle onto shared state, matching the reference-typed
    // VoiceClone in the Swift and Java bindings. A real TTS is unnecessary here
    // because this case never reaches extract.
    moonshine::VoiceClone clone(/*ttsSynthesizerHandle=*/-1);
    moonshine::VoiceClone alias = clone;
    const std::vector<float> silence(16000, 0.0f);
    alias.addAudio(silence, 16000);
    CHECK(clone.recordedSeconds() == doctest::Approx(1.0f));
    CHECK_FALSE(clone.isReady());
  }
}