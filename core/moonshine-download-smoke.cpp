// moonshine-download-smoke: a tiny CLI used by scripts/test-model-downloads.sh
// to verify that the native download manifests are complete. It has two modes:
//
//   moonshine-download-smoke manifest <modality> [spec...]
//       Prints one "<url>\t<relative_path>" line per required file, resolved
//       from the C API dependency functions (the single source of truth). The
//       bash harness downloads each url into <root>/<relative_path>.
//
//   moonshine-download-smoke run <modality> <root> [spec...]
//       Loads the engine from the freshly downloaded <root> and runs one
//       trivial inference, exiting 0 on success. If a required file was missing
//       from the manifest, loading or inference fails here - which is exactly
//       what this test is meant to catch.
//
// Modalities and their [spec...] arguments:
//   stt    <language> [<model_arch>]
//   intent <model_name> <variant>
//   tts    <language> <voice>
//   g2p    <language>
//
// HTTP lives entirely in the bash harness (curl); this tool never touches the
// network. It only resolves manifests and exercises the on-disk load path.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.h>

#include "debug-utils.h"
#include "moonshine-c-api.h"

namespace {

// Canonical TTS/G2P asset keys are relative to this CDN tree (mirrors
// TTS_CDN_BASE_URL in python/src/moonshine_voice/download.py).
constexpr const char* kTtsCdnBase = "https://download.moonshine.ai/tts/";

void print_usage() {
  std::cerr
      << "Usage:\n"
      << "  moonshine-download-smoke manifest <stt|intent|tts|g2p> [spec...]\n"
      << "  moonshine-download-smoke run <stt|intent|tts|g2p> <root> "
         "[spec...]\n";
}

std::string url_encode_path(const std::string& key) {
  // Encode per path segment, leaving '/' as a separator (matches the Python
  // downloader's cdn_url_for_tts_asset_key).
  static const char* hex = "0123456789ABCDEF";
  std::string out;
  for (unsigned char c : key) {
    const bool unreserved = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                            (c >= '0' && c <= '9') || c == '-' || c == '_' ||
                            c == '.' || c == '~' || c == '/';
    if (unreserved) {
      out.push_back(static_cast<char>(c));
    } else {
      out.push_back('%');
      out.push_back(hex[(c >> 4) & 0xF]);
      out.push_back(hex[c & 0xF]);
    }
  }
  return out;
}

int fail(const std::string& message) {
  std::cerr << "moonshine-download-smoke: " << message << "\n";
  return 1;
}

// ------------------------------ manifest mode ------------------------------

// Emits "<url>\t<relative_path>" for every file in a {"groups":[...]} manifest
// produced by moonshine_get_stt_dependencies / moonshine_get_intent_dependencies.
void print_group_manifest(const std::string& json_text) {
  const nlohmann::json parsed = nlohmann::json::parse(json_text);
  for (const auto& group : parsed.at("groups")) {
    const std::string base_url = group.at("base_url").get<std::string>();
    for (const auto& file : group.at("files")) {
      const std::string name = file.get<std::string>();
      std::cout << base_url << "/" << name << "\t" << name << "\n";
    }
  }
}

int manifest_stt(const std::vector<std::string>& spec) {
  if (spec.empty()) {
    return fail("stt manifest needs <language> [<model_arch>]");
  }
  std::vector<moonshine_option_t> options;
  std::string arch_value;
  if (spec.size() >= 2) {
    arch_value = spec[1];
    options.push_back({"model_arch", arch_value.c_str()});
  }
  char* out = nullptr;
  const int32_t err = moonshine_get_stt_dependencies(
      spec[0].c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out);
  if (err != MOONSHINE_ERROR_NONE || out == nullptr) {
    return fail("moonshine_get_stt_dependencies failed");
  }
  print_group_manifest(out);
  std::free(out);
  return 0;
}

int manifest_intent(const std::vector<std::string>& spec) {
  const std::string model_name = spec.empty() ? "embeddinggemma-300m" : spec[0];
  std::vector<moonshine_option_t> options;
  std::string variant_value;
  if (spec.size() >= 2) {
    variant_value = spec[1];
    options.push_back({"variant", variant_value.c_str()});
  }
  char* out = nullptr;
  const int32_t err = moonshine_get_intent_dependencies(
      model_name.c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out);
  if (err != MOONSHINE_ERROR_NONE || out == nullptr) {
    return fail("moonshine_get_intent_dependencies failed");
  }
  print_group_manifest(out);
  std::free(out);
  return 0;
}

int manifest_tts(const std::vector<std::string>& spec) {
  if (spec.empty()) {
    return fail("tts manifest needs <language> [<voice>]");
  }
  std::vector<moonshine_option_t> options;
  std::string voice_value;
  if (spec.size() >= 2) {
    voice_value = spec[1];
    options.push_back({"voice", voice_value.c_str()});
  }
  char* out = nullptr;
  const int32_t err = moonshine_get_tts_dependencies(
      spec[0].c_str(), options.empty() ? nullptr : options.data(),
      options.size(), &out);
  if (err != MOONSHINE_ERROR_NONE || out == nullptr) {
    return fail("moonshine_get_tts_dependencies failed");
  }
  const nlohmann::json keys = nlohmann::json::parse(out);
  std::free(out);
  for (const auto& entry : keys) {
    const std::string key = entry.get<std::string>();
    // Skip in-memory override labels (no path component), like the Python
    // downloader's is_downloadable_tts_asset_key.
    if (key.find('/') == std::string::npos) {
      continue;
    }
    std::cout << kTtsCdnBase << url_encode_path(key) << "\t" << key << "\n";
  }
  return 0;
}

int manifest_g2p(const std::vector<std::string>& spec) {
  if (spec.empty()) {
    return fail("g2p manifest needs <language>");
  }
  char* out = nullptr;
  const int32_t err =
      moonshine_get_g2p_dependencies(spec[0].c_str(), nullptr, 0, &out);
  if (err != MOONSHINE_ERROR_NONE || out == nullptr) {
    return fail("moonshine_get_g2p_dependencies failed");
  }
  const std::string csv(out);
  std::free(out);
  size_t start = 0;
  while (start <= csv.size()) {
    const size_t comma = csv.find(',', start);
    const std::string key =
        csv.substr(start, comma == std::string::npos ? std::string::npos
                                                     : comma - start);
    if (!key.empty() && key.find('/') != std::string::npos) {
      std::cout << kTtsCdnBase << url_encode_path(key) << "\t" << key << "\n";
    }
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return 0;
}

// -------------------------------- run mode ---------------------------------

std::vector<float> load_speech_or_tone() {
  // Prefer a real 16 kHz speech clip so the encoder/decoder actually run; fall
  // back to a synthetic tone if the asset is not beside the working directory.
  const char* candidates[] = {
      "test-assets/two_cities_16k.wav",
      "../test-assets/two_cities_16k.wav",
      "../../test-assets/two_cities_16k.wav",
  };
  for (const char* path : candidates) {
    if (!std::filesystem::exists(path)) {
      continue;
    }
    float* data = nullptr;
    size_t count = 0;
    int32_t sample_rate = 0;
    if (load_wav_data(path, &data, &count, &sample_rate) && data != nullptr &&
        count > 0) {
      // Cap to ~5 seconds to keep the test fast.
      const size_t max_samples =
          static_cast<size_t>(sample_rate > 0 ? sample_rate : 16000) * 5;
      const size_t used = count < max_samples ? count : max_samples;
      std::vector<float> audio(data, data + used);
      std::free(data);
      return audio;
    }
    if (data != nullptr) {
      std::free(data);
    }
  }
  std::vector<float> audio(16000, 0.0f);
  for (size_t i = 0; i < audio.size(); ++i) {
    audio[i] = 0.05f * std::sin(2.0 * 3.14159265 * 220.0 *
                                static_cast<double>(i) / 16000.0);
  }
  return audio;
}

bool is_streaming_arch(uint32_t arch) {
  return arch == MOONSHINE_MODEL_ARCH_TINY_STREAMING ||
         arch == MOONSHINE_MODEL_ARCH_BASE_STREAMING ||
         arch == MOONSHINE_MODEL_ARCH_SMALL_STREAMING ||
         arch == MOONSHINE_MODEL_ARCH_MEDIUM_STREAMING;
}

int run_stt(const std::string& root, const std::vector<std::string>& spec) {
  const std::string language = spec.empty() ? "en" : spec[0];
  uint32_t arch = MOONSHINE_MODEL_ARCH_TINY;
  if (spec.size() >= 2) {
    arch = static_cast<uint32_t>(std::stoul(spec[1]));
  }
  const int32_t handle = moonshine_load_transcriber_from_files(
      root.c_str(), arch, nullptr, 0, MOONSHINE_HEADER_VERSION);
  if (handle < 0) {
    return fail("failed to load transcriber: " +
                std::string(moonshine_error_to_string(handle)));
  }
  std::vector<float> audio = load_speech_or_tone();
  transcript_t* transcript = nullptr;
  int32_t err = MOONSHINE_ERROR_NONE;
  // The transcript is owned by the transcriber/stream and only valid until the
  // next call on it, so capture the line count before any teardown.
  uint64_t line_count = 0;
  if (is_streaming_arch(arch)) {
    const int32_t stream = moonshine_create_stream(handle, 0);
    if (stream < 0) {
      moonshine_free_transcriber(handle);
      return fail("failed to create stream");
    }
    moonshine_start_stream(handle, stream);
    err = moonshine_transcribe_add_audio_to_stream(
        handle, stream, audio.data(), audio.size(), 16000, 0);
    if (err == MOONSHINE_ERROR_NONE) {
      err = moonshine_transcribe_stream(handle, stream,
                                        MOONSHINE_FLAG_FORCE_UPDATE,
                                        &transcript);
    }
    if (err == MOONSHINE_ERROR_NONE && transcript != nullptr) {
      line_count = transcript->line_count;
    }
    moonshine_stop_stream(handle, stream);
    moonshine_free_stream(handle, stream);
  } else {
    err = moonshine_transcribe_without_streaming(
        handle, audio.data(), audio.size(), 16000, 0, &transcript);
    if (err == MOONSHINE_ERROR_NONE && transcript != nullptr) {
      line_count = transcript->line_count;
    }
  }
  if (err != MOONSHINE_ERROR_NONE) {
    moonshine_free_transcriber(handle);
    return fail("transcription failed: " +
                std::string(moonshine_error_to_string(err)));
  }
  std::cerr << "stt ok: " << line_count << " line(s) from " << language << "\n";
  moonshine_free_transcriber(handle);
  return 0;
}

int run_intent(const std::string& root, const std::vector<std::string>& spec) {
  const std::string variant = spec.size() >= 2 ? spec[1] : std::string("q4");
  const int32_t handle = moonshine_create_intent_recognizer(
      root.c_str(), MOONSHINE_EMBEDDING_MODEL_ARCH_GEMMA_300M, variant.c_str());
  if (handle < 0) {
    return fail("failed to create intent recognizer: " +
                std::string(moonshine_error_to_string(handle)));
  }
  int32_t err =
      moonshine_register_intent(handle, "turn on the lights", nullptr, 0, 0);
  if (err != MOONSHINE_ERROR_NONE) {
    moonshine_free_intent_recognizer(handle);
    return fail("failed to register intent");
  }
  moonshine_intent_match_t* matches = nullptr;
  uint64_t count = 0;
  err = moonshine_get_closest_intents(handle, "switch on the lights", 0.0f,
                                      &matches, &count);
  if (err != MOONSHINE_ERROR_NONE) {
    moonshine_free_intent_recognizer(handle);
    return fail("failed to rank intents");
  }
  std::cerr << "intent ok: " << count << " match(es)\n";
  moonshine_free_intent_matches(matches, count);
  moonshine_free_intent_recognizer(handle);
  return 0;
}

int run_tts(const std::string& root, const std::vector<std::string>& spec) {
  if (spec.empty()) {
    return fail("tts run needs <language> [<voice>]");
  }
  const std::string language = spec[0];
  std::vector<moonshine_option_t> options;
  options.push_back({"g2p_root", root.c_str()});
  std::string voice_value;
  if (spec.size() >= 2) {
    voice_value = spec[1];
    options.push_back({"voice", voice_value.c_str()});
  }
  const int32_t handle = moonshine_create_tts_synthesizer_from_files(
      language.c_str(), nullptr, 0, options.data(), options.size(),
      MOONSHINE_HEADER_VERSION);
  if (handle < 0) {
    return fail("failed to create tts synthesizer: " +
                std::string(moonshine_error_to_string(handle)));
  }
  float* audio = nullptr;
  uint64_t audio_size = 0;
  int32_t sample_rate = 0;
  const int32_t err = moonshine_text_to_speech(
      handle, "Hello world.", nullptr, 0, &audio, &audio_size, &sample_rate);
  if (err != MOONSHINE_ERROR_NONE || audio_size == 0) {
    if (audio != nullptr) {
      std::free(audio);
    }
    moonshine_free_tts_synthesizer(handle);
    return fail("tts synthesis failed or produced no audio");
  }
  std::cerr << "tts ok: " << audio_size << " samples at " << sample_rate
            << " Hz\n";
  std::free(audio);
  moonshine_free_tts_synthesizer(handle);
  return 0;
}

int run_g2p(const std::string& root, const std::vector<std::string>& spec) {
  if (spec.empty()) {
    return fail("g2p run needs <language>");
  }
  const std::string language = spec[0];
  const moonshine_option_t options[] = {{"g2p_root", root.c_str()}};
  const int32_t handle = moonshine_create_grapheme_to_phonemizer_from_files(
      language.c_str(), nullptr, 0, options, 1, MOONSHINE_HEADER_VERSION);
  if (handle < 0) {
    return fail("failed to create phonemizer: " +
                std::string(moonshine_error_to_string(handle)));
  }
  const char* phonemes = nullptr;
  uint64_t phoneme_count = 0;
  const int32_t err = moonshine_text_to_phonemes(handle, "hello", nullptr, 0,
                                                 &phonemes, &phoneme_count);
  if (err != MOONSHINE_ERROR_NONE || phoneme_count == 0) {
    moonshine_free_grapheme_to_phonemizer(handle);
    return fail("g2p produced no phonemes");
  }
  std::cerr << "g2p ok: " << phoneme_count << " phoneme byte(s)\n";
  moonshine_free_grapheme_to_phonemizer(handle);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    print_usage();
    return 2;
  }
  const std::string mode = argv[1];
  const std::string modality = argv[2];

  try {
    if (mode == "manifest") {
      std::vector<std::string> spec;
      for (int i = 3; i < argc; ++i) {
        spec.push_back(argv[i]);
      }
      if (modality == "stt") {
        return manifest_stt(spec);
      }
      if (modality == "intent") {
        return manifest_intent(spec);
      }
      if (modality == "tts") {
        return manifest_tts(spec);
      }
      if (modality == "g2p") {
        return manifest_g2p(spec);
      }
      return fail("unknown modality: " + modality);
    }
    if (mode == "run") {
      if (argc < 4) {
        print_usage();
        return 2;
      }
      const std::string root = argv[3];
      std::vector<std::string> spec;
      for (int i = 4; i < argc; ++i) {
        spec.push_back(argv[i]);
      }
      if (modality == "stt") {
        return run_stt(root, spec);
      }
      if (modality == "intent") {
        return run_intent(root, spec);
      }
      if (modality == "tts") {
        return run_tts(root, spec);
      }
      if (modality == "g2p") {
        return run_g2p(root, spec);
      }
      return fail("unknown modality: " + modality);
    }
  } catch (const std::exception& e) {
    return fail(std::string("exception: ") + e.what());
  }
  print_usage();
  return 2;
}
