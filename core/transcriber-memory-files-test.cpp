// Integration test for moonshine_load_transcriber_from_memory_files: the
// keyed-buffer in-memory loader that reaches parity with
// moonshine_load_transcriber_from_files. Exercises several buffer-requirement
// variants (non-streaming, non-streaming + word timestamps, streaming,
// streaming + word timestamps) plus a negative case where a required asset is
// missing.
//
// Model assets are read from the test-assets tree into buffers up front, then
// the current working directory is switched to an empty sandbox before loading
// so the test also proves the models are loaded purely from memory (no
// implicit disk access relative to the model directory).
//
// Run from the repo's test-assets directory (scripts/test-core.sh does this),
// or pass the test-assets path as argv[1].

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "debug-utils.h"
#include "moonshine-c-api.h"

namespace {

namespace fs = std::filesystem;

fs::path g_assets_root;

std::vector<uint8_t> read_file(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    return {};
  }
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)),
                              std::istreambuf_iterator<char>());
}

// Owns the bytes for one model directory, keyed by canonical filename, and
// builds the parallel arrays the C API expects. Keeping this alive for the
// lifetime of the transcriber satisfies the "buffers must outlive the
// transcriber" contract.
struct MemoryBundle {
  std::vector<std::string> names;
  std::vector<std::vector<uint8_t>> buffers;

  void add(const fs::path& dir, const std::string& name) {
    add_as(dir, name, name);
  }

  // Registers real model bytes under a different key, for the cases where the
  // key itself is what is under test.
  void add_as(const fs::path& dir, const std::string& source_name,
              const std::string& key) {
    std::vector<uint8_t> bytes = read_file(dir / source_name);
    REQUIRE_MESSAGE(!bytes.empty(), "missing/empty model asset: "
                                        << (dir / source_name).string());
    names.push_back(key);
    buffers.push_back(std::move(bytes));
  }

  std::vector<const char*> name_ptrs() const {
    std::vector<const char*> out;
    out.reserve(names.size());
    for (const std::string& n : names) {
      out.push_back(n.c_str());
    }
    return out;
  }

  std::vector<const uint8_t*> data_ptrs() const {
    std::vector<const uint8_t*> out;
    out.reserve(buffers.size());
    for (const std::vector<uint8_t>& b : buffers) {
      out.push_back(b.data());
    }
    return out;
  }

  std::vector<uint64_t> sizes() const {
    std::vector<uint64_t> out;
    out.reserve(buffers.size());
    for (const std::vector<uint8_t>& b : buffers) {
      out.push_back(static_cast<uint64_t>(b.size()));
    }
    return out;
  }
};

std::vector<float> load_audio_clip(float max_seconds) {
  float* wav_data = nullptr;
  size_t wav_samples = 0;
  int32_t sample_rate = 0;
  const fs::path wav = g_assets_root / "two_cities_16k.wav";
  const bool ok = load_wav_data(wav.string().c_str(), &wav_data, &wav_samples,
                                &sample_rate);
  REQUIRE(ok);
  REQUIRE(wav_data != nullptr);
  REQUIRE(sample_rate == 16000);
  const size_t max_samples =
      static_cast<size_t>(max_seconds * static_cast<float>(sample_rate));
  const size_t n = std::min(wav_samples, max_samples);
  std::vector<float> audio(wav_data, wav_data + n);
  std::free(wav_data);
  return audio;
}

std::string join_transcript_text(const transcript_t* transcript) {
  std::string text;
  if (transcript == nullptr) {
    return text;
  }
  for (uint64_t i = 0; i < transcript->line_count; ++i) {
    if (transcript->lines[i].text != nullptr) {
      text += transcript->lines[i].text;
      text += " ";
    }
  }
  return text;
}

uint64_t count_words(const transcript_t* transcript) {
  uint64_t total = 0;
  if (transcript == nullptr) {
    return 0;
  }
  for (uint64_t i = 0; i < transcript->line_count; ++i) {
    total += transcript->lines[i].word_count;
  }
  return total;
}

// Switches CWD to a fresh empty sandbox for the duration of a scope, restoring
// the previous directory on destruction. Proves that loads read no assets from
// the model directory implicitly (everything must come from the buffers).
struct SandboxCwd {
  fs::path previous;
  fs::path sandbox;
  SandboxCwd() {
    static int counter = 0;
    previous = fs::current_path();
    sandbox = fs::temp_directory_path() /
              ("moonshine_stt_mem_" + std::to_string(counter++));
    fs::create_directories(sandbox);
    fs::current_path(sandbox);
  }
  ~SandboxCwd() {
    std::error_code ec;
    fs::current_path(previous);
    fs::remove_all(sandbox, ec);
  }
};

int32_t load_from_bundle(const MemoryBundle& bundle, uint32_t model_arch,
                         bool word_timestamps) {
  std::vector<const char*> names = bundle.name_ptrs();
  std::vector<const uint8_t*> datas = bundle.data_ptrs();
  std::vector<uint64_t> sizes = bundle.sizes();
  std::vector<moonshine_option_t> options;
  if (word_timestamps) {
    options.push_back(moonshine_option_t{"word_timestamps", "true"});
  }
  return moonshine_load_transcriber_from_memory_files(
      names.data(), datas.data(), sizes.data(),
      static_cast<uint64_t>(names.size()), model_arch,
      options.empty() ? nullptr : options.data(),
      static_cast<uint64_t>(options.size()), MOONSHINE_HEADER_VERSION);
}

}  // namespace

TEST_CASE(
    "stt-memory-files: non-streaming loads from buffers and transcribes") {
  const fs::path dir = g_assets_root / "tiny-en";
  MemoryBundle bundle;
  bundle.add(dir, "encoder_model.ort");
  bundle.add(dir, "decoder_model_merged.ort");
  bundle.add(dir, "tokenizer.bin");
  const std::vector<float> audio = load_audio_clip(10.0f);

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY, false);
  REQUIRE(handle >= 0);

  transcript_t* transcript = nullptr;
  REQUIRE(moonshine_transcribe_without_streaming(
              handle, const_cast<float*>(audio.data()), audio.size(), 16000, 0,
              &transcript) == MOONSHINE_ERROR_NONE);
  const std::string text = join_transcript_text(transcript);
  MESSAGE("non-streaming transcript: " << text);
  CHECK_FALSE(text.empty());

  moonshine_free_transcriber(handle);
}

TEST_CASE(
    "stt-memory-files: non-streaming with word timestamps populates words") {
  const fs::path dir = g_assets_root / "tiny-en";
  MemoryBundle bundle;
  bundle.add(dir, "encoder_model.ort");
  bundle.add(dir, "decoder_model_merged.ort");
  bundle.add(dir, "tokenizer.bin");
  // The attention-enabled decoder is what makes word timestamps possible; it is
  // an extra buffer requirement over the minimal non-streaming set.
  bundle.add(dir, "decoder_with_attention.ort");
  const std::vector<float> audio = load_audio_clip(10.0f);

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY, true);
  REQUIRE(handle >= 0);

  transcript_t* transcript = nullptr;
  REQUIRE(moonshine_transcribe_without_streaming(
              handle, const_cast<float*>(audio.data()), audio.size(), 16000, 0,
              &transcript) == MOONSHINE_ERROR_NONE);
  CHECK_FALSE(join_transcript_text(transcript).empty());
  CHECK(count_words(transcript) > 0);

  moonshine_free_transcriber(handle);
}

TEST_CASE("stt-memory-files: streaming loads full buffer set and transcribes") {
  const fs::path dir = g_assets_root / "tiny-streaming-en";
  MemoryBundle bundle;
  bundle.add(dir, "frontend.ort");
  bundle.add(dir, "encoder.ort");
  bundle.add(dir, "adapter.ort");
  bundle.add(dir, "cross_kv.ort");
  bundle.add(dir, "decoder_kv.ort");
  bundle.add(dir, "streaming_config.json");
  bundle.add(dir, "tokenizer.bin");
  const std::vector<float> audio = load_audio_clip(10.0f);

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY_STREAMING, false);
  REQUIRE(handle >= 0);

  const int32_t stream = moonshine_create_stream(handle, 0);
  REQUIRE(stream >= 0);
  REQUIRE(moonshine_start_stream(handle, stream) == MOONSHINE_ERROR_NONE);

  const size_t chunk = 16000;  // 1s chunks
  for (size_t off = 0; off < audio.size(); off += chunk) {
    const size_t n = std::min(chunk, audio.size() - off);
    REQUIRE(moonshine_transcribe_add_audio_to_stream(
                handle, stream, const_cast<float*>(audio.data() + off), n,
                16000, 0) == MOONSHINE_ERROR_NONE);
    transcript_t* partial = nullptr;
    moonshine_transcribe_stream(handle, stream, 0, &partial);
  }
  REQUIRE(moonshine_stop_stream(handle, stream) == MOONSHINE_ERROR_NONE);

  transcript_t* final_transcript = nullptr;
  REQUIRE(moonshine_transcribe_stream(
              handle, stream, MOONSHINE_FLAG_FORCE_UPDATE, &final_transcript) ==
          MOONSHINE_ERROR_NONE);
  const std::string text = join_transcript_text(final_transcript);
  MESSAGE("streaming transcript: " << text);
  CHECK_FALSE(text.empty());

  moonshine_free_stream(handle, stream);
  moonshine_free_transcriber(handle);
}

TEST_CASE("stt-memory-files: streaming split frontend pair loads") {
  const fs::path dir = g_assets_root / "tiny-streaming-en";
  if (!fs::exists(dir / "frontend.model.ort") ||
      !fs::exists(dir / "frontend.weights.ort")) {
    MESSAGE("skipping: frontend.model.ort + frontend.weights.ort not present");
    return;
  }
  MemoryBundle bundle;
  bundle.add(dir, "frontend.model.ort");
  bundle.add(dir, "frontend.weights.ort");
  bundle.add(dir, "encoder.ort");
  bundle.add(dir, "adapter.ort");
  bundle.add(dir, "cross_kv.ort");
  bundle.add(dir, "decoder_kv.ort");
  bundle.add(dir, "streaming_config.json");
  bundle.add(dir, "tokenizer.bin");
  const std::vector<float> audio = load_audio_clip(10.0f);

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY_STREAMING, false);
  REQUIRE(handle >= 0);

  const int32_t stream = moonshine_create_stream(handle, 0);
  REQUIRE(stream >= 0);
  REQUIRE(moonshine_start_stream(handle, stream) == MOONSHINE_ERROR_NONE);

  const size_t chunk = 16000;  // 1s chunks
  for (size_t off = 0; off < audio.size(); off += chunk) {
    const size_t n = std::min(chunk, audio.size() - off);
    REQUIRE(moonshine_transcribe_add_audio_to_stream(
                handle, stream, const_cast<float*>(audio.data() + off), n,
                16000, 0) == MOONSHINE_ERROR_NONE);
    transcript_t* partial = nullptr;
    moonshine_transcribe_stream(handle, stream, 0, &partial);
  }
  REQUIRE(moonshine_stop_stream(handle, stream) == MOONSHINE_ERROR_NONE);

  transcript_t* final_transcript = nullptr;
  REQUIRE(moonshine_transcribe_stream(
              handle, stream, MOONSHINE_FLAG_FORCE_UPDATE, &final_transcript) ==
          MOONSHINE_ERROR_NONE);
  const std::string text = join_transcript_text(final_transcript);
  MESSAGE("split-frontend streaming transcript: " << text);
  CHECK_FALSE(text.empty());

  moonshine_free_stream(handle, stream);
  moonshine_free_transcriber(handle);
}

TEST_CASE("stt-memory-files: streaming with attention decoder loads") {
  const fs::path dir = g_assets_root / "tiny-streaming-en";
  MemoryBundle bundle;
  bundle.add(dir, "frontend.ort");
  bundle.add(dir, "encoder.ort");
  bundle.add(dir, "adapter.ort");
  bundle.add(dir, "cross_kv.ort");
  bundle.add(dir, "decoder_kv.ort");
  bundle.add(dir, "streaming_config.json");
  bundle.add(dir, "tokenizer.bin");
  // Extra buffer that swaps in the attention-enabled streaming decoder.
  bundle.add(dir, "decoder_kv_with_attention.ort");
  const std::vector<float> audio = load_audio_clip(6.0f);

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY_STREAMING, true);
  REQUIRE(handle >= 0);

  const int32_t stream = moonshine_create_stream(handle, 0);
  REQUIRE(stream >= 0);
  REQUIRE(moonshine_start_stream(handle, stream) == MOONSHINE_ERROR_NONE);
  REQUIRE(moonshine_transcribe_add_audio_to_stream(
              handle, stream, const_cast<float*>(audio.data()), audio.size(),
              16000, 0) == MOONSHINE_ERROR_NONE);
  transcript_t* transcript = nullptr;
  REQUIRE(moonshine_transcribe_stream(handle, stream,
                                      MOONSHINE_FLAG_FORCE_UPDATE,
                                      &transcript) == MOONSHINE_ERROR_NONE);
  CHECK_FALSE(join_transcript_text(transcript).empty());

  moonshine_free_stream(handle, stream);
  moonshine_free_transcriber(handle);
}

TEST_CASE("stt-memory-files: missing required asset fails to load") {
  const fs::path dir = g_assets_root / "tiny-en";
  MemoryBundle bundle;
  // Deliberately omit decoder_model_merged.ort.
  bundle.add(dir, "encoder_model.ort");
  bundle.add(dir, "tokenizer.bin");

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY, false);
  CHECK(handle < 0);
}

TEST_CASE("stt-memory-files: unrecognized filename key is rejected") {
  const fs::path dir = g_assets_root / "tiny-en";
  MemoryBundle bundle;
  bundle.add(dir, "encoder_model.ort");
  bundle.add(dir, "decoder_model_merged.ort");
  // A plausible typo for tokenizer.bin. Ignoring it would report the missing
  // tokenizer rather than the key the caller actually got wrong.
  bundle.add_as(dir, "tokenizer.bin", "tokenizer.bn");

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY, false);
  CHECK(handle == MOONSHINE_ERROR_INVALID_ARGUMENT);
}

TEST_CASE(
    "stt-memory-files: unrecognized key is rejected even when the model is "
    "complete") {
  const fs::path dir = g_assets_root / "tiny-en";
  MemoryBundle bundle;
  bundle.add(dir, "encoder_model.ort");
  bundle.add(dir, "decoder_model_merged.ort");
  bundle.add(dir, "tokenizer.bin");
  // Every required asset is present, so the rejection can only come from the
  // extra key itself.
  bundle.add_as(dir, "tokenizer.bin", "not_a_moonshine_asset.ort");

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY, false);
  CHECK(handle == MOONSHINE_ERROR_INVALID_ARGUMENT);
}

TEST_CASE(
    "stt-memory-files: recognized keys this load does not need are kept") {
  const fs::path dir = g_assets_root / "tiny-en";
  MemoryBundle bundle;
  bundle.add(dir, "encoder_model.ort");
  bundle.add(dir, "decoder_model_merged.ort");
  bundle.add(dir, "tokenizer.bin");
  // Downloaded for word timestamps but loaded without the option. The key
  // check is a spelling check, not a second pass at the requirements, so
  // handing over more than this particular load needs still works.
  bundle.add(dir, "decoder_with_attention.ort");

  SandboxCwd sandbox;
  const int32_t handle =
      load_from_bundle(bundle, MOONSHINE_MODEL_ARCH_TINY, false);
  REQUIRE(handle >= 0);
  moonshine_free_transcriber(handle);
}

int main(int argc, char** argv) {
  fs::path root = (argc > 1) ? fs::path(argv[1]) : fs::current_path();
  std::error_code ec;
  g_assets_root = fs::weakly_canonical(root, ec);
  if (ec) {
    g_assets_root = root;
  }

  doctest::Context ctx;
  // Forward any doctest flags (skip our own argv[1] asset-path argument).
  std::vector<char*> forwarded;
  forwarded.push_back(argv[0]);
  for (int i = 2; i < argc; ++i) {
    forwarded.push_back(argv[i]);
  }
  ctx.applyCommandLine(static_cast<int>(forwarded.size()), forwarded.data());
  return ctx.run();
}
