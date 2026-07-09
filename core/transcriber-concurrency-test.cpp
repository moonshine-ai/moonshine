// Multi-threaded stress test for the Transcriber's thread-safe entry points.
//
// The library is written to be used from several threads at once: create_stream
// / add_audio_to_stream / transcribe_stream / free_stream are guarded by a web
// of mutexes (streams_mutex, the shared model mutexes, and per-stream mutexes).
// The single-threaded unit tests never exercise those locks concurrently, so
// this test drives many independent streams in parallel to surface first-party
// data races.
//
// It exists primarily to give ThreadSanitizer something real to analyse. Because
// TSan's interceptors deadlock inside onnxruntime's uninstrumented thread pool,
// the reliability script runs this (and the TSan build in general) with
// MOONSHINE_ORT_SINGLE_THREAD=1 so onnxruntime stays on the calling thread; the
// first-party synchronization we care about is unaffected by that flag.
//
// Each worker owns its stream for the whole iteration and only ever reads that
// stream's own transcript output, so any race TSan reports here is in the
// library, not in the test harness.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <thread>
#include <vector>

#include "debug-utils.h"
#include "transcriber.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

namespace {

// Keep the clip short so each (single-threaded-onnxruntime) inference is quick;
// the point is lock contention, not transcription accuracy.
constexpr int kSampleRate = 16000;
constexpr size_t kMaxClipSamples = 3 * kSampleRate;  // <= 3 seconds
constexpr int kChunksPerStream = 4;
constexpr int kIterationsPerThread = 3;

std::vector<float> load_clip() {
  const std::string wav_path = "two_cities.wav";
  if (!std::filesystem::exists(wav_path)) {
    return {};
  }
  float *wav_data = nullptr;
  size_t wav_data_size = 0;
  int32_t wav_sample_rate = 0;
  if (!load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                     &wav_sample_rate) ||
      wav_data == nullptr) {
    return {};
  }
  // load_wav_data hands back a raw C-allocated buffer; adopt it in a unique_ptr
  // with a std::free deleter so it is released on every path without a bare
  // deallocation call (see STYLE_GUIDE.md), then copy the leading window into an
  // owning vector.
  std::unique_ptr<float, decltype(&std::free)> owned(wav_data, &std::free);
  const size_t count = std::min(wav_data_size, kMaxClipSamples);
  return std::vector<float>(wav_data, wav_data + count);
}

}  // namespace

TEST_CASE("transcriber-concurrency") {
  std::vector<float> clip = load_clip();
  REQUIRE_MESSAGE(!clip.empty(),
                  "two_cities.wav fixture is required for the concurrency test");

  std::string root_model_path = "tiny-en";
  REQUIRE(std::filesystem::exists(root_model_path));
  std::vector<uint8_t> encoder_model_data =
      load_file_into_memory(root_model_path + "/encoder_model.ort");
  std::vector<uint8_t> decoder_model_data =
      load_file_into_memory(root_model_path + "/decoder_model_merged.ort");
  std::vector<uint8_t> tokenizer_data =
      load_file_into_memory(root_model_path + "/tokenizer.bin");
  REQUIRE(encoder_model_data.size() > 0);
  REQUIRE(decoder_model_data.size() > 0);
  REQUIRE(tokenizer_data.size() > 0);

  TranscriberOptions options;
  options.model_source = TranscriberOptions::ModelSource::MEMORY;
  options.encoder_model_data = encoder_model_data.data();
  options.encoder_model_data_size = encoder_model_data.size();
  options.decoder_model_data = decoder_model_data.data();
  options.decoder_model_data_size = decoder_model_data.size();
  options.tokenizer_data = tokenizer_data.data();
  options.tokenizer_data_size = tokenizer_data.size();
  options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
  // Speaker identification runs the cpp-annote pipeline, which is vendored and
  // spawns its own threads; keep it off so TSan only sees first-party code.
  options.identify_speakers = false;
  options.return_audio_data = false;

  // One transcriber shared across all worker threads: this is the object whose
  // locks we want to stress.
  Transcriber transcriber(options);

  unsigned hw = std::thread::hardware_concurrency();
  const int thread_count =
      static_cast<int>(std::clamp<unsigned>(hw, 3u, 6u));

  std::atomic<int> completed_streams{0};
  std::atomic<bool> saw_null_transcript{false};

  auto worker = [&](int worker_index) {
    // Each worker slices the clip into contiguous chunks so different workers
    // feed different offsets, exercising the per-stream buffers concurrently.
    const size_t chunk = std::max<size_t>(1, clip.size() / kChunksPerStream);
    for (int iter = 0; iter < kIterationsPerThread; ++iter) {
      int32_t stream_id = transcriber.create_stream();
      transcriber.start_stream(stream_id);
      for (int c = 0; c < kChunksPerStream; ++c) {
        const size_t start = static_cast<size_t>(c) * chunk;
        if (start >= clip.size()) {
          break;
        }
        const size_t len = std::min(chunk, clip.size() - start);
        transcriber.add_audio_to_stream(stream_id, clip.data() + start, len,
                                        kSampleRate);
        struct transcript_t *transcript = nullptr;
        transcriber.transcribe_stream(stream_id, 0, &transcript);
        // Only this worker touches this stream, so reading its output here is
        // race-free with respect to the harness; it exercises the read side of
        // the library's per-stream locking.
        if (transcript == nullptr) {
          saw_null_transcript.store(true);
        }
      }
      transcriber.stop_stream(stream_id);
      transcriber.free_stream(stream_id);
      completed_streams.fetch_add(1);
    }
    (void)worker_index;
  };

  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (int i = 0; i < thread_count; ++i) {
    workers.emplace_back(worker, i);
  }
  for (auto &t : workers) {
    t.join();
  }

  CHECK(completed_streams.load() == thread_count * kIterationsPerThread);
  CHECK_FALSE(saw_null_transcript.load());
}
