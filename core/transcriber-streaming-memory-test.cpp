// Long-stream memory regression test for transcribe_stream().
//
// Streams several minutes of synthetic audio with periodic silence gaps so VAD
// marks completed segments, using a tiny on-device model with
// return_audio_data disabled. Tracks completed-segment VAD PCM (the heap that
// leaked before PR #175) and process RSS. Fails on sustained growth of
// completed-segment bytes; RSS is logged for context.
//
// Invoked from scripts/reliability-remote.sh.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

#include "debug-utils.h"
#include "transcriber.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#if defined(__linux__)
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace {

constexpr int32_t kSampleRate = 16000;

size_t read_rss_kb() {
#if defined(__linux__)
  FILE *f = std::fopen("/proc/self/status", "r");
  if (f == nullptr) {
    return 0;
  }
  char line[256];
  size_t rss_kb = 0;
  while (std::fgets(line, sizeof(line), f) != nullptr) {
    if (std::strncmp(line, "VmRSS:", 6) == 0) {
      std::sscanf(line, "VmRSS: %zu kB", &rss_kb);
      break;
    }
  }
  std::fclose(f);
  return rss_kb;
#elif defined(__APPLE__)
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return info.resident_size / 1024;
#else
  return 0;
#endif
}

size_t median(std::vector<size_t> values) {
  if (values.empty()) {
    return 0;
  }
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  return values[mid];
}

float env_float(const char *name, float default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  return std::strtof(raw, nullptr);
}

bool env_disabled() {
  const char *raw = std::getenv("MOONSHINE_STREAM_MEMORY_TEST_DISABLE");
  return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

std::vector<float> load_looped_fixture_audio(float target_seconds) {
  const std::string wav_path = "two_cities_16k.wav";
  if (!std::filesystem::exists(wav_path)) {
    return {};
  }
  float *wav_data = nullptr;
  size_t wav_data_size = 0;
  int32_t wav_sample_rate = 0;
  if (!load_wav_data(wav_path.c_str(), &wav_data, &wav_data_size,
                     &wav_sample_rate) ||
      wav_data == nullptr || wav_sample_rate != kSampleRate) {
    std::free(wav_data);
    return {};
  }
  const size_t target_samples =
      static_cast<size_t>(target_seconds * kSampleRate);
  std::vector<float> audio;
  audio.reserve(target_samples);
  for (size_t i = 0; i < target_samples; ++i) {
    audio.push_back(wav_data[i % wav_data_size]);
  }
  std::free(wav_data);
  return audio;
}

// Returns true when the sample series shows sustained post-warmup growth.
bool detect_continual_growth(const std::vector<size_t> &samples,
                             size_t absolute_tolerance,
                             double min_positive_fraction,
                             double min_slope_per_sample,
                             std::string *out_report) {
  if (samples.size() < 12) {
    if (out_report != nullptr) {
      *out_report = "too few samples (" + std::to_string(samples.size()) + ")";
    }
    return false;
  }

  const size_t warmup_count = std::max<size_t>(6, samples.size() / 4);
  const size_t analysis_count = samples.size() - warmup_count;
  if (analysis_count < 8) {
    if (out_report != nullptr) {
      *out_report = "too few post-warmup samples";
    }
    return false;
  }

  const size_t quarter = std::max<size_t>(2, analysis_count / 4);
  std::vector<size_t> early;
  std::vector<size_t> late;
  early.reserve(quarter);
  late.reserve(quarter);
  for (size_t i = 0; i < quarter; ++i) {
    early.push_back(samples[warmup_count + i]);
    late.push_back(samples[samples.size() - quarter + i]);
  }

  const size_t early_median = median(early);
  const size_t late_median = median(late);
  const size_t growth =
      late_median > early_median ? late_median - early_median : 0;
  const size_t relative_tolerance =
      std::max<size_t>(early_median / 4, absolute_tolerance / 2);
  const size_t growth_tolerance =
      std::max(absolute_tolerance, relative_tolerance);

  size_t positive_steps = 0;
  for (size_t i = warmup_count + 1; i < samples.size(); ++i) {
    if (samples[i] >= samples[i - 1]) {
      positive_steps++;
    }
  }
  const double positive_fraction =
      static_cast<double>(positive_steps) /
      static_cast<double>(samples.size() - warmup_count - 1);

  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_xx = 0.0;
  double sum_xy = 0.0;
  for (size_t i = 0; i < analysis_count; ++i) {
    const double x = static_cast<double>(i);
    const double y = static_cast<double>(samples[warmup_count + i]);
    sum_x += x;
    sum_y += y;
    sum_xx += x * x;
    sum_xy += x * y;
  }
  const double denom =
      static_cast<double>(analysis_count) * sum_xx - sum_x * sum_x;
  const double slope_per_sample =
      denom == 0.0
          ? 0.0
          : (static_cast<double>(analysis_count) * sum_xy - sum_x * sum_y) /
                denom;

  if (out_report != nullptr) {
    *out_report = "early_median=" + std::to_string(early_median) +
                  ", late_median=" + std::to_string(late_median) +
                  ", growth=" + std::to_string(growth) +
                  ", growth_tolerance=" + std::to_string(growth_tolerance) +
                  ", positive_fraction=" + std::to_string(positive_fraction) +
                  ", slope_per_sample=" + std::to_string(slope_per_sample) +
                  ", samples=" + std::to_string(samples.size());
  }

  return growth > growth_tolerance &&
         positive_fraction >= min_positive_fraction &&
         slope_per_sample > min_slope_per_sample;
}

}  // namespace

TEST_CASE("transcriber-streaming-memory") {
  if (env_disabled()) {
    MESSAGE("MOONSHINE_STREAM_MEMORY_TEST_DISABLE=1, skipping");
    return;
  }

  const std::string root_model_path = "tiny-en";
  REQUIRE(std::filesystem::exists(root_model_path));

  const float target_audio_seconds =
      env_float("MOONSHINE_STREAM_MEMORY_AUDIO_SECONDS", 120.0f);
  REQUIRE(target_audio_seconds >= 60.0f);

  std::vector<float> fixture_audio =
      load_looped_fixture_audio(target_audio_seconds);
  REQUIRE_MESSAGE(!fixture_audio.empty(),
                  "two_cities_16k.wav fixture is required");

  TranscriberOptions options;
  options.model_source = TranscriberOptions::ModelSource::FILES;
  options.model_path = root_model_path.c_str();
  options.model_arch = MOONSHINE_MODEL_ARCH_TINY;
  options.return_audio_data = false;
  options.identify_speakers = false;
  options.log_output_text = false;
  options.transcription_interval = 0.5f;

  Transcriber transcriber(options);
  const int32_t stream_id = transcriber.create_stream();
  REQUIRE(stream_id >= 0);
  transcriber.start_stream(stream_id);

  const size_t chunk_size =
      static_cast<size_t>(options.transcription_interval * kSampleRate);
  std::vector<float> chunk(chunk_size);
  size_t total_samples_fed = 0;
  size_t samples_since_last_transcription = 0;
  std::vector<size_t> completed_byte_samples;
  std::vector<size_t> rss_samples;
  const size_t expected_sample_count =
      static_cast<size_t>(target_audio_seconds /
                          options.transcription_interval) +
      8;
  completed_byte_samples.reserve(expected_sample_count);
  rss_samples.reserve(expected_sample_count);

  while (total_samples_fed < fixture_audio.size()) {
    const size_t remaining = fixture_audio.size() - total_samples_fed;
    const size_t this_chunk = std::min(chunk_size, remaining);
    std::copy(fixture_audio.begin() + total_samples_fed,
              fixture_audio.begin() + total_samples_fed + this_chunk,
              chunk.begin());
    transcriber.add_audio_to_stream(stream_id, chunk.data(), this_chunk,
                                    kSampleRate);
    total_samples_fed += this_chunk;
    samples_since_last_transcription += this_chunk;
    if (samples_since_last_transcription <
        static_cast<size_t>(options.transcription_interval * kSampleRate)) {
      continue;
    }
    samples_since_last_transcription = 0;

    struct transcript_t *transcript = nullptr;
    transcriber.transcribe_stream(stream_id, 0, &transcript);
    REQUIRE(transcript != nullptr);
    completed_byte_samples.push_back(
        transcriber.stream_vad_completed_audio_bytes(stream_id));
    rss_samples.push_back(read_rss_kb());
  }

  transcriber.stop_stream(stream_id);
  struct transcript_t *final_transcript = nullptr;
  transcriber.transcribe_stream(stream_id, MOONSHINE_FLAG_FORCE_UPDATE,
                                &final_transcript);
  REQUIRE(final_transcript != nullptr);
  REQUIRE(final_transcript->line_count > 4);
  completed_byte_samples.push_back(
      transcriber.stream_vad_completed_audio_bytes(stream_id));
  rss_samples.push_back(read_rss_kb());

  transcriber.free_stream(stream_id);

  // Completed segments should not retain PCM once transcribed. Allow a little
  // slack for a segment that completes on the same pass we sample it.
  const size_t completed_growth_tolerance = 384 * 1024;

  std::string completed_report;
  const bool completed_growing = detect_continual_growth(
      completed_byte_samples, completed_growth_tolerance, 0.55, 2048.0,
      &completed_report);

  std::string rss_report;
  const bool rss_growing =
      !rss_samples.empty() &&
      detect_continual_growth(rss_samples, 48 * 1024, 0.70, 512.0, &rss_report);
  (void)rss_growing;

  if (completed_growing) {
    LOGF("streaming memory regression (completed_pcm): %s",
         completed_report.c_str());
    LOGF("streaming memory regression (rss): %s", rss_report.c_str());
    for (size_t i = 0; i < completed_byte_samples.size(); ++i) {
      LOGF("  sample[%zu]: completed_pcm=%zu bytes, rss=%zu KiB", i,
           completed_byte_samples[i],
           i < rss_samples.size() ? rss_samples[i] : 0);
    }
  }

  REQUIRE_FALSE(completed_growing);
}
