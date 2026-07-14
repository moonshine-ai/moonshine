// Repeated-use memory regression test for the text-to-speech synthesizers.
//
// The transcriber has a long-stream memory test (transcriber-streaming-memory-
// test.cpp); this is the TTS analogue. It exercises each available engine
// (Kokoro, Piper, ZipVoice, English only) through the public C API in two loops
// that probe the two distinct leak surfaces:
//
//   * "synth"  - one synthesizer, many moonshine_text_to_speech() calls (each
//                output buffer freed). Catches leaks in the inference / G2P /
//                per-call allocation path.
//   * "reload" - repeatedly create and free the synthesizer handle. Catches
//                leaks in model load / teardown and the handle table.
//
// TTS exposes no internal retained-byte accessor (unlike the transcriber's
// stream_vad_completed_audio_bytes), so the signal here is process RSS: the
// test samples resident memory after every iteration and fails on sustained,
// post-warmup growth using the same regression heuristic as the transcriber
// test. onnxruntime arena allocators grow during warmup and then plateau, so
// the warmup window is skipped and only a persistent upward trend is treated as
// a regression.
//
// The bundled TTS data (core/moonshine-tts/data) is required. Pass its path as
// argv[1], or run with a working directory from which it can be discovered
// (repo root, test-assets/, or core/build). Engines whose assets are absent are
// skipped. Invoked from scripts/reliability-remote.sh.

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "moonshine-c-api.h"

#if defined(__linux__)
#elif defined(__APPLE__)
#include <mach/mach.h>
#endif

namespace {

namespace fs = std::filesystem;

fs::path g_data_root;

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
      // Parse with strtoull rather than sscanf so the conversion reports errors
      // (clang-tidy cert-err34-c flags scanf-family string-to-number use). It
      // skips leading whitespace after "VmRSS:" and stops at the trailing " kB".
      char *end = nullptr;
      const unsigned long long parsed = std::strtoull(line + 6, &end, 10);
      if (end != line + 6) {
        rss_kb = static_cast<size_t>(parsed);
      }
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

size_t env_size(const char *name, size_t default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  char *end = nullptr;
  const unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (end == raw) {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}

// Mirrors transcriber-streaming-memory-test's detector: fits a line to the
// post-warmup samples and reports sustained growth only when the median gain,
// the fraction of non-decreasing steps, and the fitted slope all clear their
// thresholds. This tolerates one-off jumps and allocator caching while still
// catching an unbounded per-iteration leak.
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
    *out_report =
        "early_median=" + std::to_string(early_median) +
        " KiB, late_median=" + std::to_string(late_median) +
        " KiB, growth=" + std::to_string(growth) +
        " KiB, growth_tolerance=" + std::to_string(growth_tolerance) +
        " KiB, positive_fraction=" + std::to_string(positive_fraction) +
        ", slope_per_sample=" + std::to_string(slope_per_sample) +
        " KiB, samples=" + std::to_string(samples.size());
  }

  return growth > growth_tolerance &&
         positive_fraction >= min_positive_fraction &&
         slope_per_sample > min_slope_per_sample;
}

// Rotating text so the G2P / inference paths do real, varying work per call.
const char *kTexts[] = {
    "The quick brown fox jumps over the lazy dog.",
    "Speech synthesis should not leak memory over repeated calls.",
    "One, two, three, four, five, six, seven, eight, nine, ten.",
    "How much wood would a woodchuck chuck if it could chuck wood?",
    "A journey of a thousand miles begins with a single step.",
    "Testing repeated synthesis for unbounded memory growth.",
    "She sells seashells by the seashore on a sunny afternoon.",
    "Moonshine turns text into natural sounding speech on device.",
};
constexpr size_t kTextCount = sizeof(kTexts) / sizeof(kTexts[0]);

struct EngineSpec {
  const char *name;
  const char *language;
  std::string voice;
  const char *root_key;  // "model_root" (Kokoro/Piper) or "g2p_root" (ZipVoice)
};

int32_t create_synth(const EngineSpec &spec) {
  const std::string root = g_data_root.string();
  const moonshine_option_t opts[] = {
      {"voice", spec.voice.c_str()},
      {"speed", "1.0"},
      {spec.root_key, root.c_str()},
  };
  return moonshine_create_tts_synthesizer_from_files(
      spec.language, nullptr, 0, opts,
      static_cast<uint64_t>(sizeof(opts) / sizeof(opts[0])),
      MOONSHINE_HEADER_VERSION);
}

bool synth_once(int32_t handle, size_t text_index) {
  float *audio = nullptr;
  uint64_t audio_n = 0;
  int32_t sr = 0;
  const int32_t rc =
      moonshine_text_to_speech(handle, kTexts[text_index % kTextCount], nullptr,
                               0, &audio, &audio_n, &sr);
  const bool ok = rc == MOONSHINE_ERROR_NONE && audio != nullptr && audio_n > 0;
  std::free(audio);
  return ok;
}

bool reload_strict() {
  const char *raw = std::getenv("MOONSHINE_TTS_MEMORY_RELOAD_STRICT");
  return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

// Runs `iterations` of `body`, sampling RSS after each, and checks for
// sustained post-warmup growth. Reports the regression summary either way for
// context. When `advisory` is true a positive finding is logged but does not
// fail the run (RSS across repeated model load/free is dominated by allocator /
// OS page retention rather than a reliable leak signal); otherwise it is a hard
// gate.
void run_growth_phase(const std::string &label, size_t iterations,
                      const std::function<bool(size_t)> &body,
                      size_t absolute_tolerance_kb,
                      double min_positive_fraction,
                      double min_slope_kb_per_sample, bool advisory) {
  std::vector<size_t> rss_samples;
  rss_samples.reserve(iterations);
  for (size_t i = 0; i < iterations; ++i) {
    REQUIRE_MESSAGE(body(i), label << ": iteration " << i << " failed");
    rss_samples.push_back(read_rss_kb());
  }

  std::string report;
  const bool growing = detect_continual_growth(
      rss_samples, absolute_tolerance_kb, min_positive_fraction,
      min_slope_kb_per_sample, &report);
  MESSAGE(label << ": " << report);
  if (growing) {
    for (size_t i = 0; i < rss_samples.size(); ++i) {
      std::printf("  %s sample[%zu]: rss=%zu KiB\n", label.c_str(), i,
                  rss_samples[i]);
    }
    std::fflush(stdout);
  }
  if (advisory && !reload_strict()) {
    if (growing) {
      MESSAGE(label << ": sustained RSS growth (advisory, not failing): "
                    << report);
    }
    return;
  }
  CHECK_FALSE_MESSAGE(growing,
                      label << " shows sustained RSS growth: " << report);
}

void exercise_engine(const EngineSpec &spec) {
  const size_t synth_iterations =
      env_size("MOONSHINE_TTS_MEMORY_SYNTH_ITERATIONS", 64);
  const size_t reload_iterations =
      env_size("MOONSHINE_TTS_MEMORY_RELOAD_ITERATIONS", 24);

  // A persistent handle exercised repeatedly: probes the per-call path. A leak
  // of the output buffer or an inference-scratch buffer shows up as a steady
  // KiB/iteration climb.
  {
    const int32_t handle = create_synth(spec);
    REQUIRE_MESSAGE(handle >= 0, spec.name << ": failed to create synthesizer");
    run_growth_phase(
        std::string(spec.name) + " synth", synth_iterations,
        [handle](size_t i) { return synth_once(handle, i); },
        /*absolute_tolerance_kb=*/32 * 1024,
        /*min_positive_fraction=*/0.70,
        /*min_slope_kb_per_sample=*/256.0,
        /*advisory=*/false);
    moonshine_free_tts_synthesizer(handle);
  }

  // Repeated create/synthesize/free: probes model load + teardown. Reloading is
  // heavier (especially ZipVoice), so it runs fewer iterations with a looser
  // tolerance, and is advisory by default because repeated model load/free
  // leaves resident pages the allocator/OS does not return, which looks like
  // growth without being a leak. Set MOONSHINE_TTS_MEMORY_RELOAD_STRICT=1 to
  // make it a hard gate.
  run_growth_phase(
      std::string(spec.name) + " reload", reload_iterations,
      [&spec](size_t i) {
        const int32_t handle = create_synth(spec);
        if (handle < 0) {
          return false;
        }
        const bool ok = synth_once(handle, i);
        moonshine_free_tts_synthesizer(handle);
        return ok;
      },
      /*absolute_tolerance_kb=*/64 * 1024,
      /*min_positive_fraction=*/0.70,
      /*min_slope_kb_per_sample=*/512.0,
      /*advisory=*/true);
}

bool file_present(const fs::path &p) {
  std::error_code ec;
  return fs::is_regular_file(p, ec);
}

std::optional<EngineSpec> kokoro_spec() {
  const fs::path kokoro = g_data_root / "kokoro";
  const bool model =
      file_present(kokoro / "model.onnx") || file_present(kokoro / "model.ort");
  const bool voice = file_present(kokoro / "voices" / "af_heart.kokorovoice");
  if (!model || !voice) {
    return std::nullopt;
  }
  return EngineSpec{"kokoro", "en_us", "kokoro_af_heart", "model_root"};
}

std::optional<EngineSpec> piper_spec() {
  const fs::path voices = g_data_root / "en_us" / "piper-voices";
  std::error_code ec;
  if (!fs::is_directory(voices, ec)) {
    return std::nullopt;
  }
  // Prefer a small, low-quality voice to keep the loop fast, else the first
  // .onnx we find.
  std::string chosen;
  for (const auto &ent : fs::directory_iterator(voices, ec)) {
    if (!ent.is_regular_file()) {
      continue;
    }
    const fs::path &p = ent.path();
    if (p.extension() != ".onnx") {
      continue;
    }
    const std::string stem = p.stem().string();
    if (!file_present(voices / (stem + ".onnx.json"))) {
      continue;
    }
    if (chosen.empty() || stem.find("-low") != std::string::npos) {
      chosen = stem;
    }
    if (chosen.find("-low") != std::string::npos) {
      break;
    }
  }
  if (chosen.empty()) {
    return std::nullopt;
  }
  return EngineSpec{"piper", "en_us", "piper_" + chosen, "model_root"};
}

std::optional<EngineSpec> zipvoice_spec() {
  const fs::path zv = g_data_root / "zipvoice";
  const bool have =
      (file_present(zv / "text_encoder.ort") ||
       file_present(zv / "text_encoder.onnx")) &&
      (file_present(zv / "fm_decoder.ort") ||
       file_present(zv / "fm_decoder.onnx")) &&
      (file_present(zv / "vocoder.ort") || file_present(zv / "vocoder.onnx")) &&
      file_present(zv / "tokens.txt");
  if (!have) {
    return std::nullopt;
  }
  return EngineSpec{"zipvoice", "en_us", "zipvoice_american_female",
                    "g2p_root"};
}

}  // namespace

TEST_CASE("tts-repeated-memory-kokoro") {
  const auto spec = kokoro_spec();
  if (!spec) {
    MESSAGE("skip: Kokoro assets not present under data/kokoro");
    return;
  }
  exercise_engine(*spec);
}

TEST_CASE("tts-repeated-memory-piper") {
  const auto spec = piper_spec();
  if (!spec) {
    MESSAGE(
        "skip: Piper en_us voices not present under data/en_us/piper-voices");
    return;
  }
  exercise_engine(*spec);
}

TEST_CASE("tts-repeated-memory-zipvoice") {
  const auto spec = zipvoice_spec();
  if (!spec) {
    MESSAGE("skip: ZipVoice model bundle not present under data/zipvoice");
    return;
  }
  exercise_engine(*spec);
}

namespace {

// Discovers core/moonshine-tts/data relative to common working directories used
// by the test scripts (repo root, test-assets/, core/build/).
std::optional<fs::path> discover_data_root() {
  const fs::path cwd = fs::current_path();
  const fs::path candidates[] = {
      cwd / "core" / "moonshine-tts" / "data",
      cwd.parent_path() / "core" / "moonshine-tts" / "data",
      cwd / "moonshine-tts" / "data",
      cwd.parent_path() / "moonshine-tts" / "data",
      cwd / ".." / "moonshine-tts" / "data",
      cwd / ".." / ".." / "moonshine-tts" / "data",
  };
  for (const auto &p : candidates) {
    std::error_code ec;
    const fs::path abs = fs::absolute(p, ec);
    if (ec) {
      continue;
    }
    if (fs::is_directory(abs / "en_us", ec)) {
      return abs;
    }
  }
  return std::nullopt;
}

}  // namespace

int main(int argc, char **argv) {
  const char *disable = std::getenv("MOONSHINE_TTS_MEMORY_TEST_DISABLE");
  if (disable != nullptr && disable[0] == '1' && disable[1] == '\0') {
    std::printf("MOONSHINE_TTS_MEMORY_TEST_DISABLE=1, skipping\n");
    return 0;
  }

  // Treat a first argument only as the data dir when it actually resolves to
  // one; otherwise leave it for doctest (e.g. a --test-case filter).
  std::optional<fs::path> data_root;
  int first_doctest_arg = 1;
  if (argc >= 2) {
    std::error_code ec;
    const fs::path candidate = fs::weakly_canonical(fs::path(argv[1]), ec);
    if (!ec && fs::is_directory(candidate / "en_us", ec)) {
      data_root = candidate;
      first_doctest_arg = 2;
    }
  }
  if (!data_root) {
    data_root = discover_data_root();
  }
  if (!data_root) {
    std::fprintf(stderr,
                 "error: could not locate core/moonshine-tts/data. Pass its "
                 "absolute path as the first argument, or run from the repo "
                 "root / test-assets.\n");
    return 2;
  }
  g_data_root = *data_root;
  std::printf("tts-repeated-memory: data root = %s\n",
              g_data_root.string().c_str());

  doctest::Context ctx;
  // Forward remaining args to doctest (filters, etc.).
  std::vector<char *> doctest_argv;
  doctest_argv.push_back(argv[0]);
  for (int i = first_doctest_arg; i < argc; ++i) {
    doctest_argv.push_back(argv[i]);
  }
  ctx.applyCommandLine(static_cast<int>(doctest_argv.size()),
                       doctest_argv.data());
  return ctx.run();
}
