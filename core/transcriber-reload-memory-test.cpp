// Repeated create/destroy memory regression for file-backed Transcribers.
//
// GitHub issue #216: each Transcriber built from files retained mmap'd
// cross_kv.ort / decoder_kv.ort (and decoder_kv_with_attention.ort when word
// timestamps were on) after close(), growing mapped bytes by the on-disk size
// every cycle. This test constructs and frees a transcriber in a loop and
// checks that mapped .ort ranges (/proc/self/maps on Linux, vmmap on macOS)
// return to the post-warmup baseline. Invoked from
// scripts/reliability-remote.sh.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "moonshine-c-api.h"

#if defined(__APPLE__)
#include <unistd.h>
#endif

namespace {

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

bool env_disabled() {
  const char *raw =
      std::getenv("MOONSHINE_TRANSCRIBER_RELOAD_MEMORY_TEST_DISABLE");
  return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
}

std::string read_process_maps() {
#if defined(__linux__)
  std::ifstream maps("/proc/self/maps");
  if (!maps) {
    return {};
  }
  std::ostringstream oss;
  oss << maps.rdbuf();
  return oss.str();
#elif defined(__APPLE__)
  const std::string cmd =
      "vmmap -w " + std::to_string(getpid()) + " 2>/dev/null";
  FILE *pipe = popen(cmd.c_str(), "r");
  if (pipe == nullptr) {
    return {};
  }
  std::string out;
  char buf[4096];
  while (std::fgets(buf, sizeof(buf), pipe) != nullptr) {
    out += buf;
  }
  pclose(pipe);
  return out;
#else
  return {};
#endif
}

bool maps_available() {
#if defined(__linux__) || defined(__APPLE__)
  return true;
#else
  return false;
#endif
}

bool parse_hex_range(const std::string &line, unsigned long long *start,
                     unsigned long long *end) {
  const auto dash = line.find('-');
  if (dash == std::string::npos || dash == 0) {
    return false;
  }
  size_t hex_begin = dash;
  while (hex_begin > 0 &&
         std::isxdigit(static_cast<unsigned char>(line[hex_begin - 1]))) {
    hex_begin--;
  }
  if (hex_begin == dash) {
    return false;
  }
  char *end_start = nullptr;
  char *end_end = nullptr;
  *start = std::strtoull(line.c_str() + hex_begin, &end_start, 16);
  if (end_start != line.c_str() + dash) {
    return false;
  }
  *end = std::strtoull(line.c_str() + dash + 1, &end_end, 16);
  return end_end != line.c_str() + dash + 1 && *end > *start;
}

std::map<std::string, size_t> mapped_ort_bytes_by_file() {
  std::map<std::string, size_t> totals;
  std::istringstream maps(read_process_maps());
  std::string line;
  while (std::getline(maps, line)) {
    while (!line.empty() &&
           (line.back() == ' ' || line.back() == '\t' || line.back() == '\r')) {
      line.pop_back();
    }
    const auto last_space = line.find_last_of(" \t");
    if (last_space == std::string::npos || last_space + 1 >= line.size()) {
      continue;
    }
    const std::string path = line.substr(last_space + 1);
    if (path.size() < 4 || path.compare(path.size() - 4, 4, ".ort") != 0) {
      continue;
    }
    unsigned long long start = 0;
    unsigned long long end = 0;
    if (!parse_hex_range(line, &start, &end)) {
      continue;
    }
    const std::string name = std::filesystem::path(path).filename().string();
    totals[name] += static_cast<size_t>(end - start);
  }
  return totals;
}

size_t total_mapped_bytes(const std::map<std::string, size_t> &by_file) {
  size_t total = 0;
  for (const auto &entry : by_file) {
    total += entry.second;
  }
  return total;
}

std::string format_mapped(const std::map<std::string, size_t> &by_file) {
  if (by_file.empty()) {
    return "(none)";
  }
  std::string out;
  for (const auto &entry : by_file) {
    if (!out.empty()) {
      out += ", ";
    }
    out += entry.first;
    out += "=";
    out += std::to_string(entry.second);
  }
  return out;
}

int32_t load_transcriber(const char *model_path, uint32_t model_arch,
                         bool word_timestamps) {
  const moonshine_option_t ts_opt{"word_timestamps", "true"};
  return moonshine_load_transcriber_from_files(
      model_path, model_arch, word_timestamps ? &ts_opt : nullptr,
      word_timestamps ? 1 : 0, MOONSHINE_HEADER_VERSION);
}

struct ReloadCase {
  const char *name;
  const char *model_path;
  uint32_t model_arch;
  bool word_timestamps;
  const char *expect_live;
  const char *expect_absent_live;
};

void run_reload_case(const ReloadCase &spec, size_t cycles,
                     size_t slack_bytes) {
  REQUIRE_MESSAGE(std::filesystem::exists(spec.model_path), spec.model_path);

  int32_t warm =
      load_transcriber(spec.model_path, spec.model_arch, spec.word_timestamps);
  REQUIRE(warm >= 0);
  if (maps_available()) {
    const auto live = mapped_ort_bytes_by_file();
    const auto live_it = live.find(spec.expect_live);
    const bool has_expected = live_it != live.end() && live_it->second > 0;
    INFO(spec.name << " live maps missing " << spec.expect_live << ": "
                   << format_mapped(live));
    REQUIRE(has_expected);
    if (spec.expect_absent_live != nullptr) {
      const auto absent_it = live.find(spec.expect_absent_live);
      const bool old_unmapped =
          absent_it == live.end() || absent_it->second == 0;
      INFO(spec.name << " live maps still hold " << spec.expect_absent_live
                     << " after decoder swap: " << format_mapped(live));
      REQUIRE(old_unmapped);
    }
  }
  moonshine_free_transcriber(warm);

  const auto baseline = mapped_ort_bytes_by_file();
  const size_t baseline_total = total_mapped_bytes(baseline);

  for (size_t i = 0; i < cycles; ++i) {
    int32_t handle = load_transcriber(spec.model_path, spec.model_arch,
                                      spec.word_timestamps);
    REQUIRE(handle >= 0);
    moonshine_free_transcriber(handle);

    if (!maps_available()) {
      continue;
    }
    const auto now = mapped_ort_bytes_by_file();
    const size_t now_total = total_mapped_bytes(now);
    const size_t growth =
        now_total > baseline_total ? now_total - baseline_total : 0;
    if (growth > slack_bytes) {
      FAIL(spec.name << " mapped .ort bytes grew after cycle " << (i + 1)
                     << ": baseline=" << baseline_total << " now=" << now_total
                     << " growth=" << growth << " slack=" << slack_bytes
                     << " baseline_files=" << format_mapped(baseline)
                     << " now_files=" << format_mapped(now));
    }
  }
}

}  // namespace

TEST_CASE("transcriber-reload-memory") {
  if (env_disabled()) {
    MESSAGE("MOONSHINE_TRANSCRIBER_RELOAD_MEMORY_TEST_DISABLE=1, skipping");
    return;
  }

  const size_t cycles =
      env_size("MOONSHINE_TRANSCRIBER_RELOAD_MEMORY_CYCLES", 6);
  REQUIRE(cycles >= 3);
  // The leak in issue #216 was the full on-disk size of decoder_kv.ort
  // (~33 MB for tiny-streaming). A megabyte of slack covers measurement noise
  // without letting a whole extra mapping through.
  constexpr size_t kSlackBytes = 1024 * 1024;

  if (!maps_available()) {
    MESSAGE("process map listing is unavailable; running load/free smoke only");
  }

  const ReloadCase cases[] = {
      {"tiny-streaming", "tiny-streaming-en",
       MOONSHINE_MODEL_ARCH_TINY_STREAMING, false, "decoder_kv.ort",
       "decoder_kv_with_attention.ort"},
      {"tiny-streaming-word-timestamps", "tiny-streaming-en",
       MOONSHINE_MODEL_ARCH_TINY_STREAMING, true,
       "decoder_kv_with_attention.ort", "decoder_kv.ort"},
      {"tiny-en", "tiny-en", MOONSHINE_MODEL_ARCH_TINY, false,
       "decoder_model_merged.ort", "decoder_with_attention.ort"},
      {"tiny-en-word-timestamps", "tiny-en", MOONSHINE_MODEL_ARCH_TINY, true,
       "decoder_with_attention.ort", "decoder_model_merged.ort"},
  };

  for (const ReloadCase &spec : cases) {
    CAPTURE(spec.name);
    run_reload_case(spec, cycles, kSlackBytes);
  }
}
