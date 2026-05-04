#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "moonshine-c-api.h"

static float* load_wav(const char* path, long* num_samples_out) {
  FILE* f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END);
  long file_size = ftell(f);
  fseek(f, 44, SEEK_SET);
  long data_size = file_size - 44;
  int16_t* raw = (int16_t*)malloc(data_size);
  fread(raw, 1, data_size, f);
  fclose(f);
  long n = data_size / sizeof(int16_t);
  float* audio = (float*)malloc(n * sizeof(float));
  for (long i = 0; i < n; i++) audio[i] = raw[i] / 32768.0f;
  free(raw);
  *num_samples_out = n;
  return audio;
}

struct BenchResult {
  double avg_ms;
  double min_ms;
  double max_ms;
  int total_words;
};

static BenchResult run_benchmark(const char* model_path, const char* wav_path,
                  bool word_timestamps, int warmup, int runs,
                  int model_arch = MOONSHINE_MODEL_ARCH_TINY) {
  struct moonshine_option_t options[2];
  int opt_count = 1;
  options[0] = {.name = "identify_speakers", .value = "false"};
  if (word_timestamps) {
    options[1] = {.name = "word_timestamps", .value = "true"};
    opt_count = 2;
  }

  int32_t handle = moonshine_load_transcriber_from_files(
    model_path, model_arch, options, opt_count,
    moonshine_get_version());
  if (handle < 0) {
    fprintf(stderr, "Failed to load model\n");
    exit(1);
  }

  long num_samples = 0;
  float* audio = load_wav(wav_path, &num_samples);

  // Warmup
  for (int i = 0; i < warmup; i++) {
    struct transcript_t* t = NULL;
    moonshine_transcribe_without_streaming(handle, audio, num_samples, 16000, 0, &t);
  }

  // Timed runs
  std::vector<double> times;
  int total_words = 0;
  for (int i = 0; i < runs; i++) {
    struct transcript_t* t = NULL;
    auto start = std::chrono::high_resolution_clock::now();
    moonshine_transcribe_without_streaming(handle, audio, num_samples, 16000, 0, &t);
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    times.push_back(ms);

    if (i == runs - 1 && t) {
      for (uint64_t j = 0; j < t->line_count; j++) {
        total_words += (int)t->lines[j].word_count;
      }
    }
  }

  free(audio);
  moonshine_free_transcriber(handle);

  double sum = 0, mn = 1e9, mx = 0;
  for (double t : times) { sum += t; if (t < mn) mn = t; if (t > mx) mx = t; }
  return {sum / times.size(), mn, mx, total_words};
}

int main(int argc, char** argv) {
  const char* model_path = "../../test-assets/tiny-en";
  const char* wav_path = "../../test-assets/beckett.wav";
  int warmup = 2;
  int runs = 5;

  int model_arch = MOONSHINE_MODEL_ARCH_TINY;
  if (argc > 1) model_path = argv[1];
  if (argc > 2) wav_path = argv[2];
  if (argc > 3) warmup = atoi(argv[3]);
  if (argc > 4) runs = atoi(argv[4]);
  if (argc > 5) model_arch = atoi(argv[5]);

  // Load audio to report duration
  long n = 0;
  float* a = load_wav(wav_path, &n);
  float dur = n / 16000.0f;
  free(a);

  printf("Audio: %s (%.2fs)\n", wav_path, dur);
  printf("Model: %s\n", model_path);
  printf("Warmup: %d, Runs: %d\n\n", warmup, runs);

  printf("Model arch: %d (%s)\n\n", model_arch,
       model_arch == 0 ? "tiny" : model_arch == 2 ? "tiny-streaming" : "other");

  printf("=== 1. Without word timestamps ===\n");
  auto r1 = run_benchmark(model_path, wav_path, false, warmup, runs, model_arch);
  printf("  Average: %.1f ms\n", r1.avg_ms);
  printf("  Min: %.1f ms, Max: %.1f ms\n", r1.min_ms, r1.max_ms);
  printf("  RTF: %.4fx\n", r1.avg_ms / 1000.0 / dur);
  printf("  Words: %d\n\n", r1.total_words);

  printf("=== 2. With word timestamps ===\n");
  auto r2 = run_benchmark(model_path, wav_path, true, warmup, runs, model_arch);
  printf("  Average: %.1f ms\n", r2.avg_ms);
  printf("  Min: %.1f ms, Max: %.1f ms\n", r2.min_ms, r2.max_ms);
  printf("  RTF: %.4fx\n", r2.avg_ms / 1000.0 / dur);
  printf("  Words: %d\n\n", r2.total_words);

  printf("============================================================\n");
  printf("SUMMARY\n");
  printf("============================================================\n");
  printf("  Audio duration:          %.2fs\n", dur);
  printf("  Without word timestamps: %7.1f ms  (baseline)\n", r1.avg_ms);
  printf("  With word timestamps:    %7.1f ms  (%.2fx baseline)\n", r2.avg_ms, r2.avg_ms / r1.avg_ms);
  printf("  Overhead:                %+.1f ms  (%+.1f%%)\n",
       r2.avg_ms - r1.avg_ms,
       (r2.avg_ms - r1.avg_ms) / r1.avg_ms * 100.0);
  printf("  Word count:              %d words\n", r2.total_words);

  return 0;
}
