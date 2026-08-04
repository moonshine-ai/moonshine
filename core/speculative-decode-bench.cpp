// Compare greedy AR redecode vs decode_full speculative verification on
// streaming updates. Encoder work is shared; only decoder time is timed.
//
// Usage:
//   speculative-decode-bench [-m model_dir] [-w wav] [-i update_interval_s]
//                            [-r repeats]

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "file-utils.h"
#include "moonshine-c-api.h"
#include "moonshine-streaming-model.h"

namespace {

constexpr int kChunkSamples = 1280;  // 80 ms at 16 kHz
constexpr int kSampleRate = 16000;

struct WavData {
  std::vector<float> samples;
  int32_t sample_rate = 0;
};

WavData load_wav(const std::string &path) {
  WavData out;
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) {
    throw std::runtime_error("Failed to open WAV: " + path);
  }

  char riff[4];
  if (std::fread(riff, 1, 4, file) != 4 || std::strncmp(riff, "RIFF", 4) != 0) {
    std::fclose(file);
    throw std::runtime_error("Not a RIFF file: " + path);
  }
  std::fseek(file, 4, SEEK_CUR);
  char wave[4];
  if (std::fread(wave, 1, 4, file) != 4 || std::strncmp(wave, "WAVE", 4) != 0) {
    std::fclose(file);
    throw std::runtime_error("Not a WAVE file: " + path);
  }

  char chunk_id[4];
  uint32_t chunk_size = 0;
  uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0;
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
    throw std::runtime_error("No fmt chunk: " + path);
  }
  fread_exact(&audio_format, sizeof(uint16_t), 1, file, "fmt");
  fread_exact(&num_channels, sizeof(uint16_t), 1, file, "fmt");
  fread_exact(&sample_rate, sizeof(uint32_t), 1, file, "fmt");
  std::fseek(file, 6, SEEK_CUR);  // byte_rate + block_align
  fread_exact(&bits_per_sample, sizeof(uint16_t), 1, file, "fmt");
  if (chunk_size > 16) std::fseek(file, chunk_size - 16, SEEK_CUR);
  if (audio_format != 1 || bits_per_sample != 16) {
    std::fclose(file);
    throw std::runtime_error("Only 16-bit PCM supported: " + path);
  }

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
    throw std::runtime_error("No data chunk: " + path);
  }

  const size_t num_samples = chunk_size / sizeof(int16_t);
  out.samples.resize(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    if (std::fread(&sample, sizeof(int16_t), 1, file) != 1) {
      out.samples.resize(i);
      break;
    }
    out.samples[i] = static_cast<float>(sample) / 32768.0f;
  }
  std::fclose(file);
  out.sample_rate = static_cast<int32_t>(sample_rate);
  return out;
}

std::vector<float> to_16k(const WavData &wav) {
  if (wav.sample_rate == kSampleRate) return wav.samples;
  if (wav.sample_rate <= 0) return {};
  const double ratio =
      static_cast<double>(kSampleRate) / static_cast<double>(wav.sample_rate);
  const size_t out_len =
      static_cast<size_t>(std::llround(wav.samples.size() * ratio));
  std::vector<float> out(out_len);
  for (size_t i = 0; i < out_len; ++i) {
    const double src = i / ratio;
    const size_t i0 = static_cast<size_t>(src);
    const size_t i1 = std::min(i0 + 1, wav.samples.size() - 1);
    const float frac = static_cast<float>(src - i0);
    out[i] = wav.samples[i0] * (1.0f - frac) + wav.samples[i1] * frac;
  }
  return out;
}

std::vector<int> greedy_decode(MoonshineStreamingModel &model,
                               MoonshineStreamingState *state, int max_tokens) {
  const auto &cfg = model.config;
  model.decoder_reset(state);

  std::vector<int> tokens;
  tokens.push_back(cfg.bos_id);
  std::vector<float> logits(cfg.vocab_size);
  int current = cfg.bos_id;
  for (int step = 0; step < max_tokens; ++step) {
    if (model.decode_step(state, current, logits.data()) != 0) break;
    int best = 0;
    float best_score = logits[0];
    for (int i = 1; i < cfg.vocab_size; ++i) {
      if (logits[i] > best_score) {
        best_score = logits[i];
        best = i;
      }
    }
    tokens.push_back(best);
    current = best;
    if (best == cfg.eos_id) break;
  }
  return tokens;
}

std::vector<int> content_tokens(const std::vector<int> &tokens, int bos_id,
                                int eos_id) {
  std::vector<int> out;
  for (int t : tokens) {
    if (t == bos_id || t == eos_id) continue;
    out.push_back(t);
  }
  return out;
}

int longest_common_prefix(const std::vector<int> &a, const std::vector<int> &b) {
  const int n = static_cast<int>(std::min(a.size(), b.size()));
  int i = 0;
  for (; i < n; ++i) {
    if (a[i] != b[i]) break;
  }
  return i;
}

double median(std::vector<double> values) {
  if (values.empty()) return 0.0;
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + mid, values.end());
  return values[mid];
}

double mean(const std::vector<double> &values) {
  if (values.empty()) return 0.0;
  return std::accumulate(values.begin(), values.end(), 0.0) /
         static_cast<double>(values.size());
}

struct UpdateStats {
  double greedy_ms = 0;
  double speculative_ms = 0;
  double decode_full_nospec_ms = 0;
  int draft_len = 0;
  int accepted_prefix = 0;
  int greedy_token_count = 0;
  int speculative_token_count = 0;
  bool match = false;
  bool nospec_match = false;
  bool self_draft_match = false;  // decode_full(greedy content) == greedy
  int mismatch_at = -1;
  int nospec_mismatch_at = -1;
  int self_draft_mismatch_at = -1;
  float audio_sec = 0;
  std::string greedy_text;
  std::string speculative_text;
  std::string nospec_text;
};

int first_diff(const std::vector<int> &a, const std::vector<int> &b) {
  const int n = static_cast<int>(std::min(a.size(), b.size()));
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) return i;
  }
  if (a.size() != b.size()) return n;
  return -1;
}

std::vector<int> decode_full_vec(MoonshineStreamingModel &model,
                                 MoonshineStreamingState *state,
                                 const int *draft, int draft_len) {
  model.decoder_reset(state);
  int *out = nullptr;
  int out_len = 0;
  if (model.decode_full(state, draft, draft_len, &out, &out_len) != 0) {
    throw std::runtime_error("decode_full failed");
  }
  std::vector<int> tokens(out, out + out_len);
  std::free(out);
  return tokens;
}

struct FileResult {
  std::string path;
  float duration_sec = 0;
  std::vector<UpdateStats> updates;
};

FileResult run_file(MoonshineStreamingModel &model, const std::string &wav_path,
                    float update_interval_sec, int repeats) {
  FileResult result;
  result.path = wav_path;

  const WavData wav = load_wav(wav_path);
  std::vector<float> audio = to_16k(wav);
  if (audio.empty()) {
    throw std::runtime_error("Empty audio after resample: " + wav_path);
  }
  // Trim to whole frontend chunks.
  audio.resize((audio.size() / kChunkSamples) * kChunkSamples);
  result.duration_sec = audio.size() / static_cast<float>(kSampleRate);

  const int samples_per_update =
      std::max(kChunkSamples,
               static_cast<int>(std::lround(update_interval_sec * kSampleRate)));
  const int updates_per_file =
      static_cast<int>(audio.size() / samples_per_update);
  if (updates_per_file < 2) {
    fprintf(stderr, "Skipping %s: need >= 2 updates (got %d)\n",
            wav_path.c_str(), updates_per_file);
    return result;
  }

  MoonshineStreamingState *state = model.create_state();
  const auto &cfg = model.config;
  std::vector<int> prev_content;

  size_t processed = 0;
  while (processed + kChunkSamples <= audio.size()) {
    const size_t update_end =
        std::min(processed + static_cast<size_t>(samples_per_update),
                 audio.size());
    // Feed new audio chunks for this update.
    while (processed + kChunkSamples <= update_end) {
      if (model.process_audio_chunk(state, audio.data() + processed,
                                    kChunkSamples, nullptr) != 0) {
        throw std::runtime_error("process_audio_chunk failed");
      }
      processed += kChunkSamples;
    }

    const bool is_final = (processed >= audio.size());
    int new_frames = 0;
    if (model.encode(state, is_final, &new_frames) != 0) {
      throw std::runtime_error("encode failed");
    }
    if (state->memory_len == 0) {
      if (is_final) break;
      continue;
    }

    // Match decode_full's token budget (memory frames are 20 ms each).
    const float duration_sec = processed / static_cast<float>(kSampleRate);
    const float memory_sec = state->memory_len * 0.020f;
    const int max_tokens = std::min(
        static_cast<int>(std::ceil(memory_sec * 6.5f)),
        cfg.max_seq_len > 0 ? cfg.max_seq_len : 256);

    UpdateStats stats;
    stats.audio_sec = duration_sec;
    stats.draft_len = static_cast<int>(prev_content.size());

    std::vector<double> greedy_times;
    std::vector<double> speculative_times;
    std::vector<double> nospec_times;
    std::vector<int> greedy_tokens;
    std::vector<int> speculative_tokens;

    for (int r = 0; r < repeats; ++r) {
      // Greedy AR (current live path).
      {
        auto t0 = std::chrono::high_resolution_clock::now();
        greedy_tokens = greedy_decode(model, state, max_tokens);
        auto t1 = std::chrono::high_resolution_clock::now();
        greedy_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
      }

      // Speculative decode_full with previous content tokens.
      {
        const int *draft =
            prev_content.empty() ? nullptr : prev_content.data();
        const int draft_len = static_cast<int>(prev_content.size());
        model.decoder_reset(state);
        int *out = nullptr;
        int out_len = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        const int err =
            model.decode_full(state, draft, draft_len, &out, &out_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        speculative_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (err != 0) {
          throw std::runtime_error("decode_full speculative failed");
        }
        speculative_tokens.assign(out, out + out_len);
        std::free(out);
      }

      // decode_full with no draft (same codepath, no speculation).
      {
        model.decoder_reset(state);
        int *out = nullptr;
        int out_len = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        const int err = model.decode_full(state, nullptr, 0, &out, &out_len);
        auto t1 = std::chrono::high_resolution_clock::now();
        nospec_times.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (err != 0) {
          throw std::runtime_error("decode_full nospec failed");
        }
        std::free(out);
      }
    }

    stats.greedy_ms = median(greedy_times);
    stats.speculative_ms = median(speculative_times);
    stats.decode_full_nospec_ms = median(nospec_times);
    stats.greedy_token_count = static_cast<int>(greedy_tokens.size());
    stats.speculative_token_count = static_cast<int>(speculative_tokens.size());

    const std::vector<int> greedy_content =
        content_tokens(greedy_tokens, cfg.bos_id, cfg.eos_id);
    const std::vector<int> speculative_content =
        content_tokens(speculative_tokens, cfg.bos_id, cfg.eos_id);
    stats.accepted_prefix =
        longest_common_prefix(prev_content, greedy_content);
    stats.match = (greedy_content == speculative_content);

    result.updates.push_back(stats);
    prev_content = greedy_content;
    if (is_final) break;
  }

  delete state;
  return result;
}

void print_file_result(const FileResult &result) {
  printf("\n=== %s (%.2fs audio, %zu updates) ===\n", result.path.c_str(),
         result.duration_sec, result.updates.size());
  printf("%6s %8s %10s %10s %10s %8s %8s %6s %6s\n", "upd", "audio_s",
         "greedy_ms", "spec_ms", "nospec_ms", "draft", "accept", "acc%",
         "match");

  std::vector<double> greedy;
  std::vector<double> spec;
  std::vector<double> nospec;
  std::vector<double> accept_rates;
  int mismatches = 0;
  int speculative_updates = 0;

  for (size_t i = 0; i < result.updates.size(); ++i) {
    const UpdateStats &u = result.updates[i];
    const double acc_pct =
        u.draft_len > 0 ? (100.0 * u.accepted_prefix / u.draft_len) : 0.0;
    printf("%6zu %8.2f %10.2f %10.2f %10.2f %8d %8d %5.1f%% %6s\n", i,
           u.audio_sec, u.greedy_ms, u.speculative_ms, u.decode_full_nospec_ms,
           u.draft_len, u.accepted_prefix, acc_pct, u.match ? "yes" : "NO");

    greedy.push_back(u.greedy_ms);
    spec.push_back(u.speculative_ms);
    nospec.push_back(u.decode_full_nospec_ms);
    if (u.draft_len > 0) {
      accept_rates.push_back(acc_pct);
      ++speculative_updates;
    }
    if (!u.match) ++mismatches;
  }

  const double g = mean(greedy);
  const double s = mean(spec);
  const double n = mean(nospec);
  const double speedup = s > 0 ? g / s : 0.0;
  printf("means: greedy=%.2fms  speculative=%.2fms  decode_full=%.2fms  "
         "speedup vs greedy=%.2fx  mean accept=%.1f%%  mismatches=%d/%zu "
         "(drafted updates=%d)\n",
         g, s, n, speedup, mean(accept_rates), mismatches,
         result.updates.size(), speculative_updates);
}

}  // namespace

int main(int argc, char **argv) {
  std::string model_dir = "../test-assets/tiny-streaming-en";
  float update_interval_sec = 0.5f;
  int repeats = 5;
  std::vector<std::string> wav_paths;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
      model_dir = argv[++i];
    } else if ((arg == "-i" || arg == "--interval") && i + 1 < argc) {
      update_interval_sec = std::stof(argv[++i]);
    } else if ((arg == "-r" || arg == "--repeats") && i + 1 < argc) {
      repeats = std::stoi(argv[++i]);
    } else if ((arg == "-w" || arg == "--wav") && i + 1 < argc) {
      wav_paths.push_back(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      fprintf(stderr,
              "Usage: %s [-m model_dir] [-i interval_s] [-r repeats] "
              "[-w wav]...\n",
              argv[0]);
      return 0;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      return 1;
    }
  }

  if (wav_paths.empty()) {
    wav_paths = {
        "../test-assets/beckett.wav",
        "../test-assets/two_cities_16k.wav",
        "../test-assets/endgame_nagg_nell.wav",
    };
  }

  std::setvbuf(stdout, nullptr, _IONBF, 0);
  printf("Model: %s\n", model_dir.c_str());
  printf("Update interval: %.2fs  repeats/update: %d\n", update_interval_sec,
         repeats);

  MoonshineStreamingModel model(/*log_ort_run=*/false);
  const std::string tokenizer_path = model_dir + "/tokenizer.bin";
  if (model.load(model_dir.c_str(), tokenizer_path.c_str(),
                 MOONSHINE_MODEL_ARCH_TINY_STREAMING) != 0) {
    fprintf(stderr, "Failed to load streaming model from %s\n",
            model_dir.c_str());
    return 1;
  }

  // Warmup on first file so ORT / caches settle before timed runs.
  try {
    (void)run_file(model, wav_paths.front(), update_interval_sec, 1);
  } catch (const std::exception &e) {
    fprintf(stderr, "Warmup failed: %s\n", e.what());
  }

  std::vector<double> all_greedy;
  std::vector<double> all_spec;
  std::vector<double> all_accept;
  int all_mismatches = 0;
  size_t all_updates = 0;

  for (const std::string &wav : wav_paths) {
    try {
      FileResult result = run_file(model, wav, update_interval_sec, repeats);
      print_file_result(result);
      for (const UpdateStats &u : result.updates) {
        all_greedy.push_back(u.greedy_ms);
        all_spec.push_back(u.speculative_ms);
        if (u.draft_len > 0) {
          all_accept.push_back(100.0 * u.accepted_prefix / u.draft_len);
        }
        if (!u.match) ++all_mismatches;
      }
      all_updates += result.updates.size();
    } catch (const std::exception &e) {
      fprintf(stderr, "Failed on %s: %s\n", wav.c_str(), e.what());
    }
  }

  printf("\n=== OVERALL ===\n");
  const double g = mean(all_greedy);
  const double s = mean(all_spec);
  printf("mean greedy decode: %.2f ms\n", g);
  printf("mean speculative decode: %.2f ms\n", s);
  printf("mean speedup (greedy/spec): %.2fx\n", s > 0 ? g / s : 0.0);
  printf("mean draft accept rate: %.1f%%\n", mean(all_accept));
  printf("token mismatches: %d / %zu updates\n", all_mismatches, all_updates);
  return all_mismatches == 0 ? 0 : 2;
}
