// Investigate greedy vs decode_full token mismatches.
//
// For each streaming update, compares:
//   1) greedy decode_step loop
//   2) decode_full with no draft
//   3) decode_full with previous tokens (speculative)
//   4) decode_full with greedy's own tokens as draft (oracle self-draft)
//
// (4) is the key check: if feeding the greedy result back into decode_full
// does not round-trip, the speculative multi-token path is not equivalent
// to step-by-step AR (independent of draft quality).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "file-utils.h"
#include "moonshine-c-api.h"
#include "moonshine-streaming-model.h"

namespace {

constexpr int kChunkSamples = 1280;
constexpr int kSampleRate = 16000;

struct WavData {
  std::vector<float> samples;
  int32_t sample_rate = 0;
};

WavData load_wav(const std::string &path) {
  WavData out;
  FILE *file = std::fopen(path.c_str(), "rb");
  if (!file) throw std::runtime_error("Failed to open WAV: " + path);

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
  std::fseek(file, 6, SEEK_CUR);
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

std::vector<int> content_tokens(const std::vector<int> &tokens, int bos,
                                int eos) {
  std::vector<int> out;
  for (int t : tokens) {
    if (t == bos || t == eos) continue;
    out.push_back(t);
  }
  return out;
}

int first_diff(const std::vector<int> &a, const std::vector<int> &b) {
  const int n = static_cast<int>(std::min(a.size(), b.size()));
  for (int i = 0; i < n; ++i) {
    if (a[i] != b[i]) return i;
  }
  if (a.size() != b.size()) return n;
  return -1;
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

// Compare step-by-step teacher-forced logits vs one multi-token forward.
void compare_multitoken_vs_step(MoonshineStreamingModel &model,
                                MoonshineStreamingState *state,
                                const std::vector<int> &content, int bos_id) {
  if (content.empty()) return;

  std::vector<int64_t> with_bos;
  with_bos.push_back(bos_id);
  for (int t : content) with_bos.push_back(t);

  // Multi-token
  model.decoder_reset(state);
  std::vector<float> multi_logits(with_bos.size() * model.config.vocab_size);
  {
    std::vector<int> as_int(with_bos.begin(), with_bos.end());
    if (model.decode_tokens(state, as_int.data(),
                            static_cast<int>(as_int.size()),
                            multi_logits.data()) != 0) {
      printf("  multitoken compare: decode_tokens failed\n");
      return;
    }
  }

  // Step-by-step
  model.decoder_reset(state);
  std::vector<float> step_logits(with_bos.size() * model.config.vocab_size);
  std::vector<float> one(model.config.vocab_size);
  for (size_t i = 0; i < with_bos.size(); ++i) {
    if (model.decode_step(state, static_cast<int>(with_bos[i]), one.data()) !=
        0) {
      printf("  multitoken compare: decode_step failed at %zu\n", i);
      return;
    }
    std::memcpy(step_logits.data() + i * model.config.vocab_size, one.data(),
                model.config.vocab_size * sizeof(float));
  }

  auto argmax = [&](const float *logits) {
    int best = 0;
    float best_score = logits[0];
    for (int i = 1; i < model.config.vocab_size; ++i) {
      if (logits[i] > best_score) {
        best_score = logits[i];
        best = i;
      }
    }
    return best;
  };

  int first_argmax_diff = -1;
  int first_logit_diff = -1;
  double max_abs_diff = 0.0;
  for (size_t t = 0; t < with_bos.size(); ++t) {
    const float *m = multi_logits.data() + t * model.config.vocab_size;
    const float *s = step_logits.data() + t * model.config.vocab_size;
    for (int i = 0; i < model.config.vocab_size; ++i) {
      const double d = std::fabs(static_cast<double>(m[i] - s[i]));
      if (d > max_abs_diff) max_abs_diff = d;
      if (d > 1e-3 && first_logit_diff < 0) {
        first_logit_diff = static_cast<int>(t);
      }
    }
    if (argmax(m) != argmax(s) && first_argmax_diff < 0) {
      first_argmax_diff = static_cast<int>(t);
    }
  }

  printf("  multi vs step-by-step teacher-force on %zu tokens (+BOS):\n",
         content.size());
  printf("    max|Δlogit|=%.6g  first_logit_diff_pos=%d  "
         "first_argmax_diff_pos=%d\n",
         max_abs_diff, first_logit_diff, first_argmax_diff);
}

}  // namespace

int main(int argc, char **argv) {
  std::string model_dir =
      "/Users/petewarden/projects/moonshine/test-assets/tiny-streaming-en";
  std::string wav_path =
      "/Users/petewarden/projects/moonshine/test-assets/endgame_nagg_nell.wav";
  float update_interval_sec = 0.5f;
  int max_details = 6;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
      model_dir = argv[++i];
    } else if ((arg == "-w" || arg == "--wav") && i + 1 < argc) {
      wav_path = argv[++i];
    } else if ((arg == "-i" || arg == "--interval") && i + 1 < argc) {
      update_interval_sec = std::stof(argv[++i]);
    } else if ((arg == "-n" || arg == "--max-details") && i + 1 < argc) {
      max_details = std::stoi(argv[++i]);
    }
  }

  std::setvbuf(stdout, nullptr, _IONBF, 0);
  printf("model=%s\nwav=%s\ninterval=%.2f\n", model_dir.c_str(),
         wav_path.c_str(), update_interval_sec);

  MoonshineStreamingModel model(false);
  if (model.load(model_dir.c_str(), (model_dir + "/tokenizer.bin").c_str(),
                 MOONSHINE_MODEL_ARCH_TINY_STREAMING) != 0) {
    fprintf(stderr, "Failed to load model\n");
    return 1;
  }

  WavData wav = load_wav(wav_path);
  std::vector<float> audio = to_16k(wav);
  audio.resize((audio.size() / kChunkSamples) * kChunkSamples);

  MoonshineStreamingState *state = model.create_state();
  const auto &cfg = model.config;
  std::vector<int> prev_content;

  int counts_spec = 0, counts_nospec = 0, counts_self = 0, updates = 0;
  int details = 0;
  size_t processed = 0;
  size_t upd = 0;

  const int samples_per_update = std::max(
      kChunkSamples,
      static_cast<int>(std::lround(update_interval_sec * kSampleRate)));

  while (processed + kChunkSamples <= audio.size()) {
    const size_t update_end = std::min(
        processed + static_cast<size_t>(samples_per_update), audio.size());
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
      ++upd;
      if (is_final) break;
      continue;
    }

    const float memory_sec = state->memory_len * 0.020f;
    const int max_tokens =
        std::min(static_cast<int>(std::ceil(memory_sec * 6.5f)),
                 cfg.max_seq_len > 0 ? cfg.max_seq_len : 256);

    const std::vector<int> greedy = greedy_decode(model, state, max_tokens);
    const std::vector<int> greedy_c =
        content_tokens(greedy, cfg.bos_id, cfg.eos_id);

    const std::vector<int> nospec = decode_full_vec(model, state, nullptr, 0);
    const std::vector<int> nospec_c =
        content_tokens(nospec, cfg.bos_id, cfg.eos_id);

    const std::vector<int> spec = decode_full_vec(
        model, state, prev_content.empty() ? nullptr : prev_content.data(),
        static_cast<int>(prev_content.size()));
    const std::vector<int> spec_c =
        content_tokens(spec, cfg.bos_id, cfg.eos_id);

    const std::vector<int> self_draft = decode_full_vec(
        model, state, greedy_c.empty() ? nullptr : greedy_c.data(),
        static_cast<int>(greedy_c.size()));
    const std::vector<int> self_c =
        content_tokens(self_draft, cfg.bos_id, cfg.eos_id);

    const bool nospec_ok = greedy_c == nospec_c;
    const bool spec_ok = greedy_c == spec_c;
    const bool self_ok = greedy_c == self_c;
    ++updates;
    if (!nospec_ok) ++counts_nospec;
    if (!spec_ok) ++counts_spec;
    if (!self_ok) ++counts_self;

    if ((!nospec_ok || !spec_ok || !self_ok) && details < max_details) {
      const float audio_sec = processed / static_cast<float>(kSampleRate);
      printf("\n=== upd=%zu audio=%.2fs mem=%d max_tokens=%d draft_len=%zu ===\n",
             upd, audio_sec, state->memory_len, max_tokens, prev_content.size());
      printf("greedy_len=%zu nospec_len=%zu spec_len=%zu self_len=%zu\n",
             greedy_c.size(), nospec_c.size(), spec_c.size(), self_c.size());
      printf("greedy vs nospec: %s (diff_at=%d)\n", nospec_ok ? "MATCH" : "DIFF",
             first_diff(greedy_c, nospec_c));
      printf("greedy vs spec:   %s (diff_at=%d)\n", spec_ok ? "MATCH" : "DIFF",
             first_diff(greedy_c, spec_c));
      printf("greedy vs self-draft decode_full: %s (diff_at=%d)\n",
             self_ok ? "MATCH" : "DIFF", first_diff(greedy_c, self_c));
      printf("greedy text: %s\n",
             model
                 .tokens_to_text(
                     std::vector<int64_t>(greedy.begin(), greedy.end()))
                 .c_str());
      printf("nospec text: %s\n",
             model
                 .tokens_to_text(
                     std::vector<int64_t>(nospec.begin(), nospec.end()))
                 .c_str());
      printf("spec text:   %s\n",
             model
                 .tokens_to_text(std::vector<int64_t>(spec.begin(), spec.end()))
                 .c_str());
      printf("self text:   %s\n",
             model
                 .tokens_to_text(
                     std::vector<int64_t>(self_draft.begin(), self_draft.end()))
                 .c_str());

      // If self-draft fails, check whether multi-token ≠ step-by-step.
      if (!self_ok) {
        compare_multitoken_vs_step(model, state, greedy_c, cfg.bos_id);
      }
      ++details;
    }

    prev_content = greedy_c;
    ++upd;
    if (is_final) break;
  }

  delete state;

  printf("\n=== SUMMARY (%s) ===\n", wav_path.c_str());
  printf("updates=%d\n", updates);
  printf("greedy vs decode_full(nospec) mismatches: %d\n", counts_nospec);
  printf("greedy vs decode_full(prev draft) mismatches: %d\n", counts_spec);
  printf("greedy vs decode_full(self draft) mismatches: %d\n", counts_self);
  return (counts_nospec + counts_spec + counts_self) == 0 ? 0 : 2;
}
