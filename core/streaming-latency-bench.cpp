// Per-update latency of the streaming path, the way the live transcriber
// drives it: feed `interval` seconds of audio, encode, then decode_full with
// the previous update's tokens as the draft. Audio is cut into segments of at
// most `max_segment` seconds to stand in for the VAD.
//
// Each configuration is one combination of the MoonshineStreamingModel
// latency toggles. "baseline" switches them all off, which reproduces the
// 0.1.6 behaviour on a stateless encoder graph; "optimized" is the default.
// Both run in one process on the same audio so the final transcripts can be
// compared, which is the regression check: faster is only a win if the text
// is the same.
//
// Usage:
//   streaming-latency-bench [-m model_dir]... [-w wav]... [--wav-list file]
//                           [-i interval_s] [-s max_segment_s] [-r repeats]
//                           [--only ends|all|<config>] [--json path]
//                           [--transcripts path]
//
// --transcripts writes one line per (model, config, file):
// model<TAB>config<TAB>file<TAB>text, for scoring against references outside
// this program.
//
// With several -m, the first is the reference for transcript comparison; use
// this to compare a stateless encoder.ort with the cached graph for the same
// model.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
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

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

std::vector<float> load_wav_16k(const std::string &path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<uint8_t> bytes((std::istreambuf_iterator<char>(f)),
                             std::istreambuf_iterator<char>());
  if (bytes.size() < 44 || std::memcmp(bytes.data(), "RIFF", 4) != 0) {
    throw std::runtime_error("not a RIFF file: " + path);
  }
  size_t pos = 12;
  uint16_t channels = 1;
  uint16_t bits = 16;
  uint32_t rate = kSampleRate;
  const uint8_t *data = nullptr;
  size_t data_len = 0;
  while (pos + 8 <= bytes.size()) {
    uint32_t size = 0;
    std::memcpy(&size, &bytes[pos + 4], 4);
    if (std::memcmp(&bytes[pos], "fmt ", 4) == 0) {
      std::memcpy(&channels, &bytes[pos + 10], 2);
      std::memcpy(&rate, &bytes[pos + 12], 4);
      std::memcpy(&bits, &bytes[pos + 22], 2);
    } else if (std::memcmp(&bytes[pos], "data", 4) == 0) {
      data = &bytes[pos + 8];
      data_len = std::min<size_t>(size, bytes.size() - pos - 8);
    }
    pos += 8 + size + (size & 1);
  }
  if (data == nullptr || bits != 16) {
    throw std::runtime_error("need 16-bit PCM: " + path);
  }
  std::vector<float> mono;
  const size_t frames = data_len / (2 * channels);
  mono.reserve(frames);
  for (size_t i = 0; i < frames; ++i) {
    int16_t s = 0;
    std::memcpy(&s, data + i * 2 * channels, 2);
    mono.push_back(s / 32768.0f);
  }
  if (rate == kSampleRate) return mono;
  const double ratio = static_cast<double>(kSampleRate) / rate;
  std::vector<float> out(static_cast<size_t>(mono.size() * ratio));
  for (size_t i = 0; i < out.size(); ++i) {
    const double src = i / ratio;
    const size_t i0 = static_cast<size_t>(src);
    const size_t i1 = std::min(i0 + 1, mono.size() - 1);
    const float frac = static_cast<float>(src - i0);
    out[i] = mono[i0] * (1.0f - frac) + mono[i1] * frac;
  }
  return out;
}

struct Config {
  std::string name;
  bool append_cross_kv;
  int max_realign_attempts;
  bool use_encoder_cache;
};

struct UpdateRecord {
  double encode_ms = 0;
  double decode_ms = 0;
  int draft_len = 0;
  bool diverged = false;
  int64_t decoder_calls = 0;
  int64_t realign_attempts = 0;
  int64_t encoder_frames = 0;
  int64_t cross_kv_frames = 0;
};

struct RunResult {
  std::vector<UpdateRecord> updates;
  std::vector<std::string> transcripts;  // one per segment
};

void apply(MoonshineStreamingModel &model, const Config &cfg) {
  model.append_cross_kv = cfg.append_cross_kv;
  model.max_realign_attempts = cfg.max_realign_attempts;
  model.use_encoder_cache = cfg.use_encoder_cache;
}

RunResult run_audio(MoonshineStreamingModel &model,
                    const std::vector<float> &audio, float interval_sec,
                    float max_segment_sec, bool record) {
  RunResult result;
  const auto &cfg = model.config;
  const size_t per_update = std::max<size_t>(
      kChunkSamples,
      static_cast<size_t>(std::lround(interval_sec * kSampleRate)) /
          kChunkSamples * kChunkSamples);
  const size_t per_segment =
      max_segment_sec > 0
          ? std::max(per_update, static_cast<size_t>(std::lround(
                                     max_segment_sec * kSampleRate)) /
                                     kChunkSamples * kChunkSamples)
          : audio.size();

  size_t seg_start = 0;
  while (seg_start + kChunkSamples <= audio.size()) {
    const size_t seg_end = std::min(audio.size(), seg_start + per_segment);
    std::unique_ptr<MoonshineStreamingState> state(model.create_state());
    std::vector<int> prev_tokens;
    size_t processed = seg_start;
    while (processed + kChunkSamples <= seg_end) {
      const size_t update_end = std::min(seg_end, processed + per_update);
      const bool is_final = update_end + kChunkSamples > seg_end;

      UpdateRecord rec;
      const MoonshineStreamingModel::Stats before = model.stats;
      auto t0 = Clock::now();
      while (processed + kChunkSamples <= update_end) {
        if (model.process_audio_chunk(state.get(), audio.data() + processed,
                                      kChunkSamples, nullptr) != 0) {
          throw std::runtime_error("process_audio_chunk failed");
        }
        processed += kChunkSamples;
      }
      int new_frames = 0;
      if (model.encode(state.get(), is_final, &new_frames) != 0) {
        throw std::runtime_error("encode failed");
      }
      rec.encode_ms = ms_since(t0);
      if (state->memory_len == 0) {
        if (is_final) break;
        continue;
      }

      model.decoder_reset(state.get());
      t0 = Clock::now();
      int *out = nullptr;
      int out_len = 0;
      const int *draft = prev_tokens.empty() ? nullptr : prev_tokens.data();
      if (model.decode_full(state.get(), draft,
                            static_cast<int>(prev_tokens.size()), &out,
                            &out_len) != 0) {
        throw std::runtime_error("decode_full failed");
      }
      rec.decode_ms = ms_since(t0);
      std::unique_ptr<int, decltype(&std::free)> owned(out, &std::free);
      std::vector<int> tokens(out, out + out_len);

      rec.draft_len = static_cast<int>(prev_tokens.size());
      rec.diverged =
          !prev_tokens.empty() &&
          (tokens.size() < prev_tokens.size() ||
           !std::equal(prev_tokens.begin(), prev_tokens.end(), tokens.begin()));
      rec.decoder_calls = model.stats.decoder_calls - before.decoder_calls;
      rec.realign_attempts =
          model.stats.realign_attempts - before.realign_attempts;
      rec.encoder_frames = model.stats.encoder_frames - before.encoder_frames;
      rec.cross_kv_frames =
          model.stats.cross_kv_frames - before.cross_kv_frames;
      if (record) result.updates.push_back(rec);

      prev_tokens = tokens;
      if (is_final) {
        std::vector<int64_t> with_bos;
        with_bos.push_back(cfg.bos_id);
        for (int t : tokens) with_bos.push_back(t);
        result.transcripts.push_back(model.tokens_to_text(with_bos));
        break;
      }
    }
    seg_start = seg_end;
  }
  return result;
}

double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t idx =
      std::min(v.size() - 1, static_cast<size_t>(p * (v.size() - 1) + 0.5));
  return v[idx];
}

double mean(const std::vector<double> &v) {
  return v.empty() ? 0.0 : std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

struct Summary {
  double total_mean, total_p50, total_p90, total_max;
  double encode_mean, decode_mean;
  double calls_mean, realign_mean, enc_frames_mean, ckv_frames_mean;
  double diverged_pct;
  size_t updates;
};

Summary summarize(const std::vector<UpdateRecord> &updates) {
  std::vector<double> total, enc, dec, calls, realign, encf, ckvf;
  int diverged = 0, drafted = 0;
  for (const UpdateRecord &u : updates) {
    total.push_back(u.encode_ms + u.decode_ms);
    enc.push_back(u.encode_ms);
    dec.push_back(u.decode_ms);
    calls.push_back(static_cast<double>(u.decoder_calls));
    realign.push_back(static_cast<double>(u.realign_attempts));
    encf.push_back(static_cast<double>(u.encoder_frames));
    ckvf.push_back(static_cast<double>(u.cross_kv_frames));
    if (u.draft_len > 0) {
      ++drafted;
      if (u.diverged) ++diverged;
    }
  }
  Summary s;
  s.total_mean = mean(total);
  s.total_p50 = percentile(total, 0.5);
  s.total_p90 = percentile(total, 0.9);
  s.total_max = percentile(total, 1.0);
  s.encode_mean = mean(enc);
  s.decode_mean = mean(dec);
  s.calls_mean = mean(calls);
  s.realign_mean = mean(realign);
  s.enc_frames_mean = mean(encf);
  s.ckv_frames_mean = mean(ckvf);
  s.diverged_pct = drafted ? 100.0 * diverged / drafted : 0.0;
  s.updates = updates.size();
  return s;
}

std::string json_escape(const std::string &s) {
  std::string out;
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (c == '\n') {
      out += "\\n";
    } else {
      out += c;
    }
  }
  return out;
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<std::string> model_dirs;
  std::vector<std::string> wav_paths;
  float interval_sec = 0.5f;
  float max_segment_sec = 8.0f;
  int repeats = 3;
  std::string only = "ends";
  std::string json_path;
  std::string transcripts_path;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        fprintf(stderr, "%s needs a value\n", arg.c_str());
        exit(1);
      }
      return argv[++i];
    };
    if (arg == "-m" || arg == "--model") {
      model_dirs.push_back(next());
    } else if (arg == "-w" || arg == "--wav") {
      wav_paths.push_back(next());
    } else if (arg == "-i" || arg == "--interval") {
      interval_sec = std::stof(next());
    } else if (arg == "-s" || arg == "--segment") {
      max_segment_sec = std::stof(next());
    } else if (arg == "-r" || arg == "--repeats") {
      repeats = std::stoi(next());
    } else if (arg == "--only") {
      only = next();
    } else if (arg == "--json") {
      json_path = next();
    } else if (arg == "--transcripts") {
      transcripts_path = next();
    } else if (arg == "--wav-list") {
      std::ifstream list(next());
      std::string line;
      while (std::getline(list, line)) {
        if (!line.empty()) wav_paths.push_back(line);
      }
    } else {
      fprintf(stderr, "Unknown argument: %s\n", arg.c_str());
      return 1;
    }
  }
  if (model_dirs.empty()) model_dirs = {"../test-assets/tiny-streaming-en"};
  if (wav_paths.empty()) {
    wav_paths = {"../test-assets/beckett.wav",
                 "../test-assets/two_cities_16k.wav",
                 "../test-assets/endgame_nagg_nell.wav"};
  }

  // The single-feature rows attribute both the speed-up and any transcript
  // change to one optimization at a time.
  std::vector<Config> configs = {
      {"baseline", false, 0, false}, {"append-kv", true, 0, false},
      {"realign", false, 3, false},  {"enc-cache", false, 0, true},
      {"optimized", true, 3, true},
  };
  if (!only.empty()) {
    configs.erase(
        std::remove_if(configs.begin(), configs.end(),
                       [&](const Config &c) {
                         return only != "all" && c.name != only &&
                                !(only == "ends" && (c.name == "baseline" ||
                                                     c.name == "optimized"));
                       }),
        configs.end());
    if (configs.empty()) {
      fprintf(stderr,
              "--only must be all, ends, or one of baseline, append-kv, "
              "realign, enc-cache, optimized\n");
      return 1;
    }
  }

  std::vector<std::vector<float>> audios;
  for (const std::string &w : wav_paths) {
    std::vector<float> a = load_wav_16k(w);
    a.resize(a.size() / kChunkSamples * kChunkSamples);
    audios.push_back(std::move(a));
  }

  std::setvbuf(stdout, nullptr, _IONBF, 0);
  printf("interval=%.2fs segment<=%.1fs repeats=%d files=%zu\n", interval_sec,
         max_segment_sec, repeats, wav_paths.size());

  std::vector<std::string> reference_transcripts;
  std::ofstream transcripts_out;
  if (!transcripts_path.empty()) {
    transcripts_out.open(transcripts_path);
  }
  std::string json = "[\n";
  bool first_json = true;
  int exit_code = 0;

  for (const std::string &model_dir : model_dirs) {
    MoonshineStreamingModel model(/*log_ort_run=*/false);
    const std::string tokenizer = model_dir + "/tokenizer.bin";
    if (model.load(model_dir.c_str(), tokenizer.c_str(),
                   MOONSHINE_MODEL_ARCH_TINY_STREAMING) != 0) {
      fprintf(stderr, "Failed to load %s\n", model_dir.c_str());
      return 1;
    }
    printf("\n=== %s (depth %d, encoder graph: %s) ===\n", model_dir.c_str(),
           model.config.depth,
           model.encoder_has_cache ? "cached" : "stateless");
    printf("%-10s %8s %8s %8s %8s | %7s %7s | %6s %6s %7s %7s | %6s\n",
           "config", "mean", "p50", "p90", "max", "enc_ms", "dec_ms", "calls",
           "realn", "encfrm", "ckvfrm", "div%");

    for (const Config &cfg : configs) {
      apply(model, cfg);
      // Warm-up pass on the first file, untimed.
      (void)run_audio(model, audios.front(), interval_sec, max_segment_sec,
                      false);
      std::vector<UpdateRecord> all;
      std::vector<std::string> transcripts;
      for (int r = 0; r < repeats; ++r) {
        transcripts.clear();
        for (size_t f = 0; f < audios.size(); ++f) {
          RunResult res =
              run_audio(model, audios[f], interval_sec, max_segment_sec, true);
          all.insert(all.end(), res.updates.begin(), res.updates.end());
          transcripts.insert(transcripts.end(), res.transcripts.begin(),
                             res.transcripts.end());
          if (r == 0 && transcripts_out.is_open()) {
            std::string joined;
            for (const std::string &t : res.transcripts) {
              if (!joined.empty()) joined += ' ';
              joined += t;
            }
            transcripts_out << model_dir << '\t' << cfg.name << '\t'
                            << wav_paths[f] << '\t' << joined << '\n';
          }
        }
      }
      const Summary s = summarize(all);
      printf(
          "%-10s %8.2f %8.2f %8.2f %8.2f | %7.2f %7.2f | %6.2f %6.2f %7.1f "
          "%7.1f | %5.1f%%\n",
          cfg.name.c_str(), s.total_mean, s.total_p50, s.total_p90, s.total_max,
          s.encode_mean, s.decode_mean, s.calls_mean, s.realign_mean,
          s.enc_frames_mean, s.ckv_frames_mean, s.diverged_pct);

      int changed = 0;
      if (reference_transcripts.empty()) {
        reference_transcripts = transcripts;
      } else {
        for (size_t i = 0;
             i < transcripts.size() && i < reference_transcripts.size(); ++i) {
          if (transcripts[i] != reference_transcripts[i]) {
            ++changed;
            if (changed <= 10) {
              printf(
                  "  segment %zu differs from reference:\n    ref: %s\n"
                  "    got: %s\n",
                  i, reference_transcripts[i].c_str(), transcripts[i].c_str());
            }
          }
        }
        printf("  transcripts: %d of %zu segments differ from reference\n",
               changed, transcripts.size());
      }

      json += first_json ? "" : ",\n";
      first_json = false;
      char buf[1024];
      snprintf(buf, sizeof(buf),
               "  {\"model\": \"%s\", \"config\": \"%s\", \"encoder\": \"%s\", "
               "\"updates\": %zu, \"mean_ms\": %.3f, \"p50_ms\": %.3f, "
               "\"p90_ms\": %.3f, \"max_ms\": %.3f, \"encode_ms\": %.3f, "
               "\"decode_ms\": %.3f, \"decoder_calls\": %.3f, "
               "\"realign_attempts\": %.3f, \"encoder_frames\": %.1f, "
               "\"cross_kv_frames\": %.1f, \"diverged_pct\": %.2f, "
               "\"segments_changed\": %d, \"segments\": %zu}",
               json_escape(model_dir).c_str(), cfg.name.c_str(),
               model.encoder_has_cache ? "cached" : "stateless", s.updates,
               s.total_mean, s.total_p50, s.total_p90, s.total_max,
               s.encode_mean, s.decode_mean, s.calls_mean, s.realign_mean,
               s.enc_frames_mean, s.ckv_frames_mean, s.diverged_pct, changed,
               transcripts.size());
      json += buf;
    }
  }
  json += "\n]\n";
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    out << json;
    printf("\nwrote %s\n", json_path.c_str());
  }
  return exit_code;
}
