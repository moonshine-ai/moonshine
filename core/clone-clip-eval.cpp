// Structural before/after eval for voice-clone clip word-boundary refinement.
// Loads tiny-en once, scores VAD-exact clips vs refine_clone_clip_bounds using
// full-file word timestamps as the oracle. No ZipVoice synthesis.
//
// Run from test-assets/ (see scripts/eval-clone-clip.sh).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "clone-clip.h"
#include "debug-utils.h"
#include "moonshine-c-api.h"
#include "speech-clip.h"

namespace {

constexpr float kRequestedDuration = 4.0f;
constexpr float kMaxExtension = 1.5f;
constexpr float kEndPad = 0.05f;
constexpr float kCutEpsilon = 1e-3f;

struct FixtureAudio {
  std::string name;
  std::vector<float> pcm;
  int32_t sample_rate = 0;
  // If >= 0, force this VAD window start instead of running extract_speech_clip.
  float forced_start = -1.0f;
};

struct ClipScore {
  float start = 0.0f;
  float end = 0.0f;
  int mid_word_cuts = 0;
  int head_cuts = 0;
  int tail_cuts = 0;
  float overshoot = 0.0f;
  int transcript_mismatches = 0;
  std::string transcript;
};

std::vector<CloneClipWord> collect_words(const transcript_t *transcript) {
  std::vector<CloneClipWord> words;
  if (transcript == nullptr) {
    return words;
  }
  for (uint64_t i = 0; i < transcript->line_count; ++i) {
    const transcript_line_t &line = transcript->lines[i];
    for (uint64_t j = 0; j < line.word_count; ++j) {
      const transcript_word_t &w = line.words[j];
      if (w.text == nullptr || w.text[0] == '\0') {
        continue;
      }
      words.push_back({w.text, w.start, w.end});
    }
  }
  return words;
}

int count_boundary_cuts(const std::vector<CloneClipWord> &words, float edge) {
  int cuts = 0;
  for (const CloneClipWord &w : words) {
    if (w.start + kCutEpsilon < edge && edge + kCutEpsilon < w.end) {
      ++cuts;
    }
  }
  return cuts;
}

std::vector<std::string> words_fully_inside(
    const std::vector<CloneClipWord> &words, float start, float end) {
  std::vector<std::string> out;
  for (const CloneClipWord &w : words) {
    if (w.start + kCutEpsilon >= start && w.end <= end + kCutEpsilon) {
      out.push_back(w.text);
    }
  }
  return out;
}

std::vector<std::string> split_transcript(const std::string &text) {
  std::vector<std::string> out;
  std::string cur;
  for (char c : text) {
    if (c == ' ' || c == '\t' || c == '\n') {
      if (!cur.empty()) {
        out.push_back(cur);
        cur.clear();
      }
    } else {
      cur.push_back(c);
    }
  }
  if (!cur.empty()) {
    out.push_back(cur);
  }
  return out;
}

int transcript_mismatch_count(const std::string &transcript,
                              const std::vector<CloneClipWord> &oracle,
                              float start, float end) {
  const std::vector<std::string> got = split_transcript(transcript);
  const std::vector<std::string> want = words_fully_inside(oracle, start, end);
  // Order-sensitive compare; count positions that differ plus length delta.
  const size_t n = std::max(got.size(), want.size());
  int mismatches = static_cast<int>(n);
  const size_t m = std::min(got.size(), want.size());
  mismatches -= static_cast<int>(m);
  for (size_t i = 0; i < m; ++i) {
    if (got[i] != want[i]) {
      ++mismatches;
    }
  }
  return mismatches;
}

ClipScore score_clip(float start, float end, float requested_duration,
                     const std::vector<CloneClipWord> &oracle,
                     const std::string &transcript, bool score_transcript) {
  ClipScore s;
  s.start = start;
  s.end = end;
  s.head_cuts = count_boundary_cuts(oracle, start);
  s.tail_cuts = count_boundary_cuts(oracle, end);
  s.mid_word_cuts = s.head_cuts + s.tail_cuts;
  s.overshoot = std::max(0.0f, end - (start + requested_duration));
  s.transcript = transcript;
  if (score_transcript) {
    s.transcript_mismatches =
        transcript_mismatch_count(transcript, oracle, start, end);
  }
  return s;
}

bool load_fixture(const char *path, const char *name, float forced_start,
                  FixtureAudio *out) {
  float *data = nullptr;
  size_t n = 0;
  int32_t sr = 0;
  if (!load_wav_data(path, &data, &n, &sr) || data == nullptr || n == 0) {
    std::fprintf(stderr, "Failed to load %s\n", path);
    return false;
  }
  out->name = name;
  out->pcm.assign(data, data + n);
  out->sample_rate = sr;
  out->forced_start = forced_start;
  std::free(data);
  return true;
}

float pick_synthetic_midword_start(const std::vector<CloneClipWord> &words,
                                   float audio_duration) {
  // Find a word whose midpoint leaves at least requested_duration of audio
  // after a cut through the middle of that word — guarantees a baseline tail cut.
  for (const CloneClipWord &w : words) {
    const float mid = 0.5f * (w.start + w.end);
    const float start = mid - kRequestedDuration;
    if (start >= 0.05f && mid + 0.05f < audio_duration &&
        (w.end - w.start) > 0.15f) {
      return start;
    }
  }
  // Fallback: cut 4s ending mid-file.
  return std::max(0.0f, audio_duration * 0.35f - kRequestedDuration);
}

}  // namespace

int main(int argc, char **argv) {
  const char *model_path = "tiny-en";
  if (argc > 1) {
    model_path = argv[1];
  }

  const auto wall_start = std::chrono::steady_clock::now();

  moonshine_option_t options[] = {
      {.name = "word_timestamps", .value = "true"},
      {.name = "identify_speakers", .value = "false"},
  };
  const int32_t handle = moonshine_load_transcriber_from_files(
      model_path, MOONSHINE_MODEL_ARCH_TINY, options, 2,
      moonshine_get_version());
  if (handle < 0) {
    std::fprintf(stderr, "Failed to load model at %s (err %d)\n", model_path,
                 handle);
    return 1;
  }

  std::vector<FixtureAudio> fixtures;
  FixtureAudio f;
  if (!load_fixture("two_cities_16k.wav", "two_cities_16k", -1.0f, &f)) {
    return 1;
  }
  fixtures.push_back(std::move(f));
  if (!load_fixture("beckett.wav", "beckett", -1.0f, &f)) {
    return 1;
  }
  fixtures.push_back(std::move(f));
  if (!load_fixture("../python/src/moonshine_voice/assets/clone-test.wav",
                    "clone-test", -1.0f, &f)) {
    return 1;
  }
  fixtures.push_back(std::move(f));

  // Synthetic stress case derived from two_cities after we have oracle words.
  std::printf("%-18s %-8s %8s %8s %8s %10s %10s %s\n", "fixture", "mode",
              "start", "end", "cuts", "overshoot", "tx_mis", "transcript");
  std::printf("%s\n", std::string(100, '-').c_str());

  int baseline_fixtures_with_tail = 0;
  int refined_fixtures_with_tail = 0;
  int baseline_total_cuts = 0;
  int refined_total_cuts = 0;
  float baseline_overshoot_sum = 0.0f;
  float refined_overshoot_sum = 0.0f;
  int case_count = 0;

  auto run_case = [&](const FixtureAudio &fx,
                      const std::vector<CloneClipWord> &oracle) {
    float vad_start = fx.forced_start;
    if (vad_start < 0.0f) {
      const SpeechClip vad = extract_speech_clip(
          fx.pcm.data(), fx.pcm.size(), fx.sample_rate,
          SpeechClipOptions{.clip_duration_seconds = kRequestedDuration,
                            .minimum_speech_seconds = 2.0f,
                            .vad_threshold = 0.5f});
      if (!vad.is_complete) {
        std::printf("%-18s %-8s (incomplete VAD clip; skipped)\n",
                    fx.name.c_str(), "baseline");
        return;
      }
      vad_start = vad.start_time_seconds;
    }

    const float baseline_end = vad_start + kRequestedDuration;
    const ClipScore baseline =
        score_clip(vad_start, baseline_end, kRequestedDuration, oracle, "",
                   /*score_transcript=*/false);

    const CloneClipBounds refined = refine_clone_clip_bounds(
        vad_start, kRequestedDuration, oracle, kMaxExtension, kEndPad);
    const ClipScore after =
        score_clip(refined.start_seconds, refined.end_seconds,
                   kRequestedDuration, oracle, refined.transcript,
                   /*score_transcript=*/true);

    auto print_row = [&](const char *mode, const ClipScore &s) {
      std::printf("%-18s %-8s %8.3f %8.3f %8d %10.3f %10d %.60s%s\n",
                  fx.name.c_str(), mode, s.start, s.end, s.mid_word_cuts,
                  s.overshoot, s.transcript_mismatches, s.transcript.c_str(),
                  s.transcript.size() > 60 ? "…" : "");
    };
    print_row("baseline", baseline);
    print_row("refined", after);

    ++case_count;
    baseline_total_cuts += baseline.mid_word_cuts;
    refined_total_cuts += after.mid_word_cuts;
    baseline_overshoot_sum += baseline.overshoot;
    refined_overshoot_sum += after.overshoot;
    if (baseline.tail_cuts > 0) {
      ++baseline_fixtures_with_tail;
    }
    if (after.tail_cuts > 0) {
      ++refined_fixtures_with_tail;
    }
  };

  std::vector<CloneClipWord> two_cities_words;
  for (const FixtureAudio &fx : fixtures) {
    transcript_t *transcript = nullptr;
    const int32_t err = moonshine_transcribe_without_streaming(
        handle, const_cast<float *>(fx.pcm.data()),
        static_cast<uint64_t>(fx.pcm.size()), fx.sample_rate, 0, &transcript);
    if (err != 0 || transcript == nullptr) {
      std::fprintf(stderr, "ASR failed for %s (err %d)\n", fx.name.c_str(),
                   err);
      moonshine_free_transcriber(handle);
      return 1;
    }
    const std::vector<CloneClipWord> oracle = collect_words(transcript);
    if (fx.name == "two_cities_16k") {
      two_cities_words = oracle;
    }
    run_case(fx, oracle);
  }

  // Synthetic: force a window that ends mid-word on two_cities.
  if (!fixtures.empty() && !two_cities_words.empty()) {
    FixtureAudio syn = fixtures.front();
    syn.name = "synthetic_midword";
    const float dur =
        static_cast<float>(syn.pcm.size()) / static_cast<float>(syn.sample_rate);
    syn.forced_start = pick_synthetic_midword_start(two_cities_words, dur);
    run_case(syn, two_cities_words);
  }

  std::printf("%s\n", std::string(100, '-').c_str());
  const float inv = case_count > 0 ? 1.0f / static_cast<float>(case_count) : 0.0f;
  std::printf("SUMMARY  cases=%d\n", case_count);
  std::printf("  baseline: mid_word_cuts=%d  tail_cut_rate=%.2f  "
              "mean_overshoot_s=%.3f\n",
              baseline_total_cuts,
              static_cast<float>(baseline_fixtures_with_tail) * inv,
              baseline_overshoot_sum * inv);
  std::printf("  refined:  mid_word_cuts=%d  tail_cut_rate=%.2f  "
              "mean_overshoot_s=%.3f\n",
              refined_total_cuts,
              static_cast<float>(refined_fixtures_with_tail) * inv,
              refined_overshoot_sum * inv);

  const auto wall_end = std::chrono::steady_clock::now();
  const double elapsed_s =
      std::chrono::duration<double>(wall_end - wall_start).count();
  std::printf("  wall_time_s=%.2f\n", elapsed_s);

  moonshine_free_transcriber(handle);

  if (refined_total_cuts != 0) {
    std::fprintf(stderr,
                 "FAIL: refined path still has %d mid-word cut(s)\n",
                 refined_total_cuts);
    return 2;
  }
  std::printf("PASS: refined mid_word_cuts == 0\n");
  return 0;
}
