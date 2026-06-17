#include "test_app.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "app_common.h"
#include "audio_config.h"
#include "classes.h"
#include "feature_generation/feature_generation.h"
#include "mel_tables.h"
#include "model_data.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "stt/stt.h"
#include "test_clips.h"

#ifdef SPELLING_TINY_PROFILE_OPS
#include "op_profiler.h"
#endif

#ifdef SPELLING_TINY_TTS
#include "tts_service.h"
#endif

#ifdef SPELLING_TINY_VAD
#include "kiss_fftr.h"
#include "vad/vad.h"
#include "vad_config.h"
#include "vad_mel_tables.h"
#include "vad_model_data.h"
#endif

// Catch a partial regenerate of generated/ at compile time rather than feeding
// the model a clip whose sample rate / length it does not expect.
static_assert(spelling::kEmbeddedClipSampleRate == spelling::kSampleRate,
              "test_clips.h sample rate disagrees with audio_config.h -- "
              "regenerate cpp-tiny/example-rp2350/generated/ together");
static_assert(spelling::kEmbeddedClipNumSamples == spelling::kClipNumSamples,
              "test_clips.h clip length disagrees with audio_config.h -- "
              "regenerate cpp-tiny/example-rp2350/generated/ together");

namespace spelling {

namespace {

#ifdef SPELLING_TINY_PROFILE_OPS
// File-scope (BSS) so its ~4 KiB of event arrays don't land on the stack.
OpProfiler g_op_profiler;
#endif

#ifdef SPELLING_TINY_VAD
// VAD/STT table-sharing consistency (shared 512-pt FFT + Hann window).
static_assert(kVadMelTableNMels == kVadNMels,
              "vad_mel_tables.h n_mels != vad_config.h kVadNMels; regenerate");
static_assert(kVadMelTableNFft == kVadNFft,
              "vad_mel_tables.h n_fft != vad_config.h kVadNFft; regenerate");
static_assert(kMelTableNFft == kVadMelTableNFft,
              "STT/VAD n_fft differ -- cannot share the Hann window");

// Streaming VAD demo over the embedded clips (each treated as a 1 s stream):
// one FFT per 32 ms hop through the MelStreamer -> int8 TinyVadCNN ->
// moving-average segmenter, reporting detected segments + per-hop latency. The
// VAD runs in the STT arena's temporary region and RETURNS so the sweep below
// reuses the same arena.
void RunVadDemo(uint8_t* arena, std::size_t arena_size, kiss_fftr_state* fft) {
  printf("\n=== VAD demo (streaming) ===\n");
  printf(
      "VAD model: %u bytes  n_mels=%d  window_frames=%d  hop=%d  "
      "(shared 512-pt FFT + Hann window)\n",
      g_vad_model_data_size, kVadNMels, kVadWindowFrames, kVadHop);
  fflush(stdout);

  MelStreamer streamer(kVadNMels, kVadWindowFrames, kVadNFft, kMelWindow,
                       kVadMelNzOff, kVadMelNzIdx, kVadMelNzVal, fft,
                       /*eps=*/1e-6f);

  Vad vad(g_vad_model_data, g_vad_model_data_size, arena, arena_size, kVadNMels,
          kVadWindowFrames);
  printf("VAD arena used: %u bytes (%.1f%% of %u)\n",
         static_cast<unsigned>(vad.arena_used_bytes()),
         100.0 * static_cast<double>(vad.arena_used_bytes()) /
             static_cast<double>(arena_size),
         static_cast<unsigned>(arena_size));
  printf("--- begin VAD ---\n");

  const int hop = kVadHop;
  const int n_blocks = kClipNumSamples / hop;  // non-overlapping 512-blocks

  VadSegmenter seg(kVadThreshold, kVadSmoothWindowFrames, hop,
                   kVadLookBehindSamples, kVadMaxSegmentSamples);

  constexpr float kInt16ToFp32 = 1.0f / 32768.0f;
  for (int ci = 0; ci < kNumEmbeddedClips; ++ci) {
    const EmbeddedClip& clip = kEmbeddedClips[ci];
    const unsigned n = (clip.num_samples < (unsigned)kClipNumSamples)
                           ? clip.num_samples
                           : (unsigned)kClipNumSamples;
    for (unsigned i = 0; i < n; ++i)
      g_waveform[i] = static_cast<float>(clip.samples[i]) * kInt16ToFp32;
    for (unsigned i = n; i < (unsigned)kClipNumSamples; ++i)
      g_waveform[i] = 0.0f;

    streamer.Reset();
    seg.Start();
    int n_segs = 0;
    std::size_t last_start = 0, last_end = 0;
    float max_prob = 0.0f;
    const absolute_time_t t0 = get_absolute_time();
    for (int k = 0; k < n_blocks; ++k) {
      streamer.PushHop(&g_waveform[k * hop]);
      float* feats = vad.feature_scratch();
      streamer.BuildModelInput(feats);
      const float p = vad.Predict(feats);
      if (p > max_prob) max_prob = p;
      const VadEvent ev = seg.ProcessFrame(p);
      if (ev == VadEvent::kSpeechEnd) {
        ++n_segs;
        last_start = seg.segment_start_sample();
        last_end = seg.segment_end_sample();
      }
    }
    if (seg.Finish() == VadEvent::kSpeechEnd) {
      ++n_segs;
      last_start = seg.segment_start_sample();
      last_end = seg.segment_end_sample();
    }
    const int64_t us = absolute_time_diff_us(t0, get_absolute_time());
    printf(
        "[%2d/%2d] %-5s segs=%d  max_p=%.2f  last=[%.2f..%.2fs]  "
        "vad_us/hop=%lld\n",
        ci + 1, kNumEmbeddedClips, clip.label, n_segs,
        static_cast<double>(max_prob),
        static_cast<double>(last_start) / kVadSampleRate,
        static_cast<double>(last_end) / kVadSampleRate,
        static_cast<long long>(n_blocks > 0 ? us / n_blocks : us));
  }
  printf("--- end VAD ---\n");
}
#endif  // SPELLING_TINY_VAD

}  // namespace

void RunTestApp(unsigned led_pin) {
  char line[96];
  std::snprintf(
      line, sizeof(line), "Clips:   %d embedded (%u samples @ %d Hz each)\n",
      kNumEmbeddedClips, static_cast<unsigned>(kEmbeddedClipNumSamples),
      kEmbeddedClipSampleRate);
  fputs(line, stdout);
  fflush(stdout);

  kiss_fftr_state* shared_fft = nullptr;
#ifdef SPELLING_TINY_VAD
  // One shared 512-pt twiddle state for the VAD front-end and the STT log-mel.
  shared_fft = kiss_fftr_alloc(kNFft, /*inverse_fft=*/0, nullptr, nullptr);
  if (shared_fft == nullptr) {
    printf("[boot] kiss_fftr_alloc(shared) failed\n");
    while (true) { /* halt */
    }
  }
  RunVadDemo(g_tensor_arena, kTensorArenaSize, shared_fft);
#endif

  // Log-mel front-end (flash tables -> no heap / boot trig). Reuses the shared
  // FFT when the VAD demo allocated one; otherwise allocates its own.
  printf("[boot] constructing log-mel front-end...\n");
  fflush(stdout);
  LogMelParams lm{};
  lm.sample_rate = kSampleRate;
  lm.n_fft = kNFft;
  lm.hop_length = kHopLength;
  lm.win_length = kWinLength;
  lm.n_mels = kNMels;
  lm.f_min = kFMin;
  lm.f_max = kFMax;
  lm.target_frames = kTargetFrames;
  lm.eps = 1e-6f;
  lm.center = true;
  lm.precomputed_window = kMelWindow;
  lm.precomputed_window_len = kMelTableNFft;
  lm.precomputed_nz_off = kMelNzOff;
  lm.precomputed_nz_idx = kMelNzIdx;
  lm.precomputed_nz_val = kMelNzVal;
  lm.precomputed_nz_total = kMelNzTotal;
  lm.external_fft = shared_fft;  // null is fine -- LogMel allocates its own
  LogMelSpectrogram log_mel(lm);
  printf("[boot] log-mel ready\n");
  fflush(stdout);

  // TFLM classifier (halts loudly on any sanity failure).
  printf("[boot] constructing classifier (arena=%u bytes)...\n",
         static_cast<unsigned>(kTensorArenaSize));
  fflush(stdout);
#ifdef SPELLING_TINY_PROFILE_OPS
  tflite::MicroProfilerInterface* profiler = &g_op_profiler;
  printf("[boot] per-op profiling ENABLED (SPELLING_TINY_PROFILE_OPS)\n");
#else
  tflite::MicroProfilerInterface* profiler = nullptr;
#endif
  Classifier classifier(g_spelling_model_data, g_spelling_model_data_size,
                        g_tensor_arena, kTensorArenaSize, kNMels, kTargetFrames,
                        kNumClasses, profiler);
  printf("[boot] classifier ready\n");
  fflush(stdout);
  printf("Arena used: %u bytes (%.1f%% of %u)\n",
         static_cast<unsigned>(classifier.arena_used_bytes()),
         100.0 * static_cast<double>(classifier.arena_used_bytes()) /
             static_cast<double>(kTensorArenaSize),
         static_cast<unsigned>(kTensorArenaSize));
  printf("Input  scale=%.6f  zp=%d\n",
         static_cast<double>(classifier.input_quant().scale),
         classifier.input_quant().zero_point);
  printf("Output scale=%.6f  zp=%d\n",
         static_cast<double>(classifier.output_quant().scale),
         classifier.output_quant().zero_point);
  printf("--- begin ---\n");

  int n_correct = 0;
  uint64_t total_us = 0;
  for (int ci = 0; ci < kNumEmbeddedClips; ++ci) {
    const EmbeddedClip& clip = kEmbeddedClips[ci];

    // int16 PCM -> static fp32 waveform. 1/32768 (not 32767) matches the
    // inverse of the generator's _fp32_to_int16_pcm, keeping the round-trip
    // exact at integer values.
    const absolute_time_t t_fm0 = get_absolute_time();
    constexpr float kInt16ToFp32 = 1.0f / 32768.0f;
    for (unsigned int i = 0; i < clip.num_samples; ++i) {
      g_waveform[i] = static_cast<float>(clip.samples[i]) * kInt16ToFp32;
    }
    float* features = classifier.feature_scratch();
    log_mel.Compute(g_waveform, clip.num_samples, features);
    const int64_t fm_us = absolute_time_diff_us(t_fm0, get_absolute_time());

    float logits[kNumClasses];
#ifdef SPELLING_TINY_PROFILE_OPS
    g_op_profiler.Reset();
#endif
    const absolute_time_t t_in0 = get_absolute_time();
    classifier.Run(features, logits);
    const int64_t in_us = absolute_time_diff_us(t_in0, get_absolute_time());
#ifdef SPELLING_TINY_PROFILE_OPS
    // Per-clip inference cost is near-constant, so one representative clip's
    // breakdown is enough.
    if (ci == 0) {
      g_op_profiler.Report("clip0");
      fflush(stdout);
      sleep_ms(20);
    }
#endif

    const int pred = Argmax(logits, kNumClasses);
    const float pred_prob = SoftmaxProb(logits, kNumClasses, pred);
    const bool ok = (pred == clip.label_index);
    if (ok) ++n_correct;
    total_us += static_cast<uint64_t>(fm_us + in_us);

    printf(
        "[%2d/%2d]  exp=%-5s got=%-5s p=%.3f  "
        "log_mel_us=%6lld  infer_us=%6lld  %s\n",
        ci + 1, kNumEmbeddedClips, clip.label, kClassLabels[pred],
        static_cast<double>(pred_prob), static_cast<long long>(fm_us),
        static_cast<long long>(in_us), ok ? "OK" : "FAIL");
  }
  printf("--- end ---\n");
  printf("accuracy: %d/%d = %.1f%%\n", n_correct, kNumEmbeddedClips,
         100.0 * n_correct / static_cast<double>(kNumEmbeddedClips));
  printf("avg latency: %.2f ms / clip (log_mel + infer)\n",
         static_cast<double>(total_us) / 1000.0 /
             static_cast<double>(kNumEmbeddedClips));

#ifdef SPELLING_TINY_TTS
  // Recognition is done; the classifier won't Invoke again, so hand its arena
  // to the streaming TTS service (its own USB command loop; never returns).
  printf("[boot] entering TTS service (reusing %u-byte arena for synthesis)\n",
         static_cast<unsigned>(kTensorArenaSize));
  fflush(stdout);
  RunTtsService(g_tensor_arena, kTensorArenaSize);
#endif

  // Idle heartbeat so the monitor can tell the firmware is still alive.
  uint64_t tick = 0;
  while (true) {
    gpio_put(led_pin, (tick & 1u) ? 1 : 0);
    if ((tick % 5) == 0) {
      printf("alive: t=%llus  acc=%d/%d\n",
             static_cast<long long unsigned>(tick), n_correct,
             kNumEmbeddedClips);
    }
    ++tick;
    sleep_ms(1000);
  }
}

}  // namespace spelling
