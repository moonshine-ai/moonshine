#include "audio_service.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <optional>

#include "audio_config.h"
#include "classes.h"
#include "feature_generation/feature_generation.h"
#include "mel_tables.h"
#include "model_data.h"
#include "pico/stdlib.h"
#include "spelling_labels.h"
#include "stt/stt.h"
#include "tts/tts.h"  // tts::StreamSynth / VoiceParams / DefaultVoiceParams
#include "vad/vad.h"
#include "vad_config.h"
#include "vad_mel_tables.h"
#include "vad_model_data.h"

namespace spelling {

namespace {

// After the device finishes speaking its reply, keep ignoring (muting) the
// input for this long so the tail of our own reply / room echo can't instantly
// re-trigger the VAD. Input is still DRAINED during this window so the pipe
// never backs up.
constexpr int kMuteTailMs = 250;

// Persistent front-end (outside the arena), built once by RecognizerInit. The
// streaming VAD mel front-end shares the FFT + STT Hann window with the STT
// log-mel. Both hold multi-KB working buffers; keeping them in BSS (not the
// small 4 KB core stack) prevents PushHop's FFT scratch and TFLM's deep
// Invoke() frames from corrupting them. `optional` so they construct lazily
// once the shared FFT is known (RecognizerInit), yet still live in BSS.
std::optional<MelStreamer> g_streamer;
std::optional<LogMelSpectrogram> g_stt_log_mel;

// Default voice for the readback/prompt synth; constructed on first Speak().
std::optional<tts::VoiceParams> g_voice;

}  // namespace

void RecognizerInit(kiss_fftr_state* fft) {
  // The streaming VAD front-end: shares the 512-pt FFT and 512-tap Hann window
  // with the STT log-mel (same n_fft, never concurrent).
  g_streamer.emplace(spelling::kVadNMels, spelling::kVadWindowFrames,
                     spelling::kVadNFft, spelling::kMelWindow,
                     spelling::kVadMelNzOff, spelling::kVadMelNzIdx,
                     spelling::kVadMelNzVal, fft, /*eps=*/1e-6f);

  spelling::LogMelParams slm{};
  slm.sample_rate = spelling::kSampleRate;
  slm.n_fft = spelling::kNFft;
  slm.hop_length = spelling::kHopLength;
  slm.win_length = spelling::kWinLength;
  slm.n_mels = spelling::kNMels;
  slm.f_min = spelling::kFMin;
  slm.f_max = spelling::kFMax;
  slm.target_frames = spelling::kTargetFrames;
  slm.eps = 1e-6f;
  slm.center = true;
  slm.precomputed_window = spelling::kMelWindow;
  slm.precomputed_window_len = spelling::kMelTableNFft;
  slm.precomputed_nz_off = spelling::kMelNzOff;
  slm.precomputed_nz_idx = spelling::kMelNzIdx;
  slm.precomputed_nz_val = spelling::kMelNzVal;
  slm.precomputed_nz_total = spelling::kMelNzTotal;
  slm.external_fft = fft;  // shared 512-pt FFT (STT n_fft == 512)
  g_stt_log_mel.emplace(slm);
}

int RecognizeOne(AudioInput& input, uint8_t* arena, std::size_t arena_size,
                 float* window, int window_samples, float* out_prob) {
  const int WS = window_samples;      // kClipNumSamples (16000)
  const int HOP = spelling::kVadHop;  // 512

  // The segmenter holds only a small smoothing window; a fresh per-call
  // instance keeps the listening state clean (Start() resets it anyway).
  spelling::VadSegmenter seg(
      spelling::kVadThreshold, spelling::kVadSmoothWindowFrames, HOP,
      spelling::kVadLookBehindSamples, spelling::kVadMaxSegmentSamples);

  int16_t hop[512];  // kVadHop

  // ---- VAD listening phase (Vad lives in the arena) ----
  std::size_t seg_start = 0, seg_end = 0;
  {
    spelling::Vad vad(g_vad_model_data, g_vad_model_data_size, arena,
                      arena_size, spelling::kVadNMels,
                      spelling::kVadWindowFrames);
    g_streamer->Reset();
    seg.Start();
    for (int i = 0; i < WS; ++i) window[i] = 0.0f;
    printf("[audio] listening (vad arena=%u bytes)\n",
           static_cast<unsigned>(vad.arena_used_bytes()));
    fflush(stdout);

    bool got_clip = false;
    int idle_beats = 0;
    while (!got_clip) {
      if (!input.ReadHop(hop, HOP)) {
        printf("[audio] idle, waiting for hops (%d)\n", ++idle_beats);
        fflush(stdout);
        continue;
      }
      // Slide the 1 s window left by one hop and append the new samples.
      std::memmove(window, window + HOP,
                   static_cast<std::size_t>(WS - HOP) * sizeof(float));
      for (int i = 0; i < HOP; ++i) {
        window[WS - HOP + i] = static_cast<float>(hop[i]) * (1.0f / 32768.0f);
      }
      // One FFT for the newest hop; VAD inference; segmenter.
      float* feats = vad.feature_scratch();
      g_streamer->PushHop(&window[WS - HOP]);
      g_streamer->BuildModelInput(feats);
      const float p = vad.Predict(feats);
      const spelling::VadEvent ev = seg.ProcessFrame(p);
      if (ev == spelling::VadEvent::kSpeechStart) {
        printf("VAD start\n");
        fflush(stdout);
      } else if (ev == spelling::VadEvent::kSpeechEnd) {
        got_clip = true;
        seg_start = seg.segment_start_sample();
        seg_end = seg.segment_end_sample();
      }
    }

    // Front-align the 1 s clip in `window`: shift so the segment start sits
    // at index 0, zero-padding the tail (matches ExtractClipFrontAligned).
    const std::size_t now = seg.samples_processed();
    const std::size_t wstart =
        (now >= static_cast<std::size_t>(WS)) ? (now - WS) : 0u;
    const std::size_t off = (seg_start > wstart) ? (seg_start - wstart) : 0u;
    if (off > 0 && off < static_cast<std::size_t>(WS)) {
      std::memmove(window, window + off,
                   (static_cast<std::size_t>(WS) - off) * sizeof(float));
      for (std::size_t i = static_cast<std::size_t>(WS) - off;
           i < static_cast<std::size_t>(WS); ++i) {
        window[i] = 0.0f;
      }
    }
    printf("VAD end dur=%.2f\n",
           static_cast<double>(seg_end - seg_start) /
               static_cast<double>(spelling::kVadSampleRate));
    fflush(stdout);
  }  // Vad destroyed; arena free for the classifier.

  // ---- STT phase (Classifier lives in the arena) ----
  int pred = -1;
  float prob = 0.0f;
  {
    spelling::Classifier clf(
        g_spelling_model_data, g_spelling_model_data_size, arena, arena_size,
        spelling::kNMels, spelling::kTargetFrames, spelling::kNumClasses);
    float* feats = clf.feature_scratch();
    g_stt_log_mel->Compute(window, static_cast<std::size_t>(WS), feats);
    float logits[spelling::kNumClasses];
    clf.Run(feats, logits);
    pred = spelling::Argmax(logits, spelling::kNumClasses);
    prob = spelling::SoftmaxProb(logits, spelling::kNumClasses, pred);
    printf("RESULT %s %.3f\n", spelling::kClassLabels[pred],
           static_cast<double>(prob));
    fflush(stdout);
  }  // Classifier destroyed; arena free for the synth.

  if (out_prob != nullptr) *out_prob = prob;
  return pred;
}

void Speak(const char* text, AudioOutput& output, AudioInput& input,
           uint8_t* arena, std::size_t arena_size) {
  if (!g_voice.has_value()) g_voice.emplace(tts::DefaultVoiceParams());

  // ---- TTS phase (StreamSynth uses the arena as working memory) ----
  {
    tts::StreamSynth synth(*g_voice, arena, arena_size);
    tts::StreamOptions opts;
    opts.sample_rate = 22050.0f;
    if (synth.BeginText(text, opts) == tts::kStreamOk) {
      output.Begin(synth.sample_rate(), synth.total_samples());
      float chunk[256];
      int16_t pcm[256];
      for (int n; (n = synth.Read(chunk, 256)) > 0;) {
        for (int i = 0; i < n; ++i) {
          float s = chunk[i];
          if (s > 1.0f)
            s = 1.0f;
          else if (s < -1.0f)
            s = -1.0f;
          pcm[i] = static_cast<int16_t>(std::lrintf(s * 32767.0f));
        }
        output.Write(pcm, n);
        // Mute-but-drain: keep consuming incoming hops and throw them away so
        // our own audio (and any echo) can't queue up or feed the VAD.
        input.Drain();
      }
      output.End();
    }
  }  // synth destroyed; arena free again.

  // Mute tail: ignore (but keep draining) the input for kMuteTailMs after the
  // reply, so the reply's tail / room echo doesn't immediately re-trigger
  // speech detection.
  const absolute_time_t mute_end = make_timeout_time_ms(kMuteTailMs);
  while (!time_reached(mute_end)) {
    input.Drain();
    sleep_us(500);
  }
}

void RunAudioService(AudioInput& input, AudioOutput& output, uint8_t* arena,
                     std::size_t arena_size, kiss_fftr_state* fft,
                     float* window, int window_samples) {
  printf("[audio] service entered (arena=%u, WS=%d, HOP=%d)\n",
         static_cast<unsigned>(arena_size), window_samples, spelling::kVadHop);
  fflush(stdout);

  RecognizerInit(fft);

  printf(
      "\n[audio] ready: stream 16 kHz int16 mono hops in; "
      "results + spoken reply come back.\n");
  fflush(stdout);

  for (;;) {
    float prob = 0.0f;
    const int pred =
        RecognizeOne(input, arena, arena_size, window, window_samples, &prob);

    // Ignore low-confidence recognitions: the RESULT is already logged; we just
    // don't speak them (and skip the mute tail since there's no reply to echo).
    if (prob < kMinResultProb) {
      printf("[audio] ignoring '%s' (p=%.3f < %.2f); not speaking\n",
             spelling::kClassLabels[pred], static_cast<double>(prob),
             static_cast<double>(kMinResultProb));
      fflush(stdout);
      continue;
    }

    // Speak the letter's sound-alike word ("bee" for 'b'); digits already
    // arrive as words. See spelling_labels.h.
    Speak(SpokenForLabel(spelling::kClassLabels[pred]), output, input, arena,
          arena_size);
  }
}

}  // namespace spelling
