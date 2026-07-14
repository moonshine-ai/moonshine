#include "audio_service.h"

#include <algorithm>
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
#include "neural_tts/neural_tts.h"
#include "vad/vad.h"
#include "vad_config.h"
#include "vad_mel_tables.h"
#include "vad_model_data.h"

// SPELLING_AUDIO_DIAG gates the verbose recognizer diagnostics: per-hop
// "listening" heartbeats, mic signal stats, idle/first-hop notices, the VAD
// start probability detail, the level-normalization log, and the captured-clip
// playback log. It's defined for the diagnostic / test builds (echo,
// echo_hardware) and left OFF for the product WiFi-setup apps (wifi,
// wifi_hardware) so their serial log stays quiet -- only the protocol/status
// lines (VAD start / VAD end / RESULT) remain. The separate SPELLING_DUMP_STT_USB
// flag controls the binary STT-clip capture dump.

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

// Flash-resident neural TTS pack (generated/neural_tts_pack.S), used by the
// readback/prompt synth in Speak().
extern "C" const uint8_t g_neural_tts_pack[];

// Master playback volume for the spoken reply (see SetTtsVolume). The streaming
// TTS already lands near -1 dBFS, so 1.0 == full scale; this just scales it.
float g_tts_volume = 1.0f;

// Float scratch for one VAD hop. The capture window is int16 (half the SRAM of
// fp32), but MelStreamer::PushHop() wants fp32, so we convert the newest hop
// here. In BSS (not the 4 KB core stack) and only ever touched on core 0.
float g_vad_hop_f[spelling::kVadHop];

}  // namespace

void SetTtsVolume(float volume) {
  if (volume < 0.0f) volume = 0.0f;
  if (volume > 4.0f) volume = 4.0f;  // generous ceiling; >1 will clip the reply
  g_tts_volume = volume;
}

float TtsVolume() { return g_tts_volume; }

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
                 int16_t* window, int window_samples, float* out_prob) {
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
    for (int i = 0; i < WS; ++i) window[i] = 0;
#ifdef SPELLING_AUDIO_DIAG
    printf("[audio] listening (vad arena=%u bytes, threshold=%.2f, "
           "smooth=%d frames)\n",
           static_cast<unsigned>(vad.arena_used_bytes()),
           static_cast<double>(spelling::kVadThreshold),
           spelling::kVadSmoothWindowFrames);
    fflush(stdout);

    // Heartbeat: ~1 s at 16 kHz / 512-sample hops. Prints whether hops are
    // actually arriving (if ReadHop blocks on a silent/unwired I2S mic there
    // are NONE), the live mic signal level (rms/peak of the raw hop), and the
    // VAD probability stats -- so a "nothing happens" hang is immediately
    // diagnosable: no heartbeats => no hops; flat ~0 level => no mic data;
    // signal present but p below threshold => acoustic/threshold issue.
    // Diagnostic-only (compiled out of the quiet WiFi-setup apps).
    constexpr int kHeartbeatHops = 32;
    long hops_seen = 0;
    float p_max = 0.0f;
    float p_last = 0.0f;
    int peak_max = 0;
    // Per-hop wall-clock budget. A 512-sample hop is 32 ms of real audio. The
    // PIO RX FIFO is only a few samples deep with no DMA, so any compute between
    // ReadHop calls that exceeds ~0 ms drops samples (FIFO overflow) -- the gap
    // shows up as a discontinuity/spike each hop and wrecks the VAD. Measuring
    // the compute time tells us how much audio we're losing between hops.
    uint64_t compute_us_accum = 0;
    uint64_t readhop_us_accum = 0;
    absolute_time_t hop_t0 = get_absolute_time();
    int idle_beats = 0;
#endif

    bool got_clip = false;
    while (!got_clip) {
      if (!input.ReadHop(hop, HOP)) {
#ifdef SPELLING_AUDIO_DIAG
        printf("[audio] idle, waiting for hops (%d)\n", ++idle_beats);
        fflush(stdout);
#endif
        continue;
      }
#ifdef SPELLING_AUDIO_DIAG
      const absolute_time_t t_read_end = get_absolute_time();
      readhop_us_accum += absolute_time_diff_us(hop_t0, t_read_end);
      if (hops_seen == 0) {
        printf("[audio] first hop received; mic is producing data\n");
        fflush(stdout);
      }
      ++hops_seen;
      // Cheap per-hop signal stats from the raw int16 hop (pre-normalization).
      int64_t sumsq = 0;
      int peak = 0;
      for (int i = 0; i < HOP; ++i) {
        const int a = hop[i] < 0 ? -static_cast<int>(hop[i]) : hop[i];
        if (a > peak) peak = a;
        sumsq += static_cast<int64_t>(hop[i]) * hop[i];
      }
      if (peak > peak_max) peak_max = peak;
#endif
      // Slide the 1 s window left by one hop and append the new samples. No gain
      // here -- the VAD front-end is robust to level, and the STT clip is
      // level-normalized in one shot before its log-mel (see the note just
      // before g_stt_log_mel->Compute below). We keep the raw level for the VAD.
      //
      // The slide is ESSENTIAL: without it every hop overwrites the same tail
      // region and window[0 .. WS-HOP) stays at its initial zeros, so the STT
      // only ever sees the single newest 32 ms hop (silence at kSpeechEnd) --
      // i.e. "silence + a tiny burst at the end", rms ~0. The VAD is unaffected
      // because it reads the freshly written window[WS-HOP] directly.
      std::memmove(window, window + HOP,
                   static_cast<std::size_t>(WS - HOP) * sizeof(int16_t));
      // Store the raw int16 hop in the window, and build the fp32 copy the VAD
      // front-end consumes (PushHop wants floats; the window is int16).
      for (int i = 0; i < HOP; ++i) {
        window[WS - HOP + i] = hop[i];
        g_vad_hop_f[i] = static_cast<float>(hop[i]) * (1.0f / 32768.0f);
      }
      // One FFT for the newest hop; VAD inference; segmenter.
      float* feats = vad.feature_scratch();
      g_streamer->PushHop(g_vad_hop_f);
      g_streamer->BuildModelInput(feats);
      const float p = vad.Predict(feats);
      const spelling::VadEvent ev = seg.ProcessFrame(p);
#ifdef SPELLING_AUDIO_DIAG
      p_last = p;
      if (p > p_max) p_max = p;
      const absolute_time_t t_compute_end = get_absolute_time();
      compute_us_accum += absolute_time_diff_us(t_read_end, t_compute_end);
      hop_t0 = t_compute_end;  // next ReadHop interval starts here
      if (hops_seen % kHeartbeatHops == 0) {
        const double rms = std::sqrt(static_cast<double>(sumsq) / HOP);
        // compute_us is real audio (us) lost to FIFO overflow each hop (no DMA);
        // >0 means the I2S capture is gappy. read_us ~32000 is the real-time hop.
        printf("[audio] listening: hops=%ld p=%.3f pmax=%.3f rms=%.0f "
               "peak=%d peakmax=%d read_us=%lu compute_us=%lu\n",
               hops_seen, static_cast<double>(p_last),
               static_cast<double>(p_max), rms, peak, peak_max,
               static_cast<unsigned long>(readhop_us_accum / kHeartbeatHops),
               static_cast<unsigned long>(compute_us_accum / kHeartbeatHops));
        fflush(stdout);
        p_max = 0.0f;  // reset per window so we see recent activity, not all-time
        peak_max = 0;
        compute_us_accum = 0;
        readhop_us_accum = 0;
      }
#endif
      if (ev == spelling::VadEvent::kSpeechStart) {
#ifdef SPELLING_AUDIO_DIAG
        printf("VAD start (p=%.3f rms-peak=%d)\n", static_cast<double>(p),
               peak);
#else
        printf("VAD start\n");
#endif
        fflush(stdout);
      } else if (ev == spelling::VadEvent::kSpeechEnd) {
        got_clip = true;
        seg_start = seg.segment_start_sample();
        seg_end = seg.segment_end_sample();
      }
    }

    // Energy-centroid alignment of the 1 s clip in `window`.
    //
    // The newest sample is ALWAYS at window[WS-1] (we append each hop at the
    // tail), so absolute sample `a` lives at window index (WS - now + a) whether
    // or not the rolling buffer has filled yet. The captured segment occupies
    // the tail [WS - seg_len, WS) of the window (seg_len == now - seg_start,
    // clamped to the buffer).
    //
    // We used to FRONT-align (shift so seg_start sat at index 0). But
    // segment_start is a noisy anchor: the moving-average smoothing lags the
    // true onset by up to ~0.6 s, the 0.5 s look-behind only cancels that on
    // average, and merged/early-triggered segments push it hundreds of ms off.
    // Front-aligning therefore clips leading consonants (the E-set killer) or
    // strands the word at the back of the window. Instead we find the
    // energy-weighted centroid of the captured audio and slide the buffer so
    // the spoken word sits centred at WS/2, zero-padding the vacated side. In
    // host simulation this recovered essentially the entire streaming-VAD
    // accuracy gap, +~7 pts overall and +~11 pts on the E-set (see
    // scripts/test_streaming_vad.py --align-sweep).
    const long now = static_cast<long>(seg.samples_processed());
    const long seg_len = now - static_cast<long>(seg_start);
    long region_start = (seg_len >= static_cast<long>(WS)) ? 0
                                                           : (static_cast<long>(WS) - seg_len);
    if (region_start < 0) region_start = 0;
    const std::size_t centroid = spelling::EnergyCentroidIndex(
        window, static_cast<std::size_t>(region_start),
        static_cast<std::size_t>(WS));
    const long shift = static_cast<long>(centroid) - (static_cast<long>(WS) / 2);
    if (shift > 0 && shift < static_cast<long>(WS)) {
      // Word sits late in the buffer: slide left, zero-pad the tail.
      std::memmove(window, window + shift,
                   (static_cast<std::size_t>(WS) - static_cast<std::size_t>(shift)) *
                       sizeof(int16_t));
      for (long i = static_cast<long>(WS) - shift; i < static_cast<long>(WS); ++i) {
        window[static_cast<std::size_t>(i)] = 0;
      }
    } else if (shift < 0 && -shift < static_cast<long>(WS)) {
      // Word sits early in the buffer: slide right, zero-pad the head.
      const long r = -shift;
      std::memmove(window + r, window,
                   (static_cast<std::size_t>(WS) - static_cast<std::size_t>(r)) *
                       sizeof(int16_t));
      for (long i = 0; i < r; ++i) window[static_cast<std::size_t>(i)] = 0;
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

    // ---- Level normalization (THE fix for quiet I2S mic STT) ----
    // The log-mel front-end does log(mel_power + eps) with eps=1e-6 BEFORE its
    // per-clip mean/std standardization. That standardization only cancels a
    // constant input gain when mel_power >> eps. The on-board I2S mic (SPH0645)
    // is ~50-100x quieter than the line/USB audio the model was trained and
    // validated on -- speech rms is only ~1e-3 of full scale -- so its mel_power
    // sits down near eps: the spectral valleys clamp to the log(eps) floor, the
    // formant contrast collapses, and the classifier sees near-garbage (uniform
    // ~1/51 outputs) even though the energy-based VAD fires correctly. Lifting
    // the clip to a healthy RMS puts mel_power well above eps, where the rest is
    // genuinely level-invariant. We ONLY amplify (gain > 1): already-loud USB
    // audio sits above eps and is left untouched (this is a no-op for it).
    {
      // The window is int16; work in normalized [-1, 1] sample units (divide by
      // 32768) so kTargetRms keeps its fp32 meaning, then apply the gain back in
      // the int16 domain (a pure rescale -- no precision lost vs the old fp32
      // window, since the samples were int16 to begin with).
      constexpr float kInv = 1.0f / 32768.0f;
      double sumsq = 0.0;
      for (int i = 0; i < WS; ++i) {
        const double s = static_cast<double>(window[i]) * kInv;
        sumsq += s * s;
      }
      const float rms = static_cast<float>(std::sqrt(sumsq / WS));
      constexpr float kTargetRms = 0.05f;  // healthy line level (mel_power >> eps)
      constexpr float kMaxGain = 128.0f;   // don't blow up a near-silent window
      if (rms > 1e-6f) {
        float gain = kTargetRms / rms;
        if (gain > kMaxGain) gain = kMaxGain;
        if (gain > 1.0f) {  // never attenuate loud (USB) audio
          for (int i = 0; i < WS; ++i) {
            float v = static_cast<float>(window[i]) * gain;
            if (v > 32767.0f)
              v = 32767.0f;
            else if (v < -32768.0f)
              v = -32768.0f;
            window[i] = static_cast<int16_t>(std::lrintf(v));
          }
#ifdef SPELLING_AUDIO_DIAG
          printf("[stt] level norm: rms=%.4f gain=%.1fx\n",
                 static_cast<double>(rms), static_cast<double>(gain));
          fflush(stdout);
#endif
        }
      }
    }

#ifdef SPELLING_DUMP_STT_USB
    // Debug capture: stream the EXACT buffer the STT log-mel is about to consume
    // (front-aligned AND level-normalized -- i.e. precisely "what the model
    // receives") over the USB CDC so the host can save + inspect/listen to it.
    // Framing matches usb_audio_io.cc: a text header line, the raw int16 LE
    // samples, then a text trailer. scripts/capture_stt.py extracts these into
    // numbered wav files. Harmless on the PWM-speaker path (this goes out the
    // CDC log pipe, independent of the audio output sink).
    {
      // The window is already int16 LE -- stream it straight out.
      printf("STTIN %d %d\n", spelling::kVadSampleRate, WS);
      fflush(stdout);
      fwrite(window, sizeof(int16_t), static_cast<size_t>(WS), stdout);
      fflush(stdout);
      printf("\nEND\n");
      fflush(stdout);
    }
#endif

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

// Play back the front-aligned capture window (what the classifier saw) before
// the TTS reply so mic wiring/gain issues are audible during bring-up.
void PlayCapturedClip(const int16_t* window, int num_samples,
                      AudioOutput& output, AudioInput& input) {
  // The capture window is typically only a few percent of full scale; peak-
  // normalize so it uses the whole output range (an un-normalized clip is buried
  // in noise / quantization hash on a small speaker). Clamp the gain so a quiet
  // window isn't amplified into pure noise. Work in int16 units throughout.
  int peak = 0;
  for (int i = 0; i < num_samples; ++i) {
    const int a = window[i] < 0 ? -static_cast<int>(window[i]) : window[i];
    if (a > peak) peak = a;
  }
  float gain = 1.0f;
  if (peak > 16) {  // ~5e-4 of full scale
    gain = 0.9f * 32767.0f / static_cast<float>(peak);
    if (gain > 32.0f) gain = 32.0f;
  }

#ifdef SPELLING_AUDIO_DIAG
  printf("CLIP playback: %d samples @ %d Hz (gain %.1fx)\n", num_samples,
         spelling::kVadSampleRate, static_cast<double>(gain));
  fflush(stdout);
#endif

  output.Begin(spelling::kVadSampleRate, num_samples, "CLIP");
  int16_t pcm[256];
  for (int i = 0; i < num_samples;) {
    const int n = std::min(256, num_samples - i);
    for (int j = 0; j < n; ++j) {
      float v = static_cast<float>(window[i + j]) * gain;
      if (v > 32767.0f)
        v = 32767.0f;
      else if (v < -32768.0f)
        v = -32768.0f;
      pcm[j] = static_cast<int16_t>(std::lrintf(v));
    }
    output.Write(pcm, n);
    input.Drain();
    i += n;
  }
  output.End();
}

namespace {

// Speak()'s per-chunk sink: scale by the reply volume, forward to the audio
// output, and keep draining the input so our own audio (and any echo) can't
// queue up or feed the VAD.
struct SpeakSink {
  AudioOutput* output;
  AudioInput* input;
  float volume;
  int raw_peak;  // max |sample| from the synth BEFORE volume (level diag only)
};

void SpeakEmit(void* user, const int16_t* samples, int n) {
  auto* s = static_cast<SpeakSink*>(user);
  int16_t pcm[256];
  for (int i = 0; i < n;) {
    const int m = std::min(256, n - i);
    for (int j = 0; j < m; ++j) {
      const int a =
          samples[i + j] < 0 ? -static_cast<int>(samples[i + j]) : samples[i + j];
      if (a > s->raw_peak) s->raw_peak = a;
      float v = static_cast<float>(samples[i + j]) * s->volume;
      if (v > 32767.0f)
        v = 32767.0f;
      else if (v < -32768.0f)
        v = -32768.0f;
      pcm[j] = static_cast<int16_t>(std::lrintf(v));
    }
    s->output->Write(pcm, m);
    s->input->Drain();
    i += m;
  }
}

constexpr int kMaxClauseChars = 128;

const char* SkipSpaces(const char* p) {
  while (*p == ' ' || *p == '\t') ++p;
  return p;
}

// Copy one clause from `text` into `out` (cap bytes). Return pointer to the
// remainder, or nullptr when done. Splits at punctuation or, if a clause would
// exceed kMaxClauseChars, at the last word boundary.
const char* PopClause(const char* text, char* out, size_t cap) {
  text = SkipSpaces(text);
  if (*text == '\0') {
    out[0] = '\0';
    return nullptr;
  }
  size_t n = 0;
  int last_space = -1;
  for (; text[n] != '\0' && n + 1 < cap; ++n) {
    const char c = text[n];
    out[n] = c;
    if (c == ' ') last_space = static_cast<int>(n);
    if (c == ',' || c == '.' || c == '!' || c == '?') {
      out[n + 1] = '\0';
      return SkipSpaces(text + n + 1);
    }
    if (n + 1 >= kMaxClauseChars - 1 && last_space > 0) {
      out[static_cast<size_t>(last_space)] = '\0';
      return SkipSpaces(text + last_space + 1);
    }
  }
  out[n] = '\0';
  return text[n] != '\0' ? SkipSpaces(text + n) : nullptr;
}

}  // namespace

void Speak(const char* text, AudioOutput& output, AudioInput& input,
           uint8_t* arena, std::size_t arena_size) {
  if (text == nullptr || *text == '\0') return;

  // ---- TTS phase (the neural engine uses the arena as working memory) ----
  {
    neural_tts::NeuralTts tts(g_neural_tts_pack, arena, arena_size);
    if (!tts.ok()) return;

    char clause[kMaxClauseChars];
    const char* p = text;
    int total = 0;
    do {
      p = PopClause(p, clause, sizeof(clause));
      if (clause[0] == '\0') break;
      const int n = tts.EstimateSamples(clause);
      if (n > 0) total += n;
    } while (p != nullptr);
    if (total <= 0) return;

    output.Begin(neural_tts::NeuralTts::kSampleRate, total);
    SpeakSink sink{&output, &input, g_tts_volume, /*raw_peak=*/0};
    p = text;
    do {
      p = PopClause(p, clause, sizeof(clause));
      if (clause[0] == '\0') break;
      if (tts.Synthesize(clause, SpeakEmit, &sink) < 0) break;
    } while (p != nullptr);
    output.End();
#ifdef SPELLING_AUDIO_DIAG
    printf("[tts] reply '%s' peak=%d/32767 (%.0f%% FS), volume=%.1fx\n", text,
           sink.raw_peak,
           100.0 * static_cast<double>(sink.raw_peak) / 32767.0,
           static_cast<double>(g_tts_volume));
    fflush(stdout);
#endif
  }  // synth done; arena free again.

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
                     int16_t* window, int window_samples) {
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

    // Optional bring-up aid: replay the raw capture before the TTS reply so mic
    // wiring/gain issues are audible. Off by default -- unvoiced letters (e.g.
    // /s/) peak-normalize to a hiss that is easy to mistake for bad TTS.
#ifdef SPELLING_PLAY_CAPTURE_ECHO
    PlayCapturedClip(window, window_samples, output, input);
#endif

    // Speak the letter's sound-alike word ("bee" for 'b'); digits already
    // arrive as words. See spelling_labels.h.
    Speak(SpokenForLabel(spelling::kClassLabels[pred]), output, input, arena,
          arena_size);
  }
}

}  // namespace spelling
