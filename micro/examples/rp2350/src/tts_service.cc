#include "tts_service.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "pico/stdlib.h"
#include "tts/tts.h"  // tts::VoiceParams / DefaultVoiceParams / StreamSynth

namespace spelling {

namespace {

// Longest command line accepted (bounds the front-end's transient heap use, the
// only heap the synth touches; the per-frame parameter tracks live in the
// arena).
constexpr int kMaxLineLen = 256;
// How many samples to pull from the streaming engine per Read() -- demonstrates
// arbitrary-length chunking and keeps the on-stack scratch tiny.
constexpr int kChunkSamples = 256;

// Read one newline-terminated line from USB CDC into `buf` (NUL-terminated),
// blocking until a non-empty line arrives. CR and LF both terminate; empty
// lines are skipped.
int ReadLine(char* buf, int maxlen) {
  int n = 0;
  for (;;) {
    const int c = getchar_timeout_us(1000000);  // 1 s slices, just loop
    if (c == PICO_ERROR_TIMEOUT) continue;
    if (c == '\r' || c == '\n') {
      if (n == 0) continue;  // ignore blank lines / stray CR before LF
      break;
    }
    if (n < maxlen - 1) buf[n++] = static_cast<char>(c);
  }
  buf[n] = '\0';
  return n;
}

// Whether the (trailing, ignoring spaces/quotes) punctuation makes this a
// question -> rising final boundary tone.
bool IsQuestion(const char* s) {
  for (int i = static_cast<int>(std::strlen(s)) - 1; i >= 0; --i) {
    const char c = s[i];
    if (c == ' ' || c == '"') continue;
    return c == '?';
  }
  return false;
}

}  // namespace

void RunTtsService(uint8_t* arena, std::size_t arena_size) {
  // The voice lives for the life of the loop (StreamSynth holds a reference).
  tts::VoiceParams voice = tts::DefaultVoiceParams();
  tts::StreamSynth synth(voice, arena, arena_size);

  tts::StreamOptions opts;
  opts.sample_rate = 22050.0f;

  static char line[kMaxLineLen];
  float chunk[kChunkSamples];
  int16_t pcm16[kChunkSamples];

  printf(
      "\n[tts] ready. commands: SPEAK <text> | IPA <ipa> | "
      "RATE <hz> | SPEED <x> | GENDER <0..1>\n");
  fflush(stdout);

  for (;;) {
    const int len = ReadLine(line, sizeof(line));
    if (len <= 0) continue;

    // Split "<CMD> <arg...>".
    char* arg = std::strchr(line, ' ');
    if (arg != nullptr) {
      *arg = '\0';
      ++arg;
    } else {
      arg = line + len;  // empty argument
    }

    if (std::strcmp(line, "RATE") == 0) {
      const float r = static_cast<float>(std::atof(arg));
      if (r >= 4000.0f) opts.sample_rate = r;
      printf("[tts] rate=%d\n", static_cast<int>(opts.sample_rate));
      fflush(stdout);
      continue;
    }
    if (std::strcmp(line, "SPEED") == 0) {
      const float s = static_cast<float>(std::atof(arg));
      if (s > 0.1f) opts.speed = s;
      printf("[tts] speed=%.3f\n", static_cast<double>(opts.speed));
      fflush(stdout);
      continue;
    }
    if (std::strcmp(line, "GENDER") == 0) {
      float g = static_cast<float>(std::atof(arg));
      if (g < 0.0f) g = 0.0f;
      // Same mapping as the desktop --gender flag.
      voice.formant_scale = 1.0f + 0.18f * g;
      voice.f0_scale = 1.0f + 0.90f * g;
      printf("[tts] gender=%.2f\n", static_cast<double>(g));
      fflush(stdout);
      continue;
    }

    const bool ipa = (std::strcmp(line, "IPA") == 0);
    const bool speak = (std::strcmp(line, "SPEAK") == 0);
    if (!ipa && !speak) {
      printf("[tts] err: unknown command '%s'\n", line);
      fflush(stdout);
      continue;
    }

    opts.question = IsQuestion(arg);
    const int rc = ipa ? synth.BeginIpa(arg, opts) : synth.BeginText(arg, opts);
    if (rc != tts::kStreamOk) {
      printf("[tts] err: begin failed (%d)\n", rc);
      fflush(stdout);
      continue;
    }

    // Framing header: the host reads exactly num_samples*2 binary bytes next.
    printf("AUDIO %d %d\n", synth.sample_rate(), synth.total_samples());
    fflush(stdout);

    // Stream PCM in fixed chunks: pull floats, clamp+convert to int16, write
    // the raw bytes. The whole utterance is never buffered on-device.
    int total = 0;
    for (int n; (n = synth.Read(chunk, kChunkSamples)) > 0;) {
      for (int i = 0; i < n; ++i) {
        float s = chunk[i];
        if (s > 1.0f)
          s = 1.0f;
        else if (s < -1.0f)
          s = -1.0f;
        pcm16[i] = static_cast<int16_t>(std::lrintf(s * 32767.0f));
      }
      fwrite(pcm16, sizeof(int16_t), static_cast<size_t>(n), stdout);
      total += n;
    }
    fflush(stdout);
    printf("\nEND %d\n", total);
    fflush(stdout);
  }
}

}  // namespace spelling
