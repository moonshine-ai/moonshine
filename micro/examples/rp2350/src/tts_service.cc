#include "tts_service.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hardware/watchdog.h"
#include "neural_tts/neural_tts.h"
#include "pico/stdlib.h"

// Flash-resident neural TTS pack (generated/neural_tts_pack.S).
extern "C" const uint8_t g_neural_tts_pack[];

namespace spelling {

namespace {

BootReport g_boot_report = {};

}  // namespace

void SetBootReport(const BootReport& report) { g_boot_report = report; }

namespace {

// Longest command line accepted (bounds the G2P front end's transient heap
// use, the only heap the synth touches besides the kissfft plans; everything
// else lives in the arena).
constexpr int kMaxLineLen = 256;

// Read one newline-terminated line from USB CDC into `buf` (NUL-terminated),
// blocking until a non-empty line arrives. CR and LF both terminate; empty
// lines are skipped.
int ReadLine(char* buf, int maxlen) {
  int n = 0;
  for (;;) {
    const int c = getchar_timeout_us(1000000);  // 1 s slices, just loop
    // Idle heartbeat: main_tts.cc arms an 8 s watchdog for hang recovery
    // during synthesis; keep it fed while waiting for commands too.
    watchdog_update();
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

// PCM chunks stream straight to the CDC byte pipe as they render. Flush
// every chunk: newlib's stdout buffer otherwise holds the first ~4 KB
// while the first neural decode tile (~240 ms) runs, delaying first audio.
void EmitToUsb(void* user, const int16_t* samples, int n) {
  fwrite(samples, sizeof(int16_t), static_cast<size_t>(n), stdout);
  fflush(stdout);
  *static_cast<int*>(user) += n;
}

}  // namespace

void RunTtsService(uint8_t* arena, std::size_t arena_size) {
  neural_tts::NeuralTts tts(g_neural_tts_pack, arena, arena_size);
  if (!tts.ok()) {
    printf("[tts] err: neural TTS init failed (pack or arena)\n");
    fflush(stdout);
    for (;;) sleep_ms(1000);
  }

  static char line[kMaxLineLen];

  printf("\n[tts] ready. commands: SPEAK <text> | IPA <ipa>\n");
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

    // Post-mortem: report how the previous boot ended (the boot banner is
    // usually lost because it prints before the host opens the port).
    if (std::strcmp(line, "STATUS") == 0) {
      if (g_boot_report.watchdog_reboot) {
        printf("[tts] status: WATCHDOG reboot, ckpt=%lu ckpt2=%lu "
               "trace=%08lx fault_pc=%08lx\n",
               static_cast<unsigned long>(g_boot_report.ckpt),
               static_cast<unsigned long>(g_boot_report.ckpt2),
               static_cast<unsigned long>(g_boot_report.trace),
               static_cast<unsigned long>(g_boot_report.fault_pc));
      } else {
        printf("[tts] status: clean boot\n");
      }
      fflush(stdout);
      continue;
    }

    // Legacy knobs from the Klatt-synth protocol: the neural voice has a
    // fixed 16 kHz rate and a single speaker, so acknowledge and ignore.
    if (std::strcmp(line, "RATE") == 0 || std::strcmp(line, "SPEED") == 0 ||
        std::strcmp(line, "GENDER") == 0) {
      printf("[tts] note: %s is fixed for the neural voice (16000 Hz)\n",
             line);
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

    // Plan-only pass: exact sample count for the framing header (the host
    // reads exactly num_samples*2 binary bytes next).
    const int expected =
        ipa ? tts.EstimateSamplesIpa(arg) : tts.EstimateSamples(arg);
    if (expected <= 0) {
      printf("[tts] err: synth plan failed (%d)\n", expected);
      fflush(stdout);
      continue;
    }
    printf("AUDIO %d %d\n", neural_tts::NeuralTts::kSampleRate, expected);
    fflush(stdout);

    int total = 0;
    const int rc = ipa ? tts.SynthesizeIpa(arg, EmitToUsb, &total)
                       : tts.Synthesize(arg, EmitToUsb, &total);
    fflush(stdout);
    if (rc < 0) {
      // Header already promised `expected` samples; pad with silence so the
      // host's binary read still completes, then report the error.
      static int16_t zeros[256] = {0};
      for (int left = expected - total; left > 0; left -= 256)
        fwrite(zeros, sizeof(int16_t),
               static_cast<size_t>(left < 256 ? left : 256), stdout);
      total = expected;
    }
    fflush(stdout);
    printf("\nEND %d\n", total);
    // Latency breakdown of the synthesis call (microseconds; see
    // NeuralTts::Stats). first_pcm is engine-entry -> first PCM chunk.
    const neural_tts::NeuralTts::Stats& st = tts.stats();
    printf("TIME g2p=%lu runs=%lu plan=%lu stream=%lu alloc=%lu "
           "decode=%lu invoke=%lu post=%lu render=%lu first_pcm=%lu "
           "chunks=%d tiles=%d\n",
           static_cast<unsigned long>(st.g2p_us),
           static_cast<unsigned long>(st.runs_us),
           static_cast<unsigned long>(st.plan_us),
           static_cast<unsigned long>(st.stream_us),
           static_cast<unsigned long>(st.alloc_us),
           static_cast<unsigned long>(st.decode_us),
           static_cast<unsigned long>(st.invoke_us),
           static_cast<unsigned long>(st.post_us),
           static_cast<unsigned long>(st.render_us),
           static_cast<unsigned long>(st.first_pcm_us), st.chunks,
           st.tiles);
    fflush(stdout);
  }
}

}  // namespace spelling
