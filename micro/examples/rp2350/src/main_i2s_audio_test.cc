// Standalone speaker / audio-output bring-up test (the
// `moonshine_micro_i2s_audio_test` target). It proves the I2S amp + speaker are
// wired up and making sound, driven by the same I2sAudioOutput backend the live
// apps use. Besides tones and recorded clips it also runs the neural TTS engine
// (the same one the live apps speak with) so the whole render path is exercised
// -- it does NOT touch the VAD/STT capture pipeline.
//
// Designed for the Adafruit MAX98357A I2S class-D amp (#3006,
// https://www.adafruit.com/product/3006), which has its own I2S DAC -- so
// unlike the old PWM speaker there is no carrier and no RC reconstruction
// filter. Connect your own 4-8 ohm speaker to the amp's screw terminals.
//
// Wiring (see src/i2s_out.pio; LRC = BCLK + 1 for the sideset pair):
//
//   Vin  -> Pico VSYS   (physical pin 39)   ; 2.5-5.5 V; use VSYS not VBUS
//   GND  -> Pico GND    (physical pin 38)
//   DIN  -> GP10        (physical pin 14)   ; I2S data
//   BCLK -> GP11        (physical pin 15)   ; bit clock
//   LRC  -> GP12        (physical pin 16)   ; word select (LRCLK)
//   GAIN -> (leave open -> 9 dB)   SD -> (leave open -> enabled, (L+R)/2 mono)
//
// GP10/11/12 are chosen so they don't collide with the I2S mic test (GP0/1/2).
// What you should hear: a rising A-major arpeggio, then a tone sweep, then 1 s
// recordings of "a", "b", "c", then the neural TTS voice saying the letters
// "a"/"b"/"c" and the words "hello"/"wifi", looping.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "hardware/watchdog.h"
#include "i2s_audio_out.h"       // I2sAudioOutput + PeakNormalizeGain
#include "neural_tts/neural_tts.h"
#include "pico/stdlib.h"
#include "speaker_test_clips.h"
#include "spelling_labels.h"     // SpokenForLabel (letter -> sound-alike word)

// Flash-resident neural TTS pack (generated/neural_tts_pack.S).
extern "C" const uint8_t g_neural_tts_pack[];

// Synthesis post-mortem (same scheme as main_tts.cc): the neural-TTS pipeline
// calls these checkpoint hooks (weak no-ops in neural-tts/src/hooks.cc); we
// override them to feed an 8 s hardware watchdog and stash progress in the
// watchdog scratch registers. If synth locks up (or hard-faults) the board
// auto-reboots and the boot banner prints the last checkpoint reached, instead
// of silently wedging USB and forcing a manual BOOTSEL power cycle.
// scratch[4..7] are reserved by watchdog_reboot()/the bootrom, so we use [0..3].
extern "C" void tts_checkpoint(uint32_t v) {
  watchdog_hw->scratch[0] = v;
  watchdog_update();
}
extern "C" void tts_checkpoint2(uint32_t v) { watchdog_hw->scratch[1] = v; }
extern "C" void tts_trace(uint32_t tag, uint32_t val) {
  watchdog_hw->scratch[2] = (tag << 24) | (val & 0xFFFFFFu);
}
extern "C" void HardFaultC(uint32_t* frame) {
  watchdog_hw->scratch[3] = frame[6];  // stacked (faulting) PC
  watchdog_reboot(0, 0, 1);
  for (;;) {
  }
}
extern "C" __attribute__((naked)) void isr_hardfault(void) {
  __asm volatile(
      "tst lr, #4\n"
      "ite eq\n"
      "mrseq r0, msp\n"
      "mrsne r0, psp\n"
      "b HardFaultC\n");
}

namespace {

constexpr uint kDataPin = 10;    // MAX98357A DIN
constexpr uint kClockBase = 11;  // BCLK = GP11, LRC = GP12
constexpr int kSampleRate = 16000;
constexpr int kToneChunk = 256;

// Working memory for the synthesizer (TFLM decoder arena + unit selection +
// control track). 360 KiB matches the known-good budget the echo_hardware app
// lends the same voice pack; it is only touched during Synthesize() so it can
// share SRAM with the tone/clip buffers.
constexpr std::size_t kTtsArenaBytes = 360u * 1024u;
alignas(16) uint8_t g_tts_arena[kTtsArenaBytes];

#ifdef PICO_DEFAULT_LED_PIN
constexpr uint kLedPin = PICO_DEFAULT_LED_PIN;
#else
constexpr uint kLedPin = 25u;  // Pico 2 W LED is on the CYW43; harmless fallback
#endif

// int16 sine lookup centred on zero (I2S is signed PCM, no midpoint offset).
int16_t g_sine[256];

void BuildSineTable() {
  const double amp = 32767.0 * 0.85;  // headroom so peaks don't clip
  for (int i = 0; i < 256; ++i) {
    const double theta = (2.0 * M_PI * i) / 256.0;
    g_sine[i] = static_cast<int16_t>(std::lround(amp * std::sin(theta)));
  }
}

// Play `freq` Hz for `ms` ms with a short attack/release to avoid clicks. Uses
// the same I2sAudioOutput sink as the live apps (blocking FIFO writes pace it).
void PlayTone(spelling::I2sAudioOutput& out, double freq, int ms) {
  const uint32_t total = static_cast<uint32_t>((uint64_t)kSampleRate * ms / 1000);
  const uint32_t ramp = kSampleRate / 200;  // ~5 ms fade in/out
  const uint32_t phase_inc =
      static_cast<uint32_t>((freq * 4294967296.0) / kSampleRate);  // 32-bit

  out.Begin(kSampleRate, static_cast<int>(total));
  uint32_t phase = 0;
  int16_t chunk[kToneChunk];
  for (uint32_t n = 0; n < total;) {
    int c = 0;
    for (; c < kToneChunk && n < total; ++c, ++n) {
      double env = 1.0;
      if (n < ramp) {
        env = static_cast<double>(n) / ramp;
      } else if (n > total - ramp) {
        env = static_cast<double>(total - n) / ramp;
      }
      chunk[c] = static_cast<int16_t>(g_sine[phase >> 24] * env);
      phase += phase_inc;
    }
    out.Write(chunk, c);
    watchdog_update();
  }
  out.End();
}

// Linear frequency sweep from `f0` to `f1` over `ms`, click-free at the ends.
void PlaySweep(spelling::I2sAudioOutput& out, double f0, double f1, int ms) {
  const uint32_t total = static_cast<uint32_t>((uint64_t)kSampleRate * ms / 1000);
  const uint32_t ramp = kSampleRate / 200;

  out.Begin(kSampleRate, static_cast<int>(total));
  uint32_t phase = 0;
  int16_t chunk[kToneChunk];
  for (uint32_t n = 0; n < total;) {
    int c = 0;
    for (; c < kToneChunk && n < total; ++c, ++n) {
      const double t = static_cast<double>(n) / total;
      const double freq = f0 + (f1 - f0) * t;
      const uint32_t phase_inc =
          static_cast<uint32_t>((freq * 4294967296.0) / kSampleRate);
      double env = 1.0;
      if (n < ramp) {
        env = static_cast<double>(n) / ramp;
      } else if (n > total - ramp) {
        env = static_cast<double>(total - n) / ramp;
      }
      chunk[c] = static_cast<int16_t>(g_sine[phase >> 24] * env);
      phase += phase_inc;
    }
    out.Write(chunk, c);
    watchdog_update();
  }
  out.End();
}

// Streaming sink for the neural TTS: each rendered PCM chunk is pushed straight
// to the I2S amp as it arrives (blocking FIFO writes pace it in real time), so
// there's no need to buffer the whole utterance -- matches the live apps.
struct TtsSink {
  spelling::I2sAudioOutput* out;
  int written;
};

void TtsEmit(void* user, const int16_t* samples, int n) {
  auto* sink = static_cast<TtsSink*>(user);
  // Copy into a small local buffer and write in <=256-sample chunks, mirroring
  // the live apps' SpeakEmit (audio_service.cc) rather than handing the synth's
  // arena buffer straight to the blocking I2S writer.
  int16_t pcm[256];
  for (int i = 0; i < n;) {
    const int m = (n - i) < 256 ? (n - i) : 256;
    for (int j = 0; j < m; ++j) pcm[j] = samples[i + j];
    sink->out->Write(pcm, m);
    watchdog_update();
    i += m;
  }
  sink->written += n;
}

// Synthesize `text` and play it. `label` is just the human-readable name for
// the log line (for letters it differs from the spoken sound-alike text).
void SpeakText(spelling::I2sAudioOutput& out, neural_tts::NeuralTts& tts,
               const char* label, const char* text) {
  // Plan-only pass first: gives the sink an accurate length for its banner.
  const int expected = tts.EstimateSamples(text);
  if (expected <= 0) {
    printf("saying '%s' (\"%s\"): synth plan failed (%d)\n", label, text,
           expected);
    fflush(stdout);
    return;
  }
  printf("saying: '%s' -> \"%s\" (%d samples @ %d Hz)\n", label, text, expected,
         neural_tts::NeuralTts::kSampleRate);
  fflush(stdout);

  TtsSink sink = {&out, 0};
  out.Begin(neural_tts::NeuralTts::kSampleRate, expected, "TTS");
  const int rc = tts.Synthesize(text, TtsEmit, &sink);
  out.End();
  if (rc < 0) {
    printf("  synth error %d\n", rc);
    fflush(stdout);
    return;
  }
  // Timing (the first utterance after boot is slow: cold XIP cache on the pack).
  const neural_tts::NeuralTts::Stats& st = tts.stats();
  printf("  done: %d samples, first_pcm=%lu ms, decode=%lu ms, render=%lu ms\n",
         sink.written, static_cast<unsigned long>(st.first_pcm_us / 1000u),
         static_cast<unsigned long>(st.decode_us / 1000u),
         static_cast<unsigned long>(st.render_us / 1000u));
  fflush(stdout);
}

}  // namespace

int main() {
  // NB: the deep neural-TTS decode call chain overflows core 0's 4 KiB scratch
  // stack; because Invoke() runs the dual-core GEMM (core 1 live on the adjacent
  // scratch bank), the overflow would corrupt core 1's stack. This target uses
  // the memmap_dualcore_stack.ld linker script (see CMakeLists.txt) so the
  // overflow spills into unused RAM instead. See petewarden.com/2024/01/16.

  // Match the rest of the firmware: 250 MHz (bump core voltage first).
  vreg_set_voltage(VREG_VOLTAGE_1_20);
  sleep_ms(10);
  set_sys_clock_khz(250000, true);

  gpio_init(kLedPin);
  gpio_set_dir(kLedPin, GPIO_OUT);
  for (int i = 0; i < 5; ++i) {
    gpio_put(kLedPin, 1);
    sleep_ms(80);
    gpio_put(kLedPin, 0);
    sleep_ms(80);
  }

  stdio_init_all();
  for (int i = 0; i < 1500 && !stdio_usb_connected(); ++i) {
    gpio_put(kLedPin, (i & 0x10) ? 1 : 0);
    sleep_ms(20);
  }
  gpio_put(kLedPin, 0);
  sleep_ms(200);

  // Post-mortem: if a previous run locked up or hard-faulted mid-synthesis, the
  // watchdog/fault handler rebooted us and left the last checkpoint in scratch.
  // Report it once so a regression is visible instead of silently wedging USB.
  if (watchdog_caused_reboot()) {
    printf("\n[boot] REBOOT POST-MORTEM: ckpt=%lu ckpt2=%lu trace=%08lx "
           "fault_pc=%08lx\n",
           static_cast<unsigned long>(watchdog_hw->scratch[0]),
           static_cast<unsigned long>(watchdog_hw->scratch[1]),
           static_cast<unsigned long>(watchdog_hw->scratch[2]),
           static_cast<unsigned long>(watchdog_hw->scratch[3]));
    fflush(stdout);
  }
  watchdog_hw->scratch[0] = 0;
  watchdog_hw->scratch[1] = 0;
  watchdog_hw->scratch[2] = 0;
  watchdog_hw->scratch[3] = 0;
  watchdog_enable(8000, true);  // pause_on_debug

  BuildSineTable();

  spelling::I2sAudioOutput out(kDataPin, kClockBase, kSampleRate);

  printf("\n=== I2S audio test (MAX98357A) ===\n");
  printf("Clock:   %lu MHz\n",
         static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000000u));
  printf("I2S:     DIN=GP%u BCLK=GP%u LRC=GP%u, %d Hz\n", kDataPin, kClockBase,
         kClockBase + 1, kSampleRate);
  printf("Wire:    Vin->VSYS(pin39)  GND->GND  DIN->GP%u  BCLK->GP%u  LRC->GP%u\n",
         kDataPin, kClockBase, kClockBase + 1);

  // Bring up the neural TTS engine once (it only uses the arena inside
  // Synthesize, so it can coexist with the tone/clip buffers). If the pack or
  // arena is bad we still run the tone/clip portion of the test.
  neural_tts::NeuralTts tts(g_neural_tts_pack, g_tts_arena, kTtsArenaBytes);
  const bool tts_ok = tts.ok();
  if (tts_ok) {
    printf("TTS:     neural voice ready (arena %u KiB)\n",
           static_cast<unsigned>(kTtsArenaBytes / 1024u));
  } else {
    printf("TTS:     init FAILED (pack/arena) -- skipping spoken section\n");
  }
  printf("You should hear: arpeggio, sweep, recorded a/b/c clips%s.\n\n",
         tts_ok ? ", then spoken a/b/c + hello/wifi" : "");

  // A4, C#5, E5, A5 -- an A-major arpeggio, then an octave sweep.
  const double arpeggio[] = {440.0, 554.37, 659.25, 880.0};

  while (true) {
    watchdog_update();
    printf("playing: A-major arpeggio\n");
    fflush(stdout);
    for (double f : arpeggio) {
      gpio_put(kLedPin, 1);
      PlayTone(out, f, 250);
      gpio_put(kLedPin, 0);
      sleep_ms(60);
    }
    sleep_ms(250);

    printf("playing: 200 Hz -> 2 kHz sweep\n");
    fflush(stdout);
    gpio_put(kLedPin, 1);
    PlaySweep(out, 200.0, 2000.0, 1200);
    gpio_put(kLedPin, 0);
    sleep_ms(300);

    for (int i = 0; i < speaker_test::kClipCount; ++i) {
      const speaker_test::Clip& clip = speaker_test::kClips[i];
      const float gain =
          spelling::PeakNormalizeGain(clip.samples, clip.num_samples);
      printf("playing: recorded '%s' (%d samples @ %d Hz, gain %.1fx)\n",
             clip.label, clip.num_samples, clip.sample_rate,
             static_cast<double>(gain));
      fflush(stdout);
      gpio_put(kLedPin, 1);
      out.SetGain(gain);
      out.Begin(clip.sample_rate, clip.num_samples);
      for (int off = 0; off < clip.num_samples; off += kToneChunk) {
        const int n = (clip.num_samples - off) < kToneChunk
                          ? (clip.num_samples - off)
                          : kToneChunk;
        out.Write(clip.samples + off, n);
        watchdog_update();
      }
      out.End();
      out.SetGain(1.0f);
      gpio_put(kLedPin, 0);
      sleep_ms(400);
    }

    // Neural TTS: say the letters (via their spoken sound-alikes, e.g. 'a'->
    // "hay", 'b'->"bee") then a couple of words. The pack's per-unit loudness
    // normalization keeps these at a consistent, clip-free level (gain 1.0).
    if (tts_ok) {
      printf("playing: neural TTS letters + words\n");
      fflush(stdout);
      static const char* const kLetters[] = {"a", "b", "c"};
      for (const char* letter : kLetters) {
        gpio_put(kLedPin, 1);
        SpeakText(out, tts, letter, spelling::SpokenForLabel(letter));
        gpio_put(kLedPin, 0);
        sleep_ms(350);
      }
      static const char* const kWords[] = {"hello", "wifi"};
      for (const char* word : kWords) {
        gpio_put(kLedPin, 1);
        SpeakText(out, tts, word, word);
        gpio_put(kLedPin, 0);
        sleep_ms(350);
      }
    }

    sleep_ms(600);
  }
  return 0;
}
