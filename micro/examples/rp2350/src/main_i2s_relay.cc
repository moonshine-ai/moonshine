// Dumb PCM -> I2S relay (the `moonshine_micro_i2s_relay` target).
//
// This firmware does NO synthesis, VAD or STT. It is the device half of the
// hardware-in-the-loop TTS eval harness (scripts/hw_tts_eval/): the laptop
// synthesizes speech, streams the raw PCM here over USB CDC, and this app
// plays it out the MAX98357A I2S amp so an external mic can record the real
// acoustic path. Keeping the device dumb means the loop measures the exact
// same I2sAudioOutput path the live apps use, with nothing else in the way.
//
// Wiring (identical to moonshine_micro_i2s_audio_test; see src/i2s_out.pio):
//   Vin  -> Pico VSYS (pin 39)   GND -> GND
//   DIN  -> GP10   BCLK -> GP11   LRC -> GP12   (MAX98357A / Adafruit 3006)
//   GAIN -> open (9 dB)   SD -> open (enabled, (L+R)/2 mono)
//
// Protocol (line commands over CDC; binary PCM is framed, not line-based):
//   PING              -> replies "PONG"
//   PLAY <rate> <n>   -> then read `n` int16 LE samples, streamed in hops of
//                        up to kHop samples, each hop prefixed by the 0xA5 0x5A
//                        sync (mirrors usb_audio_io.cc). Samples are written to
//                        I2sAudioOutput at <rate> Hz. When done, replies
//                        "PLAYED <m>" with the number of samples actually
//                        played (m == n on a clean stream).
//
// PICO_STDIO_DEFAULT_CRLF=0 (see CMakeLists.txt) keeps 0x0A/0x0D PCM bytes from
// being mangled on the way in, and keeps our text replies from getting CR
// injected on the way out.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "hardware/watchdog.h"
#include "i2s_audio_out.h"
#include "pico/stdlib.h"

namespace {

constexpr unsigned kDataPin = 10;    // MAX98357A DIN
constexpr unsigned kClockBase = 11;  // BCLK = GP11, LRC = GP12
constexpr int kDefaultRate = 16000;

// Whole-clip receive buffer. The host's playout (lead + chirp + gap + clip +
// tail) is well under 10 s, so buffer the entire PCM stream in RAM and only
// then feed I2S -- see PlayStream for why streaming USB straight to the FIFO
// glitches. 160000 samples @ 16 kHz = 10 s = 320 KiB (fits the RP2350's 520 KiB
// SRAM with room to spare, since this app has no models/arena).
constexpr int kMaxSamples = 160000;
alignas(2) int16_t g_pcm[kMaxSamples];

// Host->device hop framing: each hop is preceded by a 2-byte sync so the device
// can re-align if a byte is ever dropped on the CDC pipe (mirrors
// usb_audio_io.cc's UsbAudioInput::ReadHop). 512 samples/hop matches the plan.
constexpr int kSync0 = 0xA5;
constexpr int kSync1 = 0x5A;
constexpr int kHop = 512;

constexpr int kMaxLineLen = 64;

#ifdef PICO_DEFAULT_LED_PIN
constexpr unsigned kLedPin = PICO_DEFAULT_LED_PIN;
#else
constexpr unsigned kLedPin = 25u;  // Pico 2 W LED is on the CYW43; harmless
#endif

// Read one newline-terminated line into `buf` (NUL-terminated), blocking until
// a non-empty line arrives. Feeds the watchdog each idle slice. CR and LF both
// terminate; blank lines are skipped.
int ReadLine(char* buf, int maxlen) {
  int n = 0;
  for (;;) {
    const int c = getchar_timeout_us(1000000);  // 1 s slices
    watchdog_update();
    if (c == PICO_ERROR_TIMEOUT) continue;
    if (c == '\r' || c == '\n') {
      if (n == 0) continue;
      break;
    }
    if (n < maxlen - 1) buf[n++] = static_cast<char>(c);
  }
  buf[n] = '\0';
  return n;
}

// Read one byte, waiting up to ~timeout_ms. Returns -1 on timeout.
int ReadByteTimed(int timeout_ms) {
  const int c = getchar_timeout_us(timeout_ms * 1000);
  return (c == PICO_ERROR_TIMEOUT) ? -1 : (c & 0xFF);
}

// Scan for the 0xA5 0x5A sync, then read up to `n` int16 LE samples into `out`.
// Returns the number of samples read; a short return means the host stalled
// mid-hop (each sample byte read times out so a dropped frame can't wedge the
// device -- see the deadlock note in usb_audio_io.cc). Returns -1 if no sync
// arrives within ~2 s (host aborted the stream).
int ReadHop(int16_t* out, int n) {
  int idle_us = 0;
  int state = 0;  // 0: seeking 0xA5, 1: got 0xA5, seeking 0x5A
  for (;;) {
    const int c = getchar_timeout_us(100000);  // 100 ms slices
    watchdog_update();
    if (c == PICO_ERROR_TIMEOUT) {
      idle_us += 100000;
      if (idle_us >= 2000000) return -1;  // ~2 s with no bytes -> abort
      continue;
    }
    idle_us = 0;
    const int b = c & 0xFF;
    if (state == 0) {
      if (b == kSync0) state = 1;
    } else {
      if (b == kSync1) break;
      state = (b == kSync0) ? 1 : 0;
    }
  }
  int i = 0;
  for (; i < n; ++i) {
    const int lo = ReadByteTimed(300);
    if (lo < 0) break;
    const int hi = ReadByteTimed(300);
    if (hi < 0) break;
    out[i] = static_cast<int16_t>((hi << 8) | lo);
  }
  return i;
}

// Receive `total` samples from the host into RAM, then play them to the I2S amp
// at `rate` Hz. Returns the number of samples played.
//
// Why buffer the whole clip instead of streaming hop->FIFO directly: the PIO
// I2S TX FIFO is only a few frames deep, and reading a 512-sample hop over USB
// CDC (one getchar per byte) takes far longer than those few frames play out.
// A read-then-write loop therefore underruns the FIFO on every hop boundary --
// the SM stalls mid-stream and the amp clicks, which sounds like continuous
// broadband static. Receiving the whole clip first (no real-time constraint;
// USB flow-control backpressures the host) then streaming RAM->FIFO keeps the
// FIFO fed faster than real time, so playback is glitch-free.
int PlayStream(spelling::I2sAudioOutput& out, int rate, int total) {
  if (total > kMaxSamples) total = kMaxSamples;
  int recvd = 0;
  while (recvd < total) {
    const int want = (total - recvd) < kHop ? (total - recvd) : kHop;
    const int got = ReadHop(g_pcm + recvd, want);
    if (got <= 0) break;  // host stalled/aborted
    recvd += got;
    if (got < want) break;  // short hop -> stream broke mid-frame
  }

  out.Begin(rate, recvd);
  gpio_put(kLedPin, 1);
  // Write in chunks so the 8 s watchdog stays fed across a multi-second clip.
  constexpr int kChunk = 1024;
  for (int off = 0; off < recvd; off += kChunk) {
    const int n = (recvd - off) < kChunk ? (recvd - off) : kChunk;
    out.Write(g_pcm + off, n);
    watchdog_update();
  }
  gpio_put(kLedPin, 0);
  out.End();
  return recvd;
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

}  // namespace

int main() {
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

  if (watchdog_caused_reboot()) {
    printf("\n[boot] REBOOT POST-MORTEM: fault_pc=%08lx\n",
           static_cast<unsigned long>(watchdog_hw->scratch[3]));
    fflush(stdout);
  }
  watchdog_hw->scratch[3] = 0;
  watchdog_enable(8000, true);  // pause_on_debug

  spelling::I2sAudioOutput out(kDataPin, kClockBase, kDefaultRate);

  printf("\n[boot] i2s relay ready, clk_sys=%lu MHz\n",
         static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000000u));
  printf("[relay] I2S DIN=GP%u BCLK=GP%u LRC=GP%u\n", kDataPin, kClockBase,
         kClockBase + 1);
  printf("[relay] commands: PING | PLAY <rate> <n>\n");
  fflush(stdout);

  static char line[kMaxLineLen];
  for (;;) {
    const int len = ReadLine(line, sizeof(line));
    if (len <= 0) continue;

    // Split "<CMD> <arg...>".
    char* arg = std::strchr(line, ' ');
    if (arg != nullptr) {
      *arg = '\0';
      ++arg;
    } else {
      arg = line + len;
    }

    if (std::strcmp(line, "PING") == 0) {
      printf("PONG\n");
      fflush(stdout);
      continue;
    }

    if (std::strcmp(line, "PLAY") == 0) {
      // arg = "<rate> <n>"
      char* endp = nullptr;
      const long rate = std::strtol(arg, &endp, 10);
      const long n = (endp != nullptr) ? std::strtol(endp, nullptr, 10) : 0;
      if (rate <= 0 || n <= 0) {
        printf("ERR bad PLAY args (rate=%ld n=%ld)\n", rate, n);
        fflush(stdout);
        continue;
      }
      // Ack the header so the host knows we're ready to receive the PCM hops.
      printf("READY %ld %ld\n", rate, n);
      fflush(stdout);
      const int played =
          PlayStream(out, static_cast<int>(rate), static_cast<int>(n));
      printf("PLAYED %d\n", played);
      fflush(stdout);
      continue;
    }

    printf("ERR unknown command '%s'\n", line);
    fflush(stdout);
  }
}
