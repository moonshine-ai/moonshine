// Standalone I2S microphone bring-up test (the `moonshine_micro_i2s_mic_test`
// target). It does NOT touch the VAD/STT/TTS pipeline -- it only proves that an
// Adafruit SPH0645 I2S MEMS mic (https://www.adafruit.com/product/3421) is wired
// up and producing audio.
//
// Wiring (matches the arduino-pico I2SInput.ino example the user followed):
//
//   mic DOUT -> GPIO0   (data into the RP2350)
//   mic BCLK <- GPIO1   (bit clock, driven by the RP2350)
//   mic LRCL <- GPIO2   (word select; must be BCLK + 1 for the PIO sideset)
//   mic SEL  -> GND     (puts the mono element on the LEFT channel)
//   mic GND  <-> GND
//   mic 3V   <-> 3V3 (OUT)
//
// Each cycle: record 5 s at 16 kHz, print per-channel stats, then (if USB is
// connected) stream the clip to the host for 5 s of playback. The device waits
// for CLIP_ACK from usb_audio_bridge.py before recording again so the host does
// not fall behind. Plain monitor.sh still works for the text stats only.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "hardware/clocks.h"
#include "hardware/pio.h"
#include "hardware/vreg.h"
#include "i2s_mic.pio.h"
#include "i2s_mic_process.h"
#include "pico/stdlib.h"
#include "usb_audio_io.h"

namespace {

constexpr uint kDataPin = 0;
constexpr uint kClockBase = 1;
constexpr uint kBitsPerSample = 32;
constexpr uint kSampleRate = 16000;
constexpr int kRecordSeconds = 5;
constexpr int kNumSamples = kSampleRate * kRecordSeconds;

#ifdef PICO_DEFAULT_LED_PIN
constexpr uint kLedPin = PICO_DEFAULT_LED_PIN;
#else
constexpr uint kLedPin = 25u;
#endif

int16_t g_recording[kNumSamples];

struct ChannelStats {
  int64_t sum = 0;
  int64_t sum_sq = 0;
  int32_t min_v = INT32_MAX;
  int32_t max_v = INT32_MIN;
  uint32_t or_bits = 0;
  uint32_t and_bits = 0xFFFFFFFFu;
  int count = 0;

  void Add(int32_t s, uint32_t raw) {
    sum += s;
    sum_sq += static_cast<int64_t>(s) * s;
    if (s < min_v) min_v = s;
    if (s > max_v) max_v = s;
    or_bits |= raw;
    and_bits &= raw;
    ++count;
  }
};

void DrainUsbInput() {
  while (getchar_timeout_us(0) != PICO_ERROR_TIMEOUT) {
  }
}

void DrawBar(double level, double full_scale) {
  constexpr int kWidth = 32;
  int filled = static_cast<int>((level / full_scale) * kWidth);
  if (filled < 0) filled = 0;
  if (filled > kWidth) filled = kWidth;
  putchar('[');
  for (int i = 0; i < kWidth; ++i) putchar(i < filled ? '#' : ' ');
  putchar(']');
}

double ChannelRms(const ChannelStats& st) {
  if (st.count == 0) return 0.0;
  const double mean = static_cast<double>(st.sum) / st.count;
  const double mean_sq = static_cast<double>(st.sum_sq) / st.count;
  return std::sqrt(std::max(0.0, mean_sq - mean * mean));
}

void ReportChannel(const char* name, const ChannelStats& st, bool active) {
  if (st.count == 0) return;
  const double mean = static_cast<double>(st.sum) / st.count;
  const double rms = ChannelRms(st);
  const int32_t pp = st.max_v - st.min_v;

  printf("  %s dc=%-8.0f pp=%-8ld rms=%-8.0f ", name, mean,
         static_cast<long>(pp), rms);
  DrawBar(rms, 16384.0);

  if (!active) {
    printf("  (other channel; SEL->GND silences this one -- normal)");
  } else if (st.and_bits == 0xFFFFFFFFu) {
    printf("  <- stuck HIGH: check DOUT->GPIO0 / GND");
  } else if (st.or_bits == 0u) {
    printf("  <- stuck LOW: check DOUT->GPIO0, 3V, BCLK->GPIO1");
  }
  putchar('\n');
}

void RecordWindow(PIO pio, uint sm, ChannelStats* a, ChannelStats* b,
                  spelling::I2sDcBlocker* dc_blocker) {
  *a = ChannelStats{};
  *b = ChannelStats{};
  dc_blocker->Reset();
  DrainUsbInput();

  for (int i = 0; i < kNumSamples; ++i) {
    if ((i & 0x1FF) == 0) {
      DrainUsbInput();
    }
    const uint32_t raw_a = pio_sm_get_blocking(pio, sm);
    const uint32_t raw_b = pio_sm_get_blocking(pio, sm);
    g_recording[i] = dc_blocker->Process(spelling::I2sRawToInt16(raw_a));
    a->Add(spelling::I2sRawToInt32(raw_a), raw_a);
    b->Add(spelling::I2sRawToInt32(raw_b), raw_b);
  }
}

void StreamToHost(const int16_t* samples, int n) {
  spelling::UsbAudioOutput usb;
  usb.Begin(kSampleRate, n, "CLIP");
  constexpr int kChunk = 256;
  for (int i = 0; i < n; i += kChunk) {
    const int chunk = std::min(kChunk, n - i);
    usb.Write(samples + i, chunk);
  }
  usb.End();
}

// Wait for usb_audio_bridge.py to finish playing the clip and send CLIP_ACK.
bool WaitForClipAck(int timeout_ms) {
  char line[32];
  int len = 0;
  const absolute_time_t deadline = make_timeout_time_ms(timeout_ms);

  while (!time_reached(deadline)) {
    const int c = getchar_timeout_us(50000);
    if (c == PICO_ERROR_TIMEOUT) continue;
    if (c == '\r') continue;
    if (c == '\n') {
      line[len] = '\0';
      len = 0;
      if (std::strcmp(line, "CLIP_ACK") == 0) return true;
      continue;
    }
    if (len + 1 < static_cast<int>(sizeof(line))) {
      line[len++] = static_cast<char>(c);
    }
  }
  return false;
}

}  // namespace

int main() {
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

  printf("\n=== I2S mic test (SPH0645) ===\n");
  printf("Clock:   %lu MHz\n",
         static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000000u));
  printf("Pins:    DIN=GPIO%u  BCLK=GPIO%u  LRCL=GPIO%u\n", kDataPin, kClockBase,
         kClockBase + 1);
  printf("Format:  %u-bit slots, %u Hz, master mode\n", kBitsPerSample,
         kSampleRate);
  printf("Cycle:   record %d s -> stats -> host playback %d s -> repeat\n",
         kRecordSeconds, kRecordSeconds);
  printf("Run usb_audio_bridge.py to hear each clip on the host speakers.\n\n");

  PIO pio = pio0;
  uint sm = 0;
  const uint offset = pio_add_program(pio, &i2s_mic_in_program);
  i2s_mic_in_program_init(pio, sm, offset, kDataPin, kClockBase, kBitsPerSample,
                          kSampleRate);

  spelling::I2sDcBlocker dc_blocker(kSampleRate);

  while (true) {
    printf("recording %d s...\n", kRecordSeconds);
    fflush(stdout);

    ChannelStats a, b;
    RecordWindow(pio, sm, &a, &b, &dc_blocker);

    const double a_rms = ChannelRms(a);
    const double b_rms = ChannelRms(b);
    const bool a_active = a_rms >= b_rms;
    gpio_put(kLedPin, std::max(a_rms, b_rms) > 4096.0 ? 1 : 0);

    if (a.or_bits == 0u && b.or_bits == 0u) {
      printf("capture: NO DATA on either channel -- check wiring "
             "(DOUT->GPIO0, BCLK->GPIO1, LRCL->GPIO2, 3V, GND)\n");
    } else {
      printf("capture:\n");
    }
    ReportChannel("chA", a, a_active);
    ReportChannel("chB", b, !a_active);
    fflush(stdout);

    if (stdio_usb_connected()) {
      printf("streaming %d s clip to host...\n", kRecordSeconds);
      fflush(stdout);
      spelling::I2sRemoveBufferDc(g_recording, kNumSamples);
      StreamToHost(g_recording, kNumSamples);
      printf("waiting for host playback (CLIP_ACK)...\n");
      fflush(stdout);
      if (WaitForClipAck(kRecordSeconds * 1000 + 5000)) {
        printf("host playback done.\n");
      } else {
        printf("CLIP_ACK timeout; continuing anyway.\n");
      }
      fflush(stdout);
    }

    sleep_ms(200);
  }
  return 0;
}
