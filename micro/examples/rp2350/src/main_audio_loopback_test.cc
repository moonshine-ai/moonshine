// Standalone mic -> speaker loopback test (the `moonshine_micro_audio_loopback_test`
// target). Records five seconds from the I2S mic, plays it on the I2S amp, then
// repeats forever.
//
// Wiring (same as echo_hardware):
//   Mic (SPH0645):    DOUT->GP0  BCLK->GP1   LRCL->GP2   SEL->GND
//   Amp (MAX98357A):  DIN->GP10  BCLK->GP11  LRC->GP12   Vin->VSYS  GND->GND

#include <cstdint>
#include <cstdio>

#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "i2s_audio_io.h"
#include "i2s_audio_out.h"
#include "pico/stdlib.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr int kRecordSeconds = 5;
constexpr int kNumSamples = kSampleRate * kRecordSeconds;  // 80000
constexpr int kHop = 512;

int16_t g_recording[kNumSamples];

void Record(spelling::I2sAudioInput& input) {
  int16_t hop[kHop];
  int filled = 0;
  while (filled < kNumSamples) {
    const int n = (kNumSamples - filled) < kHop ? (kNumSamples - filled) : kHop;
    input.ReadHop(hop, n);
    for (int i = 0; i < n; ++i) {
      g_recording[filled + i] = hop[i];
    }
    filled += n;
  }
}

void Play(spelling::I2sAudioOutput& output) {
  // The I2S mic is quiet -- a raw recording is only a few percent of full scale.
  // Peak-normalize the whole clip so it uses the full output range (same
  // approach as the standalone speaker test and the echo service's clip
  // playback).
  const float gain =
      spelling::PeakNormalizeGain(g_recording, kNumSamples);
  printf("  (normalize gain %.1fx)\n", static_cast<double>(gain));
  fflush(stdout);
  output.SetGain(gain);
  output.Begin(kSampleRate, kNumSamples);
  constexpr int kChunk = 256;
  for (int i = 0; i < kNumSamples; i += kChunk) {
    const int n = (kNumSamples - i) < kChunk ? (kNumSamples - i) : kChunk;
    output.Write(g_recording + i, n);
  }
  output.End();
}

}  // namespace

int main() {
  vreg_set_voltage(VREG_VOLTAGE_1_20);
  sleep_ms(10);
  set_sys_clock_khz(250000, true);

  stdio_init_all();
  for (int i = 0; i < 1500 && !stdio_usb_connected(); ++i) {
    sleep_ms(20);
  }
  sleep_ms(200);

  printf("\n=== Audio loopback test ===\n");
  printf("Record %d s @ %d Hz from I2S mic, play on I2S amp, repeat.\n",
         kRecordSeconds, kSampleRate);
  printf("Mic: GP0/1/2   Amp: DIN=GP10 BCLK=GP11 LRC=GP12\n\n");

  spelling::I2sAudioInput input(kSampleRate);
  spelling::I2sAudioOutput output(/*data_pin=*/10, /*clock_base=*/11, kSampleRate);

  while (true) {
    printf("recording %d s...\n", kRecordSeconds);
    fflush(stdout);
    Record(input);
    printf("playing back...\n");
    fflush(stdout);
    Play(output);
    sleep_ms(500);
  }
  return 0;
}
