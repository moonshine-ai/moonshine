#include "usb_audio_io.h"

#include <cstddef>
#include <cstdio>

#include "pico/stdlib.h"

namespace spelling {

namespace {

// Host->device framing: each hop is preceded by a 2-byte sync so the device can
// re-align if a byte is ever dropped on the CDC pipe.
constexpr int kSync0 = 0xA5;
constexpr int kSync1 = 0x5A;

// Read one byte, waiting up to ~timeout_ms. Returns -1 on timeout.
int ReadByteTimed(int timeout_ms) {
  const int c = getchar_timeout_us(timeout_ms * 1000);
  return (c == PICO_ERROR_TIMEOUT) ? -1 : (c & 0xFF);
}

}  // namespace

bool UsbAudioInput::ReadHop(int16_t* out, int n) {
  // Scan for the 0xA5 0x5A sync; return false if no data arrives for ~1 s (so
  // the caller can emit an idle heartbeat). The sample read also times out: if
  // the host drops part of a frame, an unbounded blocking read would stall the
  // device mid-hop forever -- it would stop consuming the CDC pipe, the host's
  // write() would block on a full endpoint, and the two would deadlock (which
  // on macOS wedges the USB handle). Timing out lets the stream self-heal.
  int idle_us = 0;
  int state = 0;  // 0: seeking 0xA5, 1: got 0xA5, seeking 0x5A
  for (;;) {
    const int c = getchar_timeout_us(100000);  // 100 ms slices
    if (c == PICO_ERROR_TIMEOUT) {
      idle_us += 100000;
      if (idle_us >= 1000000) return false;  // ~1 s with no bytes -> idle
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
  for (int i = 0; i < n; ++i) {
    const int lo = ReadByteTimed(300);
    if (lo < 0) return false;  // mid-hop stall -> abandon + resync next call
    const int hi = ReadByteTimed(300);
    if (hi < 0) return false;
    out[i] = static_cast<int16_t>((hi << 8) | lo);
  }
  return true;
}

void UsbAudioInput::Drain() {
  // Read and discard every byte currently available, without blocking, so the
  // host CDC pipe stays drained while we intentionally ignore the input.
  while (getchar_timeout_us(0) != PICO_ERROR_TIMEOUT) { /* discard */
  }
}

void UsbAudioOutput::Begin(int sample_rate, int num_samples, const char* kind) {
  printf("%s %d %d\n", kind, sample_rate, num_samples);
  fflush(stdout);
}

void UsbAudioOutput::Write(const int16_t* samples, int n) {
  fwrite(samples, sizeof(int16_t), static_cast<size_t>(n), stdout);
}

void UsbAudioOutput::End() {
  fflush(stdout);
  printf("\nEND\n");
  fflush(stdout);
}

}  // namespace spelling
