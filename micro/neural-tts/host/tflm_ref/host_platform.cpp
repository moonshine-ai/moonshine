// Host (desktop) implementations of the TFLM platform hooks the RP2350 build
// gets from pico-tflmicro's system_setup.cpp / micro_time.cpp (both call into
// the Pico SDK and can't compile off-device). This replaces them for the
// native TTS build: DebugLog -> stderr, a no-op InitializeTarget, and a
// std::chrono-backed micro_time so the profiler still reports sane numbers.

#include <chrono>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "tensorflow/lite/micro/micro_time.h"

extern "C" void DebugLog(const char* format, va_list args) {
  vfprintf(stderr, format, args);
}

extern "C" int DebugVsnprintf(char* buffer, size_t buf_size, const char* format,
                              va_list vlist) {
  return vsnprintf(buffer, buf_size, format, vlist);
}

namespace tflite {

void InitializeTarget() {}

uint32_t ticks_per_second() { return 1000000; }

uint32_t GetCurrentTimeTicks() {
  // Microsecond wall clock; only used for the optional profiler breakdown.
  static const auto t0 = std::chrono::steady_clock::now();
  const auto now = std::chrono::steady_clock::now();
  return static_cast<uint32_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now - t0).count());
}

}  // namespace tflite
