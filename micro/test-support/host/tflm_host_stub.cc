// Host (desktop) implementations of the handful of TFLM platform hooks that
// the modules and the micro_test.h harness reference: the MicroPrintf logging
// family and tflite::InitializeTarget().
//
// The vendored pico-tflmicro provides these for the RP2350 via
// system_setup.cpp / debug_log.cpp, both of which call into the Pico SDK and
// therefore cannot be compiled on the host. For the pure-logic unit tests
// (feature-generation, the VAD segmenter, the STT predictor, the TTS G2P) we
// don't need the interpreter at all -- only logging and a no-op init -- so this
// tiny shim lets those tests build and run on a laptop with a normal compiler.
//
// Interpreter-level tests (real MicroInterpreter inference) are built for the
// RP2350 target instead, where the full pico-tflmicro library is available.

#include <cstdarg>
#include <cstddef>
#include <cstdio>

void MicroPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  vfprintf(stderr, format, args);
  va_end(args);
  fputc('\n', stderr);
}

void VMicroPrintf(const char* format, va_list args) {
  vfprintf(stderr, format, args);
  fputc('\n', stderr);
}

int MicroSnprintf(char* buffer, size_t buf_size, const char* format, ...) {
  va_list args;
  va_start(args, format);
  const int r = vsnprintf(buffer, buf_size, format, args);
  va_end(args);
  return r;
}

int MicroVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist) {
  return vsnprintf(buffer, buf_size, format, vlist);
}

namespace tflite {
void InitializeTarget() {}
}  // namespace tflite
