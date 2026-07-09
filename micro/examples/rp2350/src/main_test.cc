// Entry point for the embedded-clip accuracy sweep
// (the `moonshine_micro_echo_test` target).
//
// Runs the embedded clip set through the front-end + classifier and prints
// per-clip predictions, accuracy, and timing; optionally a VAD demo and/or the
// TTS USB command loop (enabled via the SPELLING_TINY_VAD / SPELLING_TINY_TTS /
// SPELLING_TINY_PROFILE_OPS build options). See test_app.cc.

#include "app_common.h"
#include "test_app.h"

int main() {
  const unsigned led_pin = spelling::BoardInit();
  spelling::PrintBootBanner();
  spelling::RunTestApp(led_pin);
  return 0;
}
