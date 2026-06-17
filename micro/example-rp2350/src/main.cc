// cpp-tiny example app entry point.
//
// Boots the board (overclock, LED, USB stdio) and dispatches to one of two
// paths -- the heavy lifting lives in the modules and these two files:
//
//   * spelling_app.cc -- the live mic/speaker recognition service. DEFAULT.
//                        Streams audio in, runs VAD -> STT, speaks the result
//                        back. Built when SPELLING_TINY_AUDIO is set (the
//                        default), which forces VAD + TTS on.
//
//   * test_app.cc     -- the embedded-clip accuracy sweep. Opt-in via
//                        -DSPELLING_TINY_TEST_SWEEP=ON. Runs the clip set
//                        through the front-end + classifier and prints
//                        per-clip predictions, accuracy, and timing.
//
// Build the live service (default):   cmake ... && cmake --build build
// Build the test sweep instead:       cmake ... -DSPELLING_TINY_TEST_SWEEP=ON

#include "app_common.h"

#if defined(SPELLING_TINY_AUDIO) && !defined(SPELLING_TINY_TEST_SWEEP)
#include "spelling_app.h"
#else
#include "test_app.h"
#endif

int main() {
  const unsigned led_pin = spelling::BoardInit();
  spelling::PrintBootBanner();

#if defined(SPELLING_TINY_AUDIO) && !defined(SPELLING_TINY_TEST_SWEEP)
  (void)led_pin;  // the live service owns the LED-free run and never returns
  spelling::RunSpellingApp();
#else
  spelling::RunTestApp(led_pin);
#endif
  return 0;
}
