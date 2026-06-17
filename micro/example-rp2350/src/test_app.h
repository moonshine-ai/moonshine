// The opt-in test path (SPELLING_TINY_TEST_SWEEP): run the embedded clip set
// through the front-end + classifier and report per-clip predictions, overall
// accuracy, and per-clip log-mel / inference timing over USB stdio. Optionally
// runs a streaming VAD demo first and/or drops into the TTS USB command loop
// after, then idles with a heartbeat.

#ifndef SPELLING_TEST_APP_H_
#define SPELLING_TEST_APP_H_

namespace spelling {

// Run the embedded-clip test sweep, optional VAD demo / TTS loop, then idle.
// `led_pin` is blinked as a liveness heartbeat. Never returns.
[[noreturn]] void RunTestApp(unsigned led_pin);

}  // namespace spelling

#endif  // SPELLING_TEST_APP_H_
