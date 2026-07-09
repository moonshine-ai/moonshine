// Shared boot + state for the RP2350 example, used by both app paths:
//   * echo_app -- the live mic/speaker recognition service (default)
//   * test_app     -- the embedded-clip accuracy sweep (opt-in test)
//
// main.cc does the board bring-up here, then dispatches to one path.

#ifndef SPELLING_APP_COMMON_H_
#define SPELLING_APP_COMMON_H_

#include <cstddef>
#include <cstdint>

#include "audio_config.h"  // kNMels, kClipNumSamples

namespace spelling {

// Tensor arena. Auto-scales with the model dimensions: it clears the 64x128
// model's ~346 KiB working set with headroom; a model with >64 mels needs
// ~1.28 MB and can never fit, so it falls back to 256 KiB and the build still
// links + boots far enough to print a clear "arena too small" error.
//
// There is no separate fp32 feature buffer -- features are computed into a
// slice of the (idle) activation overlay, so feature generation and inference
// share the same bytes. VAD/STT/TTS reuse this one arena sequentially.
//
// The provisioned size for the <=64-mel model is overridable per target via the
// SPELLING_TINY_ARENA_BYTES build define. The WiFi target trims it from 384 KiB
// to 360 KiB to free ~24 KiB of SRAM for the CYW43 driver + lwIP stack; 360 KiB
// still clears the ~346 KiB classifier working set with ~14 KiB of headroom.
#ifndef SPELLING_TINY_ARENA_BYTES
#define SPELLING_TINY_ARENA_BYTES (384u * 1024u)
#endif
constexpr std::size_t kTensorArenaSize =
    (kNMels <= 64) ? (SPELLING_TINY_ARENA_BYTES) : (256u * 1024u);

// Shared large static buffers (defined, 16-byte aligned, in app_common.cc).
// g_waveform (1 s @ fp32) doubles as the live capture window and the STT clip.
extern uint8_t g_tensor_arena[kTensorArenaSize];
extern float g_waveform[kClipNumSamples];

// Bring up the board: overclock to 250 MHz (core voltage bumped first), LED
// POST pulses, USB CDC stdio, and wait up to ~30 s for the host to open the
// port. Returns the LED GPIO pin so a caller can keep blinking it.
unsigned BoardInit();

// Print the boot banner shared by both paths (clock, cores, model size,
// classes, mel config, arena size).
void PrintBootBanner();

// Pulse the LED `count` times (POST / fault / heartbeat signaling).
void LedPulse(unsigned pin, int count, int on_ms, int off_ms);

}  // namespace spelling

#endif  // SPELLING_APP_COMMON_H_
