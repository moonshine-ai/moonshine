// Default (no-op) progress hooks. pb_decoder.cc calls tts_checkpoint /
// tts_trace before every TFLM op so a bring-up app can trap hangs and
// feed a watchdog; production apps that don't care get these weak stubs,
// and any app can override them with strong definitions.

#include <cstdint>

extern "C" __attribute__((weak)) void tts_checkpoint(uint32_t) {}
extern "C" __attribute__((weak)) void tts_checkpoint2(uint32_t) {}
extern "C" __attribute__((weak)) void tts_trace(uint32_t, uint32_t) {}
