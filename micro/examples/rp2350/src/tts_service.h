// USB streaming text-to-speech service for the RP2350 firmware.
//
// After the spelling-recognition sweep finishes, the app hands its (now-idle)
// TFLM tensor arena to this service, which drops into a USB CDC command loop
// driving the neural TTS engine (neural-tts/, diphone units + RVQ decoder +
// WORLD-lite vocoder; fixed 16 kHz output):
//
//   host -> "SPEAK the quick brown fox\n"   (on-device G2P)
//   host -> "IPA  h\xC9\x99lo\xCA\x8A\n"     (raw IPA, bypasses G2P)
//
//   device -> "AUDIO <sample_rate> <num_samples>\n"   (framing header)
//   device -> <num_samples * int16 little-endian PCM bytes>
//   device -> "\nEND <num_samples>\n"
//
// The sample count in the header is exact: the engine runs its deterministic
// planning passes first (EstimateSamples), then streams PCM as it renders.
// Recognition and synthesis never run at the same time, so the synthesizer
// safely reuses the whole tensor arena as its working memory.

#ifndef SPELLING_TTS_SERVICE_H_
#define SPELLING_TTS_SERVICE_H_

#include <cstddef>
#include <cstdint>

namespace spelling {

// Post-mortem from the previous boot (filled in by main from the watchdog
// scratch registers before it clears them). Reported by the STATUS command:
// the boot banner itself usually prints before the host has opened the CDC
// port, so this is the reliable way to read a hang report.
struct BootReport {
  bool watchdog_reboot;
  uint32_t ckpt;      // last tts_checkpoint value
  uint32_t ckpt2;     // last tts_checkpoint2 value
  uint32_t trace;     // last tts_trace (tag << 24 | val)
  uint32_t fault_pc;  // stacked PC from the hard fault handler (0 = none)
};
void SetBootReport(const BootReport& report);

// Enter the text-to-speech command loop using `arena` (the reused TFLM tensor
// arena) as the synthesizer's working memory. Never returns.
[[noreturn]] void RunTtsService(uint8_t* arena, std::size_t arena_size);

}  // namespace spelling

#endif  // SPELLING_TTS_SERVICE_H_
