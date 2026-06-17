// USB streaming text-to-speech service for the RP2350 firmware.
//
// After the spelling-recognition sweep finishes, the app hands its (now-idle)
// TFLM tensor arena to this service, which drops into a USB CDC command loop:
//
//   host -> "SPEAK the quick brown fox\n"   (on-device G2P)
//   host -> "IPA  h\xC9\x99lo\xCA\x8A\n"     (raw IPA, bypasses G2P)
//   host -> "RATE 16000\n" | "SPEED 0.9\n" | "GENDER 0.7\n"
//
//   device -> "AUDIO <sample_rate> <num_samples>\n"   (framing header)
//   device -> <num_samples * int16 little-endian PCM bytes>
//   device -> "\nEND <num_samples>\n"
//
// Recognition and synthesis never run at the same time, so the synthesizer
// safely reuses the whole tensor arena as its working memory (the streaming
// engine never materializes the full waveform -- see tts/synth_stream.h).

#ifndef SPELLING_TTS_SERVICE_H_
#define SPELLING_TTS_SERVICE_H_

#include <cstddef>
#include <cstdint>

namespace spelling {

// Enter the text-to-speech command loop using `arena` (the reused TFLM tensor
// arena) as the synthesizer's working memory. Never returns.
[[noreturn]] void RunTtsService(uint8_t* arena, std::size_t arena_size);

}  // namespace spelling

#endif  // SPELLING_TTS_SERVICE_H_
