// tts -- portable, dependency-free formant (Klatt-style) text-to-speech.
//
// This is the single public header for the module. Include only this; it pulls
// in the public synth types. The remaining headers under include/tts/ are the
// transitive implementation types that StreamSynth exposes by value (the Klatt
// resonators, the per-frame parameter tracks, the voice config); the G2P
// internals live privately in src/.
//
// The core is intentionally self-contained: C++17, float-only math, small fixed
// state, no allocation in the audio inner loop, and NO third-party dependency
// (not even kissfft or TFLM). Only the streaming synthesis path used on the MCU
// is included here.
//
// Typical use (the platform glue -- USB command loop, arena ownership -- lives
// in the example, not here):
//
//   tts::VoiceParams voice = tts::DefaultVoiceParams();
//   tts::StreamSynth synth(voice, arena, arena_size);
//   tts::StreamOptions opts; opts.sample_rate = 22050.0f;
//   synth.BeginText("hello world", opts);          // or BeginIpa(...)
//   float buf[256];
//   for (int n; (n = synth.Read(buf, 256)) > 0; )  // stream PCM, never buffers
//     emit(buf, n);

#ifndef TTS_TTS_H_
#define TTS_TTS_H_

#include "config.h"        // tts::VoiceParams, tts::DefaultVoiceParams
#include "synth_stream.h"  // tts::StreamSynth, StreamOptions, StreamStatus

#endif  // TTS_TTS_H_
