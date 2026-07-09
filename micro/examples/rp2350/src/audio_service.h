// Live USB audio service: turns the laptop into the RP2350's mic + speaker.
//
// The host (examples/rp2350/scripts/usb_audio_bridge.py) captures the laptop mic at
// 16 kHz and streams it to the device as framed 32 ms hops over USB CDC. The
// device runs the streaming VAD on those hops; when a speech segment ends it
// front-aligns the 1 s clip, classifies it with the SpellingCNN, sends the
// result back, and speaks the recognized letter/digit via the formant TTS,
// streaming that PCM back to the host to play on the laptop speaker.
//
// VAD (~36 KiB arena), the SpellingCNN classifier (~366 KiB arena) and the TTS
// synth (arena working memory) never run at the same time, so they reuse the
// SINGLE tensor arena sequentially (Vad -> Classifier -> StreamSynth -> Vad).
// `window` is the reused 1 s fp32 waveform buffer (g_waveform) that doubles as
// the sliding capture window and the front-aligned STT clip; no extra big SRAM
// buffer is allocated.

#ifndef SPELLING_AUDIO_SERVICE_H_
#define SPELLING_AUDIO_SERVICE_H_

#include <cstddef>
#include <cstdint>

#include "audio_io.h"

struct kiss_fftr_state;

namespace spelling {

// Recognitions below this softmax probability are logged but discarded (not
// acted on). The host bridge resumes listening on a sub-threshold RESULT
// instead of waiting for a spoken reply, so callers should keep this contract.
constexpr float kMinResultProb = 0.5f;

// Build the persistent recognizer front-end (the streaming VAD mel front-end,
// the STT log-mel, and their shared 512-pt FFT). Call once before RecognizeOne.
// The heavy front-end objects live in BSS (not the small core stack).
void RecognizerInit(kiss_fftr_state* fft);

// Listen on `input` until the streaming VAD reports a completed speech segment,
// front-align the 1 s clip into `window`, classify it with the SpellingCNN, and
// return the predicted class index (writing the softmax probability to
// `*out_prob`). Blocks (emitting idle heartbeats) until a clip is recognized.
// Reuses the shared tensor `arena` for the VAD then the classifier. Emits the
// same "listening" / "VAD start" / "VAD end" / "RESULT" logs the host expects.
int RecognizeOne(AudioInput& input, uint8_t* arena, std::size_t arena_size,
                 float* window, int window_samples, float* out_prob);

// Synthesize `text` with the formant TTS and stream it to `output`, draining
// `input` while we speak so our own audio (and room echo) can't re-trigger the
// VAD, then hold a short mute tail. Reuses the tensor `arena` as synth scratch.
void Speak(const char* text, AudioOutput& output, AudioInput& input,
           uint8_t* arena, std::size_t arena_size);

// Never returns: runs the live recognition loop forever. Microphone audio comes
// from `input`, the spoken reply goes to `output` -- so the same loop runs on
// host-tethered USB audio or real on-board I2S hardware unchanged. `arena`/
// `arena_size` is the shared TFLM tensor arena; `fft` is the shared 512-pt
// real-FFT state; `window`/`window_samples` is the 1 s fp32 buffer
// (kClipNumSamples) reused as the sliding capture window and the STT clip.
void RunAudioService(AudioInput& input, AudioOutput& output, uint8_t* arena,
                     std::size_t arena_size, kiss_fftr_state* fft,
                     float* window, int window_samples);

}  // namespace spelling

#endif  // SPELLING_AUDIO_SERVICE_H_
