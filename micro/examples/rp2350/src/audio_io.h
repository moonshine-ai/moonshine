// Audio input/output abstraction for the live recognition service.
//
// The live VAD -> STT -> TTS path needs a microphone (16 kHz int16 mono hops
// in) and a speaker (int16 PCM out). Today both are provided by the laptop over
// USB CDC (the laptop is the RP2350's A/D + D/A; see usb_audio_io.{h,cc}).
// Tomorrow they could be a real I2S mic + DAC wired to the board. By routing
// the service through these two interfaces, swapping host-tethered audio for
// on-board hardware is a matter of passing a different AudioInput / AudioOutput
// to RunAudioService() -- no change to the recognition logic.

#ifndef SPELLING_AUDIO_IO_H_
#define SPELLING_AUDIO_IO_H_

#include <cstdint>

namespace spelling {

// Source of microphone audio, delivered one fixed-size hop at a time.
class AudioInput {
 public:
  virtual ~AudioInput() = default;

  // Read one hop of `n` int16 mono samples into `out`. Returns true on success;
  // false if no audio arrived within the source's idle window (the caller may
  // emit a heartbeat and call again). Implementations should self-heal on
  // partial / dropped data rather than block forever.
  virtual bool ReadHop(int16_t* out, int n) = 0;

  // Discard any buffered input without blocking. Used to mute (but keep the
  // pipe drained) while the device is speaking its own reply.
  virtual void Drain() = 0;
};

// Sink for synthesized speech, streamed one chunk at a time.
class AudioOutput {
 public:
  virtual ~AudioOutput() = default;

  // Announce an utterance of `num_samples` int16 mono samples at `sample_rate`.
  // `kind` labels the stream for the host: "AUDIO" (TTS reply) or "CLIP"
  // (captured mic clip played back for debugging). Hardware sinks ignore it.
  virtual void Begin(int sample_rate, int num_samples,
                     const char* kind = "AUDIO") = 0;

  // Emit `n` int16 samples.
  virtual void Write(const int16_t* samples, int n) = 0;

  // Finish the current utterance.
  virtual void End() = 0;
};

}  // namespace spelling

#endif  // SPELLING_AUDIO_IO_H_
