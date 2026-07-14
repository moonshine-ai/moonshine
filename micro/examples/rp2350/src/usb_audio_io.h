// USB CDC implementation of the AudioInput / AudioOutput interfaces: the laptop
// acts as the RP2350's microphone and speaker.
//
//   * UsbAudioInput  -- reads 0xA5 0x5A-synced int16 hops streamed from the
//   host
//                       (scripts/usb_audio_bridge.py, in this example's scripts/,
//                       captures the laptop mic).
//   * UsbAudioOutput -- writes framed PCM the host plays back:
//                         CLIP <rate> <n>   (captured mic clip, debug)
//                         AUDIO <rate> <n>  (TTS reply)
//                         <n * int16 little-endian>
//                         \nEND\n
//
// Both are thin wrappers over the Pico SDK stdio CDC byte pipe. Swap them for
// an I2S mic/DAC implementation to run the same service on real on-board
// hardware.

#ifndef SPELLING_USB_AUDIO_IO_H_
#define SPELLING_USB_AUDIO_IO_H_

#include "audio_io.h"

namespace spelling {

class UsbAudioInput : public AudioInput {
 public:
  bool ReadHop(int16_t* out, int n) override;
  void Drain() override;
};

class UsbAudioOutput : public AudioOutput {
 public:
  void Begin(int sample_rate, int num_samples,
             const char* kind = "AUDIO") override;
  void Write(const int16_t* samples, int n) override;
  void End() override;
};

}  // namespace spelling

#endif  // SPELLING_USB_AUDIO_IO_H_
