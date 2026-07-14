// On-board hardware echo service (I2S mic + I2S amp).
//
// Same VAD -> STT -> TTS pipeline as the USB-tethered echo app, but microphone
// hops come from an SPH0645 I2S mic and spoken replies play on a MAX98357A I2S
// class-D amp. USB CDC is used only for text logs (VAD/RESULT lines), not the
// usb_audio_bridge.py audio protocol.

#ifndef SPELLING_ECHO_HARDWARE_APP_H_
#define SPELLING_ECHO_HARDWARE_APP_H_

namespace spelling {

[[noreturn]] void RunEchoHardwareApp();

}  // namespace spelling

#endif  // SPELLING_ECHO_HARDWARE_APP_H_
