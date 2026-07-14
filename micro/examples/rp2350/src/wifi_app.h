// Shared voice-driven WiFi setup state machine. Backs two targets that differ
// ONLY in their audio backend (needs a Pico 2 W / CYW43, -DPICO_BOARD=pico2_w):
//   * moonshine_micro_echo_wifi          USB audio bridge   [main_wifi.cc]
//   * moonshine_micro_echo_wifi_hardware I2S mic + PWM spkr  [wifi_hardware_app.cc]
//
// Reuses the same VAD -> STT -> TTS pipeline as the default echo service
// (RecognizeOne / Speak in audio_service.h), but drives a small state machine
// instead of echoing single letters: the user says "wifi", spells an SSID and
// password character-by-character (letters, digits, the symbol words, and the
// "capital"/"delete"/"finish"/"cancel"/"yes"/"no" command words the model already
// knows), the device joins the network over CYW43, and "ip" reads the assigned
// address back via TTS.

#ifndef SPELLING_WIFI_APP_H_
#define SPELLING_WIFI_APP_H_

#include "audio_io.h"  // AudioInput / AudioOutput

namespace spelling {

// Run the voice WiFi-setup service forever (never returns), driving the VAD ->
// STT -> TTS state machine over the supplied audio I/O. The I/O backend is the
// only thing that differs between the USB-bridge build (main_wifi.cc) and the
// on-board hardware build (wifi_hardware_app.cc): each just constructs the
// right AudioInput/AudioOutput and calls this. Assumes BoardInit() has run;
// brings up the CYW43 radio itself. This is a Pico 2 W (CYW43) build only.
[[noreturn]] void RunWifiAppWithIo(AudioInput& in, AudioOutput& out);

}  // namespace spelling

#endif  // SPELLING_WIFI_APP_H_
