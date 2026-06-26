// Voice-driven WiFi setup app (the moonshine_micro_echo_wifi target; needs a
// Pico 2 W / CYW43 board, i.e. -DPICO_BOARD=pico2_w).
//
// Reuses the same VAD -> STT -> TTS pipeline as the default echo service
// (RecognizeOne / Speak in audio_service.h), but drives a small state machine
// instead of echoing single letters: the user says "wifi", spells an SSID and
// password character-by-character (letters, digits, the symbol words, and the
// "capital"/"delete"/"done"/"cancel"/"yes"/"no" command words the model already
// knows), the device joins the network over CYW43, and "ip" reads the assigned
// address back via TTS.

#ifndef SPELLING_WIFI_APP_H_
#define SPELLING_WIFI_APP_H_

namespace spelling {

// Run the voice WiFi-setup service forever (never returns). Assumes the board
// has been brought up (BoardInit) and that this is a Pico 2 W build.
[[noreturn]] void RunWifiApp();

}  // namespace spelling

#endif  // SPELLING_WIFI_APP_H_
