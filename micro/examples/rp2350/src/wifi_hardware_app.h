// Voice-driven WiFi setup on the on-board hardware audio I/O (I2S mic + I2S
// amp) instead of the USB audio bridge.
//
// Identical setup flow to the default wifi app (see wifi_app.h): it reuses the
// shared RunWifiAppWithIo() state machine and only swaps the audio backend for
// I2sAudioInput + I2sAudioOutput. Needs a Pico 2 W / CYW43 board
// (-DPICO_BOARD=pico2_w).

#ifndef SPELLING_WIFI_HARDWARE_APP_H_
#define SPELLING_WIFI_HARDWARE_APP_H_

namespace spelling {

// Run the voice WiFi-setup service forever (never returns) on the on-board I2S
// mic + I2S amp. Assumes BoardInit() has run; brings up CYW43 itself.
[[noreturn]] void RunWifiHardwareApp();

}  // namespace spelling

#endif  // SPELLING_WIFI_HARDWARE_APP_H_
