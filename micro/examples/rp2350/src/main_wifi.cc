// Entry point for the voice-driven WiFi setup app
// (the `moonshine_micro_echo_wifi` target; needs a Pico 2 W / CYW43).
//
// Reuses the same VAD -> STT -> TTS pipeline as the live service, but drives a
// small state machine: spell an SSID + password, join the network over CYW43,
// and read the assigned IP back via TTS. See wifi_app.cc.

#include "app_common.h"
#include "wifi_app.h"

int main() {
  const unsigned led_pin = spelling::BoardInit();
  spelling::PrintBootBanner();
  (void)led_pin;  // the setup service owns the run and never returns
  spelling::RunWifiApp();
  return 0;
}
