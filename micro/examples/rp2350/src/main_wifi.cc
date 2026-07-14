// Entry point for the voice-driven WiFi setup app
// (the `moonshine_micro_echo_wifi` target; needs a Pico 2 W / CYW43).
//
// Reuses the same VAD -> STT -> TTS pipeline as the live service, but drives a
// small state machine: spell an SSID + password, join the network over CYW43,
// and read the assigned IP back via TTS. See wifi_app.cc for the shared setup
// state machine (RunWifiAppWithIo); the on-board hardware-audio variant is
// main_wifi_hardware.cc.
//
// This variant uses the USB audio bridge: the laptop is the mic + speaker over
// the USB tether (scripts/usb_audio_bridge.py).

#include "app_common.h"
#include "usb_audio_io.h"  // UsbAudioInput / UsbAudioOutput
#include "wifi_app.h"      // RunWifiAppWithIo

int main() {
  const unsigned led_pin = spelling::BoardInit();
  spelling::PrintBootBanner();
  (void)led_pin;  // the setup service owns the run and never returns

  spelling::UsbAudioInput in;
  spelling::UsbAudioOutput out;
  spelling::RunWifiAppWithIo(in, out);
  return 0;
}
