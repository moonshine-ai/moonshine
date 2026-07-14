// Entry point for the voice-driven WiFi setup app on the on-board hardware
// audio I/O (the `moonshine_micro_echo_wifi_hardware` target; needs a Pico 2 W
// / CYW43).
//
// Same setup flow as main_wifi.cc, but the mic + speaker are the on-board I2S
// mic (GP0/1/2) and I2S amp (DIN=GP10/BCLK=GP11/LRC=GP12) instead of the USB
// audio bridge. See wifi_hardware_app.cc.

#include "app_common.h"
#include "wifi_hardware_app.h"

int main() {
  const unsigned led_pin = spelling::BoardInit();
  spelling::PrintBootBanner();
  (void)led_pin;  // the setup service owns the run and never returns
  spelling::RunWifiHardwareApp();
  return 0;
}
