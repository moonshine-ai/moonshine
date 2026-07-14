// Entry point for the on-board hardware echo service
// (the `moonshine_micro_echo_hardware` target).
//
// I2S mic (GP0/1/2) + I2S amp (DIN=GP10/BCLK=GP11/LRC=GP12). No
// usb_audio_bridge.py support.

#include "app_common.h"
#include "echo_hardware_app.h"

int main() {
  const unsigned led_pin = spelling::BoardInit();
  spelling::PrintBootBanner();
  (void)led_pin;
  spelling::RunEchoHardwareApp();
  return 0;
}
