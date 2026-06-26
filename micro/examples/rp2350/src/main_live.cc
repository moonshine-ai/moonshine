// Entry point for the live mic/speaker echo service
// (the `moonshine_micro_echo` target).
//
// Streams microphone hops in (USB by default; see usb_audio_io.h), runs the
// VAD -> STT pipeline, and speaks the recognized letter/digit back via the TTS.
// The heavy lifting lives in echo_app.cc and the platform-agnostic modules.

#include "app_common.h"
#include "echo_app.h"

int main() {
  const unsigned led_pin = spelling::BoardInit();
  spelling::PrintBootBanner();
  (void)led_pin;  // the live service owns the run and never returns
  spelling::RunEchoApp();
  return 0;
}
