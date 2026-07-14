#include "wifi_hardware_app.h"

#include <cstdio>

#include "audio_config.h"   // kVadSampleRate (via vad_config), kClipNumSamples
#include "i2s_audio_io.h"   // I2sAudioInput
#include "i2s_audio_out.h"  // I2sAudioOutput
#include "vad_config.h"     // kVadSampleRate
#include "wifi_app.h"       // RunWifiAppWithIo (shared setup state machine)

namespace spelling {

void RunWifiHardwareApp() {
  // Same wiring as the hardware echo app: I2S mic on GP0/1/2, I2S amp on
  // DIN=GP10/BCLK=GP11/LRC=GP12. Constructed before RunWifiAppWithIo() brings up
  // CYW43 -- the radio's PIO/GPIO use is disjoint from the mic (pio0 SM0) and
  // the amp (pio0 SM1).
  I2sAudioInput in(kVadSampleRate);
  I2sAudioOutput out(/*data_pin=*/10, /*clock_base=*/11, kVadSampleRate);

  printf("[wifi] hardware I/O: I2S mic GP0/1/2, I2S amp DIN=GP10 BCLK=GP11 LRC=GP12\n");
  printf("[wifi] USB CDC logging only (no usb_audio_bridge protocol)\n");
  fflush(stdout);

  RunWifiAppWithIo(in, out);
}

}  // namespace spelling
