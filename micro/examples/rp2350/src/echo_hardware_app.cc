#include "echo_hardware_app.h"

#include <cstdio>

#include "app_common.h"
#include "audio_config.h"
#include "audio_service.h"
#include "i2s_audio_io.h"
#include "i2s_audio_out.h"
#include "kiss_fftr.h"
#include "mel_tables.h"
#include "vad_config.h"
#include "vad_mel_tables.h"

namespace spelling {

static_assert(kVadMelTableNMels == kVadNMels,
              "vad_mel_tables.h n_mels != vad_config.h kVadNMels; regenerate");
static_assert(kVadMelTableNFft == kVadNFft,
              "vad_mel_tables.h n_fft != vad_config.h kVadNFft; regenerate");
static_assert(kMelTableNFft == kVadMelTableNFft,
              "STT/VAD n_fft differ -- cannot share the Hann window");

void RunEchoHardwareApp() {
  kiss_fftr_state* fft =
      kiss_fftr_alloc(kNFft, /*inverse_fft=*/0, nullptr, nullptr);
  if (fft == nullptr) {
    printf("[boot] kiss_fftr_alloc(shared) failed\n");
    while (true) { /* halt */
    }
  }

  I2sAudioInput audio_in(kVadSampleRate);
  I2sAudioOutput audio_out(/*data_pin=*/10, /*clock_base=*/11, kVadSampleRate);

  printf("[audio] hardware I/O: I2S mic GP0/1/2, I2S amp DIN=GP10 BCLK=GP11 LRC=GP12\n");
  printf("[audio] USB CDC logging only (no usb_audio_bridge protocol)\n");
  fflush(stdout);

  RunAudioService(audio_in, audio_out, g_tensor_arena, kTensorArenaSize, fft,
                  g_waveform, kClipNumSamples);
  while (true) { /* unreachable */
  }
}

}  // namespace spelling
