#include "echo_app.h"

#include <cstdio>

#include "app_common.h"
#include "audio_config.h"
#include "audio_service.h"
#include "kiss_fftr.h"
#include "mel_tables.h"
#include "usb_audio_io.h"
#include "vad_config.h"
#include "vad_mel_tables.h"

namespace spelling {

// The streaming VAD front-end shares the STT 512-pt FFT state and 512-tap Hann
// window (they never compute at the same time), so the only VAD-specific table
// is its 32-mel filterbank. These guarantee the shared pieces line up; both
// table sets are emitted from the same metadata, so a mismatch only happens on
// a partial regenerate.
static_assert(kVadMelTableNMels == kVadNMels,
              "vad_mel_tables.h n_mels != vad_config.h kVadNMels; regenerate");
static_assert(kVadMelTableNFft == kVadNFft,
              "vad_mel_tables.h n_fft != vad_config.h kVadNFft; regenerate");
static_assert(kMelTableNFft == kVadMelTableNFft,
              "STT/VAD n_fft differ -- cannot share the Hann window");

void RunEchoApp() {
  // One shared 512-pt real-FFT twiddle state for both the VAD streaming
  // front-end and the STT classifier's log-mel (same n_fft, never concurrent).
  kiss_fftr_state* fft =
      kiss_fftr_alloc(kNFft, /*inverse_fft=*/0, nullptr, nullptr);
  if (fft == nullptr) {
    printf("[boot] kiss_fftr_alloc(shared) failed\n");
    while (true) { /* halt */
    }
  }

  // The laptop is the board's A/D + D/A over USB. Swap UsbAudio* for an I2S
  // mic/DAC implementation (audio_io.h) to run the same service on real
  // on-board hardware -- nothing else changes.
  UsbAudioInput audio_in;
  UsbAudioOutput audio_out;
  RunAudioService(audio_in, audio_out, g_tensor_arena, kTensorArenaSize, fft,
                  g_waveform, kClipNumSamples);
  // RunAudioService never returns.
  while (true) { /* unreachable */
  }
}

}  // namespace spelling
