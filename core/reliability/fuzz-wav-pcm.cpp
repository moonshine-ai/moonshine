// libFuzzer harness for the WAV/RIFF parsers.
//
// Both readers take a file path, so we write the fuzz input to a temporary file
// and parse it with (1) the first-party 16-bit PCM reader used in production
// and (2) the vendored header-only reader used by diarization. libFuzzer runs
// one input at a time in a single thread, so a per-iteration temp file is safe.
//
// This is test-only tooling, so the raw C file APIs and free() used here are
// exempt from the library style rules (see core/STYLE_GUIDE.md).

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <string>

#include "debug-utils.h"
#include "wav_pcm_float32.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  char path_template[] = "/tmp/moonshine_fuzz_wav_XXXXXX";
  const int fd = mkstemp(path_template);
  if (fd < 0) {
    return 0;
  }
  if (size > 0) {
    const ssize_t written = write(fd, data, size);
    (void)written;
  }
  close(fd);

  // First-party 16-bit PCM reader (production path).
  float *samples = nullptr;
  size_t num_samples = 0;
  int32_t sample_rate = 0;
  if (load_wav_data(path_template, &samples, &num_samples, &sample_rate)) {
    free(samples);
  }

  // Vendored header-only reader used by diarization.
  try {
    int sr = 0;
    wav_pcm::load_wav_pcm16_mono_float32(std::string(path_template), sr);
  } catch (const std::exception &) {
    // Malformed WAVs legitimately throw.
  }

  unlink(path_template);
  return 0;
}
