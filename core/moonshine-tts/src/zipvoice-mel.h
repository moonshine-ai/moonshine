#ifndef MOONSHINE_TTS_ZIPVOICE_MEL_H
#define MOONSHINE_TTS_ZIPVOICE_MEL_H

#include <cstdint>
#include <vector>

namespace moonshine_tts {

/// Log-mel feature frontend matching ZipVoice's ``VocosFbank`` (``zipvoice/utils/feature.py``),
/// which is a ``torchaudio.transforms.MelSpectrogram`` with defaults: sample_rate 24000, n_fft 1024,
/// hop 256, win_length 1024, periodic Hann window, center padding (reflect), power=1 (magnitude),
/// 100 mel bins, htk mel scale, no Slaney norm, followed by ``log(clamp(min=1e-7))``.
class VocosFbank {
 public:
  static constexpr int kSampleRate = 24000;
  static constexpr int kNFft = 1024;
  static constexpr int kHop = 256;
  static constexpr int kNMels = 100;

  VocosFbank();

  /// Number of STFT frames for ``num_samples`` (center padding): ``1 + num_samples / hop``.
  static int num_frames_for(size_t num_samples);

  /// Returns row-major log-mel features of shape ``[frames, 100]`` (``out[t * 100 + m]``). ``samples``
  /// is mono float PCM at ``kSampleRate``. ``out_frames`` receives the frame count.
  std::vector<float> extract(const std::vector<float>& samples, int* out_frames) const;

 private:
  std::vector<float> hann_;      // [kNFft]
  std::vector<float> fb_;        // [(kNFft/2 + 1) * kNMels], row-major [freq, mel]
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_ZIPVOICE_MEL_H
