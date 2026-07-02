#ifndef MOONSHINE_TTS_ZIPVOICE_TTS_H
#define MOONSHINE_TTS_ZIPVOICE_TTS_H

#include "file-information.h"
#include "moonshine-g2p-options.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

/// Options for the ZipVoice zero-shot voice-cloning ONNX TTS engine. The reference voice to clone
/// is supplied either as a built-in VCTK ``voice_id`` (see ``zipvoice-voices.h``) or as in-memory
/// mono float PCM (``clone_pcm`` + ``clone_sample_rate``) with an optional ``clone_transcript``
/// (when empty for a supplied clip, the caller is expected to have transcribed it upstream).
struct ZipVoiceTTSOptions {
  /// Locale tag (English only for now, e.g. ``en_us`` / ``en_gb``).
  std::string lang = "en_us";
  double speed = 1.0;
  MoonshineG2POptions g2p_options{};
  std::vector<std::string> ort_provider_names{};
  /// Peak-normalize the final waveform. Off by default: ZipVoice matches the reference clip's
  /// loudness, so peak normalization would override that. Clipping to ``[-1, 1]`` is always applied.
  bool normalize_audio = false;
  float output_volume = 1.F;

  /// Flow-matching / sampling controls. ``num_step`` <= 0 and ``guidance_scale`` < 0 mean "use the
  /// per-model default" (distill: 8 steps / guidance 3.0; full: 16 steps / guidance 1.0).
  int num_step = 0;
  float guidance_scale = -1.F;
  float t_shift = 0.5F;
  float feat_scale = 0.1F;
  float target_rms = 0.1F;
  /// Whether the deployed fm_decoder is the distilled model (affects only default steps/guidance).
  bool distill = true;
  /// Seed for the Gaussian latent used by the ODE solver (deterministic across runs).
  unsigned int seed = 666U;

  /// Built-in reference voice id (without the ``zipvoice_`` prefix), e.g. ``american_female``. When
  /// non-empty, ``clone_pcm`` / ``clone_transcript`` are ignored and the compiled-in clip is used.
  std::string voice_id{};
  /// Reference clip to clone as mono float PCM in ``[-1, 1]`` (used when ``voice_id`` is empty).
  std::vector<float> clone_pcm{};
  int clone_sample_rate = 24000;
  /// Transcript of ``clone_pcm`` (required for good cloning; empty is tolerated but degrades quality).
  std::string clone_transcript{};

  /// In-memory ZipVoice assets keyed by ``zipvoice/text_encoder.ort`` etc. (see moonshine-tts-options.h).
  FileInformationMap tts_asset_files{};
};

class ZipVoiceTTS {
 public:
  explicit ZipVoiceTTS(const ZipVoiceTTSOptions& opt);
  ZipVoiceTTS(const ZipVoiceTTS&) = delete;
  ZipVoiceTTS& operator=(const ZipVoiceTTS&) = delete;
  ZipVoiceTTS(ZipVoiceTTS&&) noexcept;
  ZipVoiceTTS& operator=(ZipVoiceTTS&&) noexcept;
  ~ZipVoiceTTS();

  static constexpr int kSampleRateHz = 24000;

  void set_speed(double speed);
  double speed() const;
  bool normalize_audio() const;
  void set_normalize_audio(bool on);
  float output_volume() const;
  void set_output_volume(float volume);

  /// Text -> IPA (MoonshineG2P) -> ZipVoice token ids -> text encoder / flow-matching decoder /
  /// vocoder -> mono float waveform at ``kSampleRateHz``, cloned in the reference voice.
  std::vector<float> synthesize(std::string_view text);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Post-process ZipVoice output: shorten internal pauses longer than ``max_silence_ms`` down to
/// ``keep_silence_ms`` with short crossfades at trim boundaries (``fade_ms``).
std::vector<float> zipvoice_compress_long_pauses(const std::vector<float>& wav, int sample_rate,
                                                  float max_silence_ms = 350.F,
                                                  float keep_silence_ms = 180.F,
                                                  float fade_ms = 12.F);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_ZIPVOICE_TTS_H
