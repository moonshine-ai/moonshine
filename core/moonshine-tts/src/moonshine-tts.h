#ifndef MOONSHINE_TTS_MOONSHINE_TTS_H
#define MOONSHINE_TTS_MOONSHINE_TTS_H

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "file-information.h"
#include "moonshine-g2p-options.h"

namespace moonshine_tts {

using MoonshineTTSFileInformation = FileInformation;

/// Canonical keys for TTS asset paths (relative to ``g2p_options.g2p_root`` unless paths are absolute).
/// When ``g2p_root`` is empty, ``MoonshineTTS`` uses the process current working directory as the root.
inline constexpr std::string_view kTtsKokoroModelOnnxKey = "kokoro/model.ort";
inline constexpr std::string_view kTtsKokoroConfigJsonKey = "kokoro/config.json";
/// Optional explicit Piper ONNX model (``*.onnx``).
inline constexpr std::string_view kTtsPiperOnnxKey = "piper/onnx";
/// Optional Piper model config (``*.onnx.json``) when it is not beside the ONNX file.
inline constexpr std::string_view kTtsPiperOnnxJsonKey = "piper/onnx.json";
/// Optional Piper voice directory (``*.onnx``) when ``kTtsPiperOnnxKey`` is unset.
inline constexpr std::string_view kTtsPiperVoicesKey = "piper/voices";
/// Optional directory of ``*.onnx.json`` files parallel to ``kTtsPiperVoicesKey`` (same basename rules).
inline constexpr std::string_view kTtsPiperVoicesJsonKey = "piper/voices_json";

/// Shared configuration for ``MoonshineTTS`` (Kokoro and Piper file paths, G2P, ORT, CLI-oriented fields).
/// Vocoder assets use ``files`` (same pattern as ``MoonshineG2POptions::files``). The **language** is passed to
/// ``MoonshineTTS``'s constructor.
struct MoonshineTTSOptions {
  MoonshineTTSOptions();

  std::vector<MoonshineTTSFileInformation> file_information{};
  /// Kokoro voice id (e.g. ``af_heart``) or Piper ONNX stem/basename when using Piper.
  std::string voice{};
  double speed = 1.0;
  std::vector<std::string> ort_provider_names{};
  MoonshineG2POptions g2p_options{};
  FileInformationMap files{};
  /// ``kokoro``, ``piper``, or ``auto`` (pick Kokoro when ``kokoro_tts_lang_supported(language, g2p_options)``).
  std::string vocoder_engine = "auto";
  bool piper_normalize_audio = true;
  float piper_output_volume = 1.F;
  std::optional<float> piper_noise_scale_override{};
  std::optional<float> piper_noise_w_override{};
  /// Default WAV path for CLI-style tooling (``-o`` / ``output`` in ``parse_options``).
  std::filesystem::path output_path = "out.wav";

  /// Relative or absolute path from ``files`` for key *k*, else ``std::filesystem::path(k)``.
  std::filesystem::path tts_relative_path(std::string_view canonical_key) const;

  /// Parses ``key=value``-style entries. G2P-specific keys are forwarded to ``g2p_options``.
  ///
  /// If ``cli_language`` is null and an entry names ``lang`` or ``language``, throws.
  /// If non-null, those keys write into ``*cli_language`` and set ``*language_was_set`` when provided.
  ///
  /// ``model_root`` / ``path_root`` / ``tts_root`` / ``g2p_root`` set ``g2p_options.g2p_root``.
  /// If none are set, the cwd is used as the asset root when constructing ``MoonshineTTS``.
  /// Piper file keys: ``piper_onnx``, ``piper_onnx_json``, ``piper_voices_dir``, ``piper_voices_json_dir``
  /// (hyphenated CLI flags are accepted). Other keys forward to ``g2p_options``.
  void parse_options(const std::vector<std::pair<std::string, std::string>>& options,
                     std::string* cli_language = nullptr,
                     bool* language_was_set = nullptr);
};

bool kokoro_tts_lang_supported(std::string_view lang_cli, const MoonshineG2POptions& g2p_opt = {});

/// Unified TTS: **Kokoro** and **Piper** ONNX backends; shared ``MoonshineG2P`` where applicable.
class MoonshineTTS {
 public:
  MoonshineTTS(std::string_view language, const MoonshineTTSOptions& opt);
  MoonshineTTS(const MoonshineTTS&) = delete;
  MoonshineTTS& operator=(const MoonshineTTS&) = delete;
  MoonshineTTS(MoonshineTTS&&) noexcept;
  MoonshineTTS& operator=(MoonshineTTS&&) noexcept;
  ~MoonshineTTS();

  static constexpr int kSampleRateHz = 24000;

  std::vector<float> synthesize(std::string_view text);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void write_wav_mono_pcm16(const std::filesystem::path& path,
                          const std::vector<float>& samples);

}  // namespace moonshine_tts

#endif
