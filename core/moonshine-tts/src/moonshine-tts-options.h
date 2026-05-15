#ifndef MOONSHINE_TTS_MOONSHINE_TTS_OPTIONS_H
#define MOONSHINE_TTS_MOONSHINE_TTS_OPTIONS_H

#include <filesystem>
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
/// Canonical map / CDN key for the Kokoro ONNX graph (``.onnx``; ``resolve_disk_model_file_path`` still prefers a sibling ``.ort`` on disk when present).
inline constexpr std::string_view kTtsKokoroModelOnnxKey = "kokoro/model.onnx";
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
  /// Prefix with ``kokoro_`` or ``piper_`` (case-insensitive) to select the vocoder; the prefix is stripped
  /// for asset resolution (e.g. ``kokoro_af_heart`` → Kokoro voice ``af_heart``).
  std::string voice{};
  double speed = 1.0;
  std::vector<std::string> ort_provider_names{};
  MoonshineG2POptions g2p_options{};
  FileInformationMap files{};
  /// ``kokoro``, ``piper``, or ``auto`` (pick Kokoro when ``kokoro_tts_lang_supported(language, g2p_options)``).
  /// Normally derived from the ``voice`` prefix; remains ``auto`` when ``voice`` has no engine prefix.
  std::string vocoder_engine = "auto";
  /// Peak-normalize the waveform produced by ``MoonshineTTS::synthesize`` (Kokoro or Piper) before applying
  /// ``output_volume`` and clipping. Matches ``piper-tts`` ``SynthesisConfig.normalize_audio`` semantics.
  bool normalize_audio = true;
  /// Linear gain applied after ``normalize_audio`` and before clipping to ``[-1, 1]``.
  /// Matches ``piper-tts`` ``SynthesisConfig.volume`` semantics.
  float output_volume = 1.F;
  std::optional<float> piper_noise_scale_override{};
  std::optional<float> piper_noise_w_override{};
  /// Default WAV path for CLI-style tooling (``-o`` / ``output`` in ``parse_options``).
  std::filesystem::path output_path = "out.wav";

  /// Relative or absolute path from ``files`` for key *k*, else ``std::filesystem::path(k)``.
  std::filesystem::path tts_relative_path(std::string_view canonical_key) const;

  // Whether to log profiling information to the console.
  bool log_profiling = false;

  /// Parses ``key=value``-style entries. G2P-specific keys are forwarded to ``g2p_options``.
  ///
  /// If ``cli_language`` is null and an entry names ``lang`` or ``language``, throws.
  /// If non-null, those keys write into ``*cli_language`` and set ``*language_was_set`` when provided.
  ///
  /// ``model_root`` / ``path_root`` / ``tts_root`` / ``g2p_root`` set ``g2p_options.g2p_root``.
  /// If none are set, the cwd is used as the asset root when constructing ``MoonshineTTS``.
  /// Piper file keys: ``piper_onnx``, ``piper_onnx_json``, ``piper_voices_dir``, ``piper_voices_json_dir``
  /// (hyphenated CLI flags are accepted). Other keys forward to ``g2p_options``.
  /// ``normalize_audio`` / ``output_volume`` keys (legacy aliases ``piper_normalize_audio`` /
  /// ``piper_output_volume``) configure the shared post-synthesis effects step.
  /// ``engine`` / ``vocoder_engine`` entries are accepted for compatibility but ignored (engine is encoded in ``voice``).
  void parse_options(const std::vector<std::pair<std::string, std::string>>& options,
                     std::string* cli_language = nullptr,
                     bool* language_was_set = nullptr);

  /// If ``voice`` starts with ``kokoro_`` or ``piper_`` (ASCII case-insensitive), sets ``vocoder_engine`` accordingly
  /// and removes that prefix from ``voice`` (after trim). Idempotent when there is no matching prefix.
  void apply_voice_engine_prefix();
};

/// Shared post-synthesis effects step for Kokoro and Piper output. Mirrors ``piper-tts``'s
/// ``SynthesisConfig`` behaviour: optional peak normalization, gain, then clip to ``[-1, 1]``.
void apply_synthesis_output_effects(std::vector<float>& audio, bool normalize_audio, float volume);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_MOONSHINE_TTS_OPTIONS_H
