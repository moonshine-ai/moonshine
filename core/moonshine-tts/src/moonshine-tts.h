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
  /// ``engine`` / ``vocoder_engine`` entries are accepted for compatibility but ignored (engine is encoded in ``voice``).
  void parse_options(const std::vector<std::pair<std::string, std::string>>& options,
                     std::string* cli_language = nullptr,
                     bool* language_was_set = nullptr);

  /// If ``voice`` starts with ``kokoro_`` or ``piper_`` (ASCII case-insensitive), sets ``vocoder_engine`` accordingly
  /// and removes that prefix from ``voice`` (after trim). Idempotent when there is no matching prefix.
  void apply_voice_engine_prefix();
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

  /// Synthesize with per-call option overrides (same key names as ``MoonshineTTSOptions::parse_options``).
  /// Currently only ``speed`` is applied at synthesis time; other entries are ignored.
  std::vector<float> synthesize(
      std::string_view text,
      const std::vector<std::pair<std::string, std::string>>& option_overrides);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void write_wav_mono_pcm16(const std::filesystem::path& path,
                          const std::vector<float>& samples);

/// Kokoro or Piper vocoder asset keys only (no G2P), relative to ``g2p_root``.
/// With default ``MoonshineTTSOptions{}``, uses ``vocoder_engine=auto`` and default voice layout.
std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(std::string_view language_cli);
/// Uses ``voice`` (optional ``kokoro_`` / ``piper_`` prefix sets vocoder), Piper/Kokoro file map entries, and
/// ``g2p_options`` (e.g. Spanish narrow obstruents for auto engine) like ``MoonshineTTS``.
std::vector<std::string> moonshine_catalog_tts_vocoder_only_dependency_keys(
    std::string_view language_cli, const MoonshineTTSOptions& options);

/// Union of vocoder keys across all languages in ``moonshine_asset_catalog_all_registered_language_tags``.
std::vector<std::string> moonshine_catalog_all_tts_vocoder_dependency_keys_union();

/// One Kokoro or Piper voice id and whether the asset is available (on disk or in-memory file map).
struct MoonshineTtsVoiceAvailability {
  std::string id;
  bool available = false;
};

/// All known voices for ``language_cli`` with availability, using the same path layout rules as
/// ``moonshine_catalog_tts_vocoder_only_dependency_keys``. Returned ``id`` values are prefixed with ``kokoro_`` or
/// ``piper_``. When vocoder is ``auto``, Kokoro and Piper catalogs are merged (both prefixes). Kokoro uses the
/// upstream Kokoro-82M voice catalog (VOICES.md) plus any extra ``*.kokorovoice`` under the resolved voices
/// directory. Piper uses the language default ONNX stem plus any ``*.onnx`` in the resolved voices directory.
/// The ``voice`` field in ``options`` does not filter the list.
std::vector<MoonshineTtsVoiceAvailability> moonshine_list_tts_voices_with_availability(
    std::string_view language_cli, const MoonshineTTSOptions& options);

}  // namespace moonshine_tts

#endif
