#ifndef MOONSHINE_TTS_PIPER_TTS_H
#define MOONSHINE_TTS_PIPER_TTS_H

#include "file-information.h"
#include "moonshine-g2p-options.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace moonshine_tts {

/// Piper ONNX TTS + ``MoonshineG2P`` IPA (filtered to each model's ``phoneme_id_map``; no eSpeak at runtime).
struct PiperTTSOptions {
  /// When non-empty, load this ``*.onnx`` directly; ``voices_dir`` / discovery is skipped.
  /// Config JSON defaults to ``explicit_onnx_path`` with ``.onnx.json`` (same folder) unless
  /// ``explicit_onnx_json_path`` is set.
  std::filesystem::path explicit_onnx_path{};
  /// Piper model config (``*.onnx.json``). Empty → beside ``explicit_onnx_path`` or, when using
  /// ``voices_dir`` discovery, ``voices_json_dir / (<onnx_filename> + ".json")`` if ``voices_json_dir``
  /// is non-empty, else beside the chosen ``*.onnx``.
  std::filesystem::path explicit_onnx_json_path{};
  /// Directory containing ``*.onnx`` for the resolved language. Empty →
  /// ``g2p_options.g2p_root / <lang-subdir> / piper-voices`` unless ``explicit_onnx_path`` is set
  /// (``g2p_root`` defaults to the process cwd when empty).
  std::filesystem::path voices_dir{};
  /// Optional directory containing ``*.onnx.json`` files whose basenames match ``*.onnx`` in
  /// ``voices_dir`` (``<name>.onnx`` → ``<name>.onnx.json`` in this folder). Empty → JSON beside ONNX.
  std::filesystem::path voices_json_dir{};
  /// Locale tag (e.g. ``en_us``, ``de``, ``es``, ``es-ES``, ``ar_msa``).
  std::string lang = "en_us";
  /// ONNX basename (e.g. ``en_US-lessac-medium.onnx``) or stem; empty → default for ``lang``.
  std::string onnx_model{};
  double speed = 1.0;
  MoonshineG2POptions g2p_options{};
  std::vector<std::string> ort_provider_names{};
  /// Match ``piper-tts`` ``SynthesisConfig.normalize_audio`` (scale chunk to full range before clip).
  bool piper_normalize_audio = true;
  /// Match ``SynthesisConfig.volume`` (applied after normalize).
  float piper_output_volume = 1.F;
  /// When set, replaces JSON ``inference.noise_scale`` for ORT (``0`` matches deterministic ``speak.py`` parity tests).
  std::optional<float> piper_noise_scale_override{};
  /// When set, replaces JSON ``inference.noise_w`` for ORT.
  std::optional<float> piper_noise_w_override{};
  /// Optional in-memory TTS assets (keys ``piper/onnx``, ``piper/onnx.json``) from ``MoonshineTTSOptions::files``.
  FileInformationMap tts_asset_files{};
};

class PiperTTS {
 public:
  explicit PiperTTS(const PiperTTSOptions& opt);
  PiperTTS(const PiperTTS&) = delete;
  PiperTTS& operator=(const PiperTTS&) = delete;
  PiperTTS(PiperTTS&&) noexcept;
  PiperTTS& operator=(PiperTTS&&) noexcept;
  ~PiperTTS();

  void set_lang(std::string_view lang_cli);
  void set_speed(double speed);
  double speed() const;
  /// Basename or stem of an ``.onnx`` under ``voices_dir``.
  void set_onnx_model(std::string_view basename_or_stem);

  static constexpr int kSampleRateHz = 24000;

  /// Text → IPA (MoonshineG2P) → Piper phoneme ids → ONNX → mono float waveform at ``kSampleRateHz``.
  std::vector<float> synthesize(std::string_view text);

  /// Run ONNX on an existing Piper phoneme-id sequence (same layout as ``piper.phoneme_ids.phonemes_to_ids``),
  /// then apply ``piper_normalize_audio`` / ``piper_output_volume`` and resample to ``kSampleRateHz``.
  /// For parity with ``speak.py`` ``--piper-inference-backend onnxruntime`` when *ids* match Piper’s eSpeak ids.
  std::vector<float> synthesize_phoneme_ids(const std::vector<int64_t>& phoneme_ids);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Default Piper ONNX + ``*.onnx.json`` paths relative to ``g2p_root`` (under ``<data_subdir>/piper-voices/``).
/// If ``onnx_model_stem`` is non-empty (basename or stem), it selects that ``*.onnx`` instead of the table default
/// (same rules as ``PiperTTSOptions::onnx_model``).
bool piper_default_model_bundle_relative_paths(std::string_view lang_cli, const MoonshineG2POptions& opt,
                                               std::string* onnx_relpath_out, std::string* onnx_json_relpath_out,
                                               std::string_view onnx_model_stem = {});

/// Piper voice stems (no ``.onnx``) with availability (on-disk ``*.onnx`` or in-memory ``piper/onnx``), same
/// resolution as ``PiperTTS``. Known voices are the language default plus any ``*.onnx`` under the resolved
/// voices directory (union, sorted).
std::vector<std::pair<std::string, bool>> piper_list_voices_with_availability(const PiperTTSOptions& opt);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_PIPER_TTS_H
