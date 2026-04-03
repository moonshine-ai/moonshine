#ifndef MOONSHINE_TTS_MOONSHINE_TTS_H
#define MOONSHINE_TTS_MOONSHINE_TTS_H

#include "builtin-cpp-data-root.h"
#include "moonshine-g2p-options.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

/** Moonshine Text to Speech C++ library: shared grapheme-to-phoneme (\c MoonshineG2P) plus multiple synthesis
 *  backends—open-source Kokoro and Piper (ONNX), and Moonshine AI in-house acoustic models where shipped.
 *  This header focuses on the Kokoro-backed \c MoonshineTTS API; \c PiperTTS is declared in \c piper-tts.h. */
namespace moonshine_tts {

/// Default Kokoro bundle: ``data/kokoro`` at the repository root (``config.json``, ``model.onnx``, ``voices/``).
std::filesystem::path builtin_kokoro_bundle_dir();

/// When ``moonshine-tts`` lives at ``<repo>/moonshine-tts/`` and ``<repo>/models/kokoro/model.onnx`` exists
/// and is strictly larger than ``moonshine-tts/data/kokoro/model.onnx``, returns ``<repo>/models/kokoro``.
/// Otherwise returns an empty path. Monorepos often keep the full Kokoro graph under ``models/kokoro`` while
/// the submodule bundles a smaller ONNX; the speak CLI uses this to default to the higher-fidelity model.
std::filesystem::path preferred_parent_models_kokoro_dir();

/// Configuration for the Kokoro-backed ``MoonshineTTS`` path (bundle layout, G2P, ONNX Runtime providers).
struct MoonshineTTSOptions {
  /// Directory with ``config.json``, ``model.onnx``, and ``voices/*.kokorovoice`` (see
  /// ``scripts/export_kokoro_voice_for_cpp.py``). Empty → ``builtin_kokoro_bundle_dir()``.
  std::filesystem::path kokoro_dir{};
  /// Locale tag: built-ins like ``en_us``, ``es``, ``ja``, or any Spanish rule id (``es_mx``, ``es-AR``, …).
  std::string lang = "en_us";
  /// Kokoro voice id (e.g. ``af_heart``). Empty → default for ``lang``.
  std::string voice{};
  double speed = 1.0;
  /// When true (default), ``g2p_options.model_root`` is replaced with ``builtin_cpp_data_root()`` (the
  /// ``cpp/data`` tree: ``ja/``, ``en_us/``, ``zh_hans/``, …). Set false to use your own ``model_root``.
  bool use_bundled_cpp_g2p_data = true;
  MoonshineG2POptions g2p_options{};
  /// ONNX Runtime provider names (e.g. ``CUDAExecutionProvider``). Empty → CPU only.
  std::vector<std::string> ort_provider_names{};
};

/// True when ``MoonshineTTS`` / the Kokoro path accepts ``lang_cli`` (built-in Kokoro locales plus Spanish rule
/// dialects). Used by the ``moonshine_tts`` CLI to choose Kokoro vs ``PiperTTS`` or other backends.
bool kokoro_tts_lang_supported(std::string_view lang_cli, const MoonshineG2POptions& g2p_opt = {});

/// Primary class for **Kokoro** synthesis in the Moonshine Text to Speech library.
///
/// This type composes ``MoonshineG2P`` with the Kokoro ONNX model and voices. The library as a whole also
/// supports **Piper** (``PiperTTS``) and **Moonshine AI** in-house models on other code paths; see the
/// namespace overview above. Options and bundled data paths are set through ``MoonshineTTSOptions``.
class MoonshineTTS {
 public:
  explicit MoonshineTTS(const MoonshineTTSOptions& opt);
  MoonshineTTS(const MoonshineTTS&) = delete;
  MoonshineTTS& operator=(const MoonshineTTS&) = delete;
  MoonshineTTS(MoonshineTTS&&) noexcept;
  MoonshineTTS& operator=(MoonshineTTS&&) noexcept;
  ~MoonshineTTS();

  static constexpr int kSampleRateHz = 24000;

  /// Kokoro path: text → IPA (Moonshine G2P) → Kokoro phoneme string → ONNX → mono float waveform at
  /// ``kSampleRateHz``. (Piper and in-house Moonshine backends use their respective APIs.)
  std::vector<float> synthesize(std::string_view text);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

/// Writes mono 16-bit PCM WAV at ``MoonshineTTS::kSampleRateHz`` (samples clipped to [-1, 1]).
void write_wav_mono_pcm16(const std::filesystem::path& path, const std::vector<float>& samples);

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_MOONSHINE_TTS_H
