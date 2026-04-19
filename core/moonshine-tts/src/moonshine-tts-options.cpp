#include "moonshine-tts.h"

#include "string-utils.h"

#include <stdexcept>
#include <string_view>

namespace moonshine_tts {

MoonshineTTSOptions::MoonshineTTSOptions() {
  files.set_path(kTtsKokoroModelOnnxKey, std::filesystem::path{kTtsKokoroModelOnnxKey});
  files.set_path(kTtsKokoroConfigJsonKey, std::filesystem::path{kTtsKokoroConfigJsonKey});
}

void MoonshineTTSOptions::apply_voice_engine_prefix() {
  voice = trim(voice);
  if (voice.empty()) {
    return;
  }
  const std::string low = to_lowercase(voice);
  static constexpr std::string_view k_k = "kokoro_";
  static constexpr std::string_view k_p = "piper_";
  if (low.size() >= k_k.size() && low.compare(0, k_k.size(), k_k) == 0) {
    vocoder_engine = "kokoro";
    voice = trim(voice.substr(k_k.size()));
  } else if (low.size() >= k_p.size() && low.compare(0, k_p.size(), k_p) == 0) {
    vocoder_engine = "piper";
    voice = trim(voice.substr(k_p.size()));
  }
}

std::filesystem::path MoonshineTTSOptions::tts_relative_path(std::string_view canonical_key) const {
  const std::string k(canonical_key);
  const auto it = files.entries.find(k);
  if (it == files.entries.end()) {
    return std::filesystem::path(canonical_key);
  }
  return it->second.path;
}

void MoonshineTTSOptions::parse_options(
    const std::vector<std::pair<std::string, std::string>>& options, std::string* cli_language,
    bool* language_was_set) {
  std::vector<std::pair<std::string, std::string>> g2p_pairs;
  g2p_pairs.reserve(options.size());

  for (const auto& entry : options) {
    const std::string& name = entry.first;
    const std::string& value = entry.second;
    const std::string key = replace_all(to_lowercase(name), "-", "_");

    if (key == "tts_root" || key == "path_root" || key == "model_root") {
      const std::string t = trim(value);
      if (!t.empty()) {
        g2p_options.g2p_root = std::filesystem::path(t);
      }
    } else if (key == "g2p_root") {
      g2p_options.g2p_root = std::filesystem::path(trim(value));
    } else if (key == "use_bundled_cpp_g2p_data" || key == "bundle_g2p_data") {
      // Deprecated: cwd-based asset discovery was removed; value ignored.
    } else if (key == "lang" || key == "language") {
      if (!cli_language) {
        throw std::runtime_error(
            "MoonshineTTSOptions: option \"" + name +
            "\" is invalid without a language output pointer; use parse_options(options, &lang, "
            "&lang_set) or pass the language to MoonshineTTS(language, options).");
      }
      *cli_language = trim(value);
      if (language_was_set) {
        *language_was_set = true;
      }
    } else if (key == "voice") {
      voice = trim(value);
    } else if (key == "speed") {
      speed = static_cast<double>(float_from_string(value.c_str()));
    } else if (key == "kokoro_dir") {
      const std::string t = trim(value);
      if (!t.empty()) {
        const std::filesystem::path d(t);
        files.set_path(kTtsKokoroModelOnnxKey, d / "model.onnx");
        files.set_path(kTtsKokoroConfigJsonKey, d / "config.json");
      }
    } else if (key == "kokoro_model" || key == "kokoro_model_onnx") {
      const std::string t = trim(value);
      if (t.empty()) {
        files.erase_key(std::string(kTtsKokoroModelOnnxKey));
      } else {
        files.set_path(kTtsKokoroModelOnnxKey, std::filesystem::path(t));
      }
    } else if (key == "kokoro_config" || key == "kokoro_config_json") {
      const std::string t = trim(value);
      if (t.empty()) {
        files.erase_key(std::string(kTtsKokoroConfigJsonKey));
      } else {
        files.set_path(kTtsKokoroConfigJsonKey, std::filesystem::path(t));
      }
    } else if (key == "piper_onnx" || key == "piper_model_onnx") {
      const std::string t = trim(value);
      if (t.empty()) {
        files.erase_key(std::string(kTtsPiperOnnxKey));
      } else {
        files.set_path(kTtsPiperOnnxKey, std::filesystem::path(t));
      }
    } else if (key == "piper_onnx_json" || key == "piper_model_json" || key == "piper_onnx_config") {
      const std::string t = trim(value);
      if (t.empty()) {
        files.erase_key(std::string(kTtsPiperOnnxJsonKey));
      } else {
        files.set_path(kTtsPiperOnnxJsonKey, std::filesystem::path(t));
      }
    } else if (key == "piper_voices_dir" || key == "voices_dir") {
      const std::string t = trim(value);
      if (t.empty()) {
        files.erase_key(std::string(kTtsPiperVoicesKey));
      } else {
        files.set_path(kTtsPiperVoicesKey, std::filesystem::path(t));
      }
    } else if (key == "piper_voices_json_dir" || key == "voices_json_dir") {
      const std::string t = trim(value);
      if (t.empty()) {
        files.erase_key(std::string(kTtsPiperVoicesJsonKey));
      } else {
        files.set_path(kTtsPiperVoicesJsonKey, std::filesystem::path(t));
      }
    } else if (key == "engine" || key == "vocoder_engine") {
      // Deprecated: vocoder is selected via voice prefix (kokoro_* / piper_*). Ignored for compatibility.
      (void)value;
    } else if (key == "output" || key == "o") {
      output_path = std::filesystem::path(trim(value));
    } else if (key == "piper_normalize_audio") {
      piper_normalize_audio = bool_from_string(value.c_str());
    } else if (key == "piper_output_volume") {
      piper_output_volume = float_from_string(value.c_str());
    } else if (key == "piper_noise_scale" || key == "piper_noise_scale_override") {
      const std::string t = trim(value);
      piper_noise_scale_override =
          t.empty() ? std::nullopt : std::optional<float>(float_from_string(t.c_str()));
    } else if (key == "piper_noise_w" || key == "piper_noise_w_override") {
      const std::string t = trim(value);
      piper_noise_w_override =
          t.empty() ? std::nullopt : std::optional<float>(float_from_string(t.c_str()));
    } else if (key == "log_profiling") {
      log_profiling = bool_from_string(value.c_str());
    } else {
      g2p_pairs.push_back(entry);
    }
  }

  g2p_options.parse_options(g2p_pairs);
  apply_voice_engine_prefix();
}

}  // namespace moonshine_tts
