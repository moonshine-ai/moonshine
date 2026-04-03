// Dev / CI: Piper ONNX from a JSON list of int64 phoneme ids (parity with ``speak.py`` ORT path).
#include "moonshine-tts.h"
#include "piper-tts.h"

#include <nlohmann/json.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr << "Usage: " << argv0
            << " --onnx PATH.onnx --ids-json PATH.json [--onnx-json PATH.onnx.json] [--lang en_us] [--speed N] "
               "[--noise-scale F] [--noise-w F] [-o out.wav]\n"
            << "  JSON file: array of int64 phoneme ids (Piper ``phonemes_to_ids`` layout).\n"
            << "  Optional --noise-scale / --noise-w override model JSON (use 0 0 for deterministic parity vs "
               "``speak.py`` tests).\n"
            << "  Writes 24 kHz mono WAV (same as ``moonshine_tts`` / ``speak.py`` Piper output).\n";
}

}  // namespace

int main(int argc, char** argv) {
  using moonshine_tts::PiperTTS;
  using moonshine_tts::PiperTTSOptions;
  using moonshine_tts::write_wav_mono_pcm16;

  std::filesystem::path onnx_path;
  std::filesystem::path onnx_config_json;
  std::filesystem::path ids_json;
  std::filesystem::path out_path = "out.wav";
  std::string lang = "en_us";
  double speed = 1.0;
  std::optional<float> noise_scale_override;
  std::optional<float> noise_w_override;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--onnx" && i + 1 < argc) {
      onnx_path = argv[++i];
    } else if (a == "--onnx-json" && i + 1 < argc) {
      onnx_config_json = argv[++i];
    } else if (a == "--ids-json" && i + 1 < argc) {
      ids_json = argv[++i];
    } else if (a == "--lang" && i + 1 < argc) {
      lang = argv[++i];
    } else if (a == "--speed" && i + 1 < argc) {
      speed = std::strtod(argv[++i], nullptr);
    } else if (a == "--noise-scale" && i + 1 < argc) {
      noise_scale_override = static_cast<float>(std::strtod(argv[++i], nullptr));
    } else if (a == "--noise-w" && i + 1 < argc) {
      noise_w_override = static_cast<float>(std::strtod(argv[++i], nullptr));
    } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
      out_path = argv[++i];
    } else {
      std::cerr << "Unknown or incomplete option: " << a << '\n';
      usage(argv[0]);
      return 2;
    }
  }

  if (onnx_path.empty() || ids_json.empty()) {
    usage(argv[0]);
    return 2;
  }

  std::vector<int64_t> ids;
  try {
    std::ifstream jf(ids_json);
    if (!jf) {
      throw std::runtime_error("cannot open ids-json");
    }
    nlohmann::json j;
    jf >> j;
    if (!j.is_array()) {
      throw std::runtime_error("ids-json must be a JSON array");
    }
    ids.reserve(j.size());
    for (const auto& el : j) {
      ids.push_back(el.get<int64_t>());
    }
  } catch (const std::exception& e) {
    std::cerr << "Error reading ids-json: " << e.what() << '\n';
    return 1;
  }

  PiperTTSOptions opt;
  opt.lang = lang;
  opt.explicit_onnx_path = onnx_path;
  if (!onnx_config_json.empty()) {
    opt.explicit_onnx_json_path = onnx_config_json;
  }
  opt.onnx_model = onnx_path.filename().string();
  opt.speed = speed;
  opt.use_bundled_cpp_g2p_data = true;
  opt.piper_normalize_audio = true;
  opt.piper_output_volume = 1.F;
  opt.piper_noise_scale_override = noise_scale_override;
  opt.piper_noise_w_override = noise_w_override;

  try {
    PiperTTS tts(opt);
    const std::vector<float> wav = tts.synthesize_phoneme_ids(ids);
    if (wav.empty()) {
      std::cerr << "Error: empty waveform.\n";
      return 1;
    }
    write_wav_mono_pcm16(out_path, wav);
    std::cout << "Wrote " << out_path << " (" << wav.size() << " samples, " << PiperTTS::kSampleRateHz << " Hz)\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
