// CLI: Moonshine G2P + Kokoro or Piper ONNX → WAV.
#include "moonshine-tts.h"
#include "piper-tts.h"
#include "utf8-utils.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--engine kokoro|piper] [--model-root DIR] [--kokoro-dir DIR] [--piper-voices-dir DIR] "
         "[--lang LANG] [--voice ID] [--speed N] [-o out.wav] [--text \"...\"] [TEXT...]\n"
      << "  G2P + layout: default ``builtin_cpp_data_root()`` (``cpp/data``). With ``--model-root DIR``, "
         "G2P uses ``DIR`` like ``cpp/data`` (``ja/``, ``en_us/``, ``zh_hans/``, …).\n"
      << "  kokoro (default): ``kokoro/`` under model root (or ``--kokoro-dir`` override).\n"
      << "  piper: ``<subdir>/piper-voices`` under model root (or ``--piper-voices-dir`` override).\n"
      << "  If --lang is omitted, a simple script heuristic picks ja (kana), ko (Hangul), else en_us.\n"
      << "  Custom layouts: use ``MoonshineTTS`` / ``PiperTTS`` from C++ with ``use_bundled_cpp_g2p_data = false``.\n"
      << "  Export Kokoro voices: python scripts/export_kokoro_voice_for_cpp.py --voices-dir voices/\n"
      << "  Piper voices: python scripts/download_piper_voices_for_g2p.py (copy/sync to cpp/data/*/piper-voices).\n"
      << "  --lang: Default engine Kokoro supports en_us, es, …, fr, ja, zh (and Spanish dialect ids); other "
         "tags use Piper when ``--engine`` is omitted or ``kokoro``.\n"
      << "  --voice: Kokoro voice id (e.g. af_heart) or Piper ONNX stem/basename.\n"
      << "  Default output: out.wav. Default text if none: \"Hello world\".\n";
}

/// When the user does not pass ``--lang``, infer a tag so Japanese text does not run through English G2P / ``af_heart``.
std::optional<std::string> infer_lang_from_text_utf8(const std::string& text) {
  for (size_t i = 0; i < text.size();) {
    char32_t cp = 0;
    size_t adv = 0;
    if (!moonshine_tts::utf8_decode_at(text, i, cp, adv)) {
      break;
    }
    i += adv;
    if (cp >= 0x3040 && cp <= 0x309F) {
      return std::string("ja");
    }  // Hiragana
    if (cp >= 0x30A0 && cp <= 0x30FF) {
      return std::string("ja");
    }  // Katakana
    if (cp >= 0x31F0 && cp <= 0x31FF) {
      return std::string("ja");
    }  // Katakana phonetic extensions
    if (cp >= 0xAC00 && cp <= 0xD7AF) {
      return std::string("ko");
    }  // Hangul syllables
  }
  return std::nullopt;
}

}  // namespace

int main(int argc, char** argv) {
  using moonshine_tts::kokoro_tts_lang_supported;
  using moonshine_tts::MoonshineG2POptions;
  using moonshine_tts::MoonshineTTS;
  using moonshine_tts::MoonshineTTSOptions;
  using moonshine_tts::PiperTTS;
  using moonshine_tts::PiperTTSOptions;
  using moonshine_tts::write_wav_mono_pcm16;

  std::string engine = "kokoro";
  std::filesystem::path model_root;
  std::filesystem::path kokoro_dir;
  std::filesystem::path piper_voices_dir;
  std::string lang = "en_us";
  bool lang_from_cli = false;
  std::string voice;
  std::string out_path = "out.wav";
  std::string text_flag;
  double speed = 1.0;

  std::vector<std::string> positionals;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--engine" && i + 1 < argc) {
      engine = argv[++i];
    } else if (a == "--model-root" && i + 1 < argc) {
      model_root = argv[++i];
    } else if (a == "--kokoro-dir" && i + 1 < argc) {
      kokoro_dir = argv[++i];
    } else if (a == "--piper-voices-dir" && i + 1 < argc) {
      piper_voices_dir = argv[++i];
    } else if (a == "--lang" && i + 1 < argc) {
      lang = argv[++i];
      lang_from_cli = true;
    } else if (a == "--voice" && i + 1 < argc) {
      voice = argv[++i];
    } else if (a == "--speed" && i + 1 < argc) {
      speed = std::strtod(argv[++i], nullptr);
    } else if ((a == "-o" || a == "--output") && i + 1 < argc) {
      out_path = argv[++i];
    } else if (a == "--text" && i + 1 < argc) {
      text_flag = argv[++i];
    } else if (a.rfind("--", 0) == 0) {
      std::cerr << "Unknown option: " << a << '\n';
      usage(argv[0]);
      return 2;
    } else {
      positionals.push_back(a);
    }
  }

  if (engine != "kokoro" && engine != "piper") {
    std::cerr << "Error: --engine must be kokoro or piper.\n";
    return 2;
  }

  if (engine == "kokoro" && kokoro_dir.empty() && model_root.empty()) {
    if (std::filesystem::path p = moonshine_tts::preferred_parent_models_kokoro_dir(); !p.empty()) {
      kokoro_dir = std::move(p);
    }
  }

  std::string text = text_flag;
  if (text.empty()) {
    for (const auto& p : positionals) {
      if (!text.empty()) {
        text += ' ';
      }
      text += p;
    }
  }
  if (text.empty()) {
    text = "Hello world";
  }

  if (!lang_from_cli) {
    if (const auto inferred = infer_lang_from_text_utf8(text)) {
      lang = *inferred;
      std::cerr << "moonshine-tts: inferred --lang " << lang << " from input text "
                   "(set --lang explicitly to override).\n";
    }
  }

  if (engine == "kokoro") {
    MoonshineG2POptions g2p_for_lang;
    if (!model_root.empty()) {
      g2p_for_lang.model_root = model_root;
    }
    if (!kokoro_tts_lang_supported(lang, g2p_for_lang)) {
      engine = "piper";
    }
  }

  try {
    if (engine == "kokoro") {
      MoonshineTTSOptions opt;
      opt.kokoro_dir = std::move(kokoro_dir);
      opt.lang = lang;
      opt.voice = voice;
      opt.speed = speed;
      if (!model_root.empty()) {
        opt.use_bundled_cpp_g2p_data = false;
        opt.g2p_options.model_root = model_root;
      } else {
        opt.use_bundled_cpp_g2p_data = true;
      }
      MoonshineTTS tts(opt);
      const std::vector<float> wav = tts.synthesize(text);
      if (wav.empty()) {
        std::cerr << "Error: empty waveform.\n";
        return 1;
      }
      write_wav_mono_pcm16(out_path, wav);
      std::cout << "Wrote " << out_path << " (" << wav.size() << " samples, "
                << MoonshineTTS::kSampleRateHz << " Hz)\n";
    } else {
      PiperTTSOptions opt;
      opt.voices_dir = std::move(piper_voices_dir);
      opt.lang = lang;
      opt.onnx_model = voice;
      opt.speed = speed;
      if (!model_root.empty()) {
        opt.use_bundled_cpp_g2p_data = false;
        opt.g2p_options.model_root = model_root;
      } else {
        opt.use_bundled_cpp_g2p_data = true;
      }
      PiperTTS tts(opt);
      const std::vector<float> wav = tts.synthesize(text);
      if (wav.empty()) {
        std::cerr << "Error: empty waveform.\n";
        return 1;
      }
      write_wav_mono_pcm16(out_path, wav);
      std::cout << "Wrote " << out_path << " (" << wav.size() << " samples, "
                << PiperTTS::kSampleRateHz << " Hz)\n";
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
