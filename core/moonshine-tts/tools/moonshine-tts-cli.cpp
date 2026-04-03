// CLI: Moonshine G2P + Kokoro or Piper ONNX → WAV (via MoonshineTTS).
#include "moonshine-tts.h"
#include "utf8-utils.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

void usage(const char* argv0) {
  std::cerr
      << "Usage: " << argv0
      << " [--engine kokoro|piper|auto] [--model-root DIR] [--kokoro-dir DIR] "
         "[--piper-onnx PATH.onnx] [--piper-onnx-json PATH.onnx.json] [--piper-voices-dir DIR] "
         "[--piper-voices-json-dir DIR] "
         "[--lang LANG] [--voice ID] [--speed N] [-o out.wav] [--text \"...\"] [TEXT...]\n"
      << "  G2P + layout: default ``builtin_cpp_data_root()`` (``cpp/data``). With ``--model-root DIR``, "
         "G2P uses ``DIR`` like ``cpp/data`` (``ja/``, ``en_us/``, ``zh_hans/``, …).\n"
      << "  engine auto (default): Kokoro when the language is supported, otherwise Piper.\n"
      << "  kokoro: ``kokoro/`` under model root (or ``--kokoro-dir`` override).\n"
      << "  piper: ``<subdir>/piper-voices`` under model root (or ``--piper-voices-dir`` / "
         "``--piper-voices-json-dir`` for split ONNX vs JSON trees, or ``--piper-onnx`` + "
         "``--piper-onnx-json`` for explicit files).\n"
      << "  If --lang is omitted, a simple script heuristic picks ja (kana), ko (Hangul), else en_us.\n"
      << "  Custom layouts: use ``MoonshineTTS`` from C++ with ``use_bundled_cpp_g2p_data = false``.\n"
      << "  Export Kokoro voices: python scripts/export_kokoro_voice_for_cpp.py --voices-dir voices/\n"
      << "  Piper voices: python scripts/download_piper_voices_for_g2p.py (copy/sync to cpp/data/*/piper-voices).\n"
      << "  --lang: Kokoro supports en_us, es, …, fr, ja, zh (and Spanish dialect ids); other tags use Piper "
         "when engine is auto or piper.\n"
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

std::string ascii_lowercase_copy(std::string_view s) {
  std::string o(s);
  for (char& c : o) {
    if (c >= 'A' && c <= 'Z') {
      c = static_cast<char>(c - 'A' + 'a');
    }
  }
  return o;
}

}  // namespace

int main(int argc, char** argv) {
  using moonshine_tts::kTtsKokoroConfigJsonKey;
  using moonshine_tts::kTtsKokoroModelOnnxKey;
  using moonshine_tts::MoonshineTTS;
  using moonshine_tts::MoonshineTTSOptions;
  using moonshine_tts::preferred_parent_models_kokoro_dir;
  using moonshine_tts::write_wav_mono_pcm16;

  std::vector<std::pair<std::string, std::string>> pairs;
  std::vector<std::string> positionals;
  std::string text_flag;

  for (int i = 1; i < argc;) {
    const std::string a = argv[i];
    if (a == "-h" || a == "--help") {
      usage(argv[0]);
      return 0;
    }
    if (a == "--text" && i + 1 < argc) {
      text_flag = argv[i + 1];
      i += 2;
      continue;
    }
    if (a == "-o" && i + 1 < argc) {
      pairs.emplace_back("output", argv[i + 1]);
      i += 2;
      continue;
    }
    if (a.rfind("--", 0) == 0) {
      const std::string key = a.substr(2);
      if (key.empty()) {
        std::cerr << "Empty option name.\n";
        usage(argv[0]);
        return 2;
      }
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --" << key << '\n';
        usage(argv[0]);
        return 2;
      }
      pairs.emplace_back(key, argv[i + 1]);
      i += 2;
      continue;
    }
    positionals.push_back(a);
    ++i;
  }

  MoonshineTTSOptions opt;
  std::string lang = "en_us";
  bool lang_set = false;
  try {
    opt.parse_options(pairs, &lang, &lang_set);
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    usage(argv[0]);
    return 2;
  }

  if (opt.use_bundled_cpp_g2p_data && opt.g2p_options.g2p_root.empty()) {
    const std::filesystem::path default_kokoro_model{kTtsKokoroModelOnnxKey};
    if (opt.tts_relative_path(kTtsKokoroModelOnnxKey) == default_kokoro_model) {
      std::string eng =
          ascii_lowercase_copy(moonshine_tts::trim_ascii_ws_copy(opt.vocoder_engine));
      if (eng.empty()) {
        eng = "auto";
      }
      if (eng == "kokoro" || eng == "auto") {
        if (std::filesystem::path p = preferred_parent_models_kokoro_dir(); !p.empty()) {
          opt.files.set_path(kTtsKokoroModelOnnxKey, p / "model.onnx");
          opt.files.set_path(kTtsKokoroConfigJsonKey, p / "config.json");
        }
      }
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

  if (!lang_set) {
    if (const auto inferred = infer_lang_from_text_utf8(text)) {
      lang = *inferred;
      std::cerr << "moonshine-tts: inferred --lang " << lang << " from input text "
                   "(set --lang explicitly to override).\n";
    }
  }

  try {
    MoonshineTTS tts(lang, opt);
    const std::vector<float> wav = tts.synthesize(text);
    if (wav.empty()) {
      std::cerr << "Error: empty waveform.\n";
      return 1;
    }
    write_wav_mono_pcm16(opt.output_path, wav);
    std::cout << "Wrote " << opt.output_path << " (" << wav.size() << " samples, "
              << MoonshineTTS::kSampleRateHz << " Hz)\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
