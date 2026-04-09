#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "moonshine-cpp.h"

namespace {
void writeWav(const std::string &path, const std::vector<float> &samples,
              int32_t sampleRate) {
  FILE *file = std::fopen(path.c_str(), "wb");
  if (!file) {
    throw std::runtime_error("Failed to open output file: " + path);
  }

  const uint16_t num_channels = 1;
  const uint16_t bits_per_sample = 16;
  const uint32_t byte_rate = sampleRate * num_channels * bits_per_sample / 8;
  const uint16_t block_align = num_channels * bits_per_sample / 8;
  const uint32_t data_size =
      static_cast<uint32_t>(samples.size()) * bits_per_sample / 8;
  const uint32_t chunk_size = 36 + data_size;

  // RIFF header
  std::fwrite("RIFF", 1, 4, file);
  std::fwrite(&chunk_size, 4, 1, file);
  std::fwrite("WAVE", 1, 4, file);

  // fmt chunk
  std::fwrite("fmt ", 1, 4, file);
  const uint32_t fmt_size = 16;
  const uint16_t audio_format = 1;  // PCM
  std::fwrite(&fmt_size, 4, 1, file);
  std::fwrite(&audio_format, 2, 1, file);
  std::fwrite(&num_channels, 2, 1, file);
  std::fwrite(&sampleRate, 4, 1, file);
  std::fwrite(&byte_rate, 4, 1, file);
  std::fwrite(&block_align, 2, 1, file);
  std::fwrite(&bits_per_sample, 2, 1, file);

  // data chunk
  std::fwrite("data", 1, 4, file);
  std::fwrite(&data_size, 4, 1, file);
  for (size_t i = 0; i < samples.size(); ++i) {
    float clamped = samples[i];
    if (clamped > 1.0f) clamped = 1.0f;
    if (clamped < -1.0f) clamped = -1.0f;
    int16_t sample = static_cast<int16_t>(clamped * 32767.0f);
    std::fwrite(&sample, sizeof(int16_t), 1, file);
  }

  std::fclose(file);
}
}  // namespace

int main(int argc, char *argv[]) {
  std::string asset_root = "../../core/moonshine-tts/data";
  std::string language = "en_us";
  std::string voice = "";
  std::string text = "Hello! This is a test of the Moonshine text to speech.";
  std::string output_path = "output.wav";
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-r" || arg == "--asset-root") {
      asset_root = argv[++i];
    } else if (arg == "-l" || arg == "--language") {
      language = argv[++i];
    } else if (arg == "-v" || arg == "--voice") {
      voice = argv[++i];
    } else if (arg == "-t" || arg == "--text") {
      text = argv[++i];
    } else if (arg == "-o" || arg == "--output") {
      output_path = argv[++i];
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::cerr << "Usage: " << argv[0]
                << " [-r asset_root] [-l language] [-v voice] [-t text] "
                   "[-o output.wav]"
                << std::endl;
      return 1;
    }
  }

  try {
    std::vector<moonshine_option_t> options = {
        {"g2p_root", asset_root.c_str()},
    };
    if (!voice.empty()) {
      options.push_back({"voice", voice.c_str()});
    }

    std::cout << "Creating TTS synthesizer for language '" << language << "'"
              << std::endl;
    moonshine::TextToSpeech tts(language, options);

    moonshine::GraphemeToPhonemizer g2p(language,
                                        {
                                            {"g2p_root", asset_root.c_str()},
                                        });
    std::string ipa = g2p.toIpa(text);
    std::cout << "IPA: " << ipa << " (" << ipa.size() << " characters)"
              << std::endl;

    std::cout << "Synthesizing: \"" << text << "\"" << std::endl;
    moonshine::TtsSynthesisResult result = tts.synthesize(text);

    std::cout << "Got " << result.samples.size() << " samples at "
              << result.sampleRateHz << " Hz ("
              << (result.samples.size() /
                  static_cast<double>(result.sampleRateHz))
              << " seconds)" << std::endl;

    writeWav(output_path, result.samples, result.sampleRateHz);
    std::cout << "Written to " << output_path << std::endl;
  } catch (const moonshine::MoonshineException &e) {
    std::cerr << "Moonshine error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
