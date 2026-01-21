#include <iostream>
#include <string>
#include <vector>

#include "moonshine-cpp.h"

namespace {
class AudioProducer {
 public:
  AudioProducer(std::string wav_path, float chunk_duration_seconds = 0.0214f)
      : current_index_(0) {
    loadWavData(wav_path);
    chunk_size_ = static_cast<size_t>(chunk_duration_seconds * sample_rate_);
  }
  bool getNextAudio(std::vector<float> &out_audio_data) {
    if (current_index_ >= audio_data_.size()) {
      return false;
    }
    size_t end_index =
        std::min(current_index_ + chunk_size_, audio_data_.size());
    out_audio_data.resize(end_index - current_index_);
    out_audio_data.assign(audio_data_.begin() + current_index_,
                          audio_data_.begin() + end_index);
    current_index_ = end_index;
    return true;
  }
  int32_t sample_rate() const { return sample_rate_; }
  void loadWavData(std::string wav_path);

 private:
  size_t chunk_size_;
  int32_t sample_rate_;
  size_t current_index_;
  std::vector<float> audio_data_;
};

class PrintLinesListener : public moonshine::TranscriptEventListener {
 public:
  void onLineStarted(const moonshine::LineStarted &event) override {
    std::cout << "Line started: " << event.line.text << std::endl;
  }
  void onLineTextChanged(const moonshine::LineTextChanged &event) override {
    std::cout << "Line text changed: " << event.line.text << std::endl;
  }
  void onLineCompleted(const moonshine::LineCompleted &event) override {
    std::cout << "Line completed: " << event.line.text << std::endl;
  }
};
}  // namespace

int main(int argc, char *argv[]) {
  std::string model_path = "../../test-assets/tiny-en";
  moonshine::ModelArch model_arch = moonshine::ModelArch::TINY;
  std::string wav_path = "../../test-assets/two_cities.wav";
  float transcription_interval_seconds = 0.481f;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-m" || arg == "--model-path") {
      model_path = argv[++i];
    } else if (arg == "-a" || arg == "--model-arch") {
      model_arch = static_cast<moonshine::ModelArch>(std::stoi(argv[++i]));
    } else if (arg == "-w" || arg == "--wav-path") {
      wav_path = argv[++i];
    } else if (arg == "-t" || arg == "--transcription-interval") {
      transcription_interval_seconds = std::stof(argv[++i]);
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  AudioProducer audio_producer(wav_path);
  moonshine::Transcriber transcriber(model_path, model_arch);
  PrintLinesListener listener;
  transcriber.addListener(&listener);
  transcriber.start();
  std::vector<float> chunk_audio_data;
  const int32_t samples_between_transcriptions = static_cast<int32_t>(
      transcription_interval_seconds * audio_producer.sample_rate());
  int32_t samples_since_last_transcription = 0;
  while (audio_producer.getNextAudio(chunk_audio_data)) {
    transcriber.addAudio(chunk_audio_data, audio_producer.sample_rate());
    samples_since_last_transcription += chunk_audio_data.size();
    if (samples_since_last_transcription < samples_between_transcriptions) {
      continue;
    }
    samples_since_last_transcription = 0;
    transcriber.updateTranscription();
  }
  transcriber.stop();
  return 0;
}

void AudioProducer::loadWavData(std::string wav_path) {
  audio_data_.clear();
  sample_rate_ = 0;

  // Open the file in binary mode
  FILE *file = std::fopen(wav_path.c_str(), "rb");
  if (!file) {
    std::perror("Failed to open WAV file");
    return;
  }

  // Read the RIFF header
  char riff_header[4];
  if (std::fread(riff_header, 1, 4, file) != 4 ||
      std::strncmp(riff_header, "RIFF", 4) != 0) {
    std::fclose(file);
    throw std::runtime_error("Not a RIFF file");
  }

  // Skip chunk size and check WAVE
  std::fseek(file, 4, SEEK_CUR);
  char wave_header[4];
  if (std::fread(wave_header, 1, 4, file) != 4 ||
      std::strncmp(wave_header, "WAVE", 4) != 0) {
    std::fclose(file);
    throw std::runtime_error("Not a WAVE file");
  }

  // Find the "fmt " chunk
  char chunk_id[4];
  uint32_t chunk_size = 0;
  bool found_fmt = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
      found_fmt = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_fmt) {
    std::fclose(file);
    throw std::runtime_error("No fmt chunk found");
  }

  // Read fmt chunk
  uint16_t audio_format = 0, num_channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0, byte_rate = 0;
  uint16_t block_align = 0;
  if (chunk_size < 16) {
    std::fclose(file);
    throw std::runtime_error("fmt chunk too small");
  }
  std::fread(&audio_format, sizeof(uint16_t), 1, file);
  std::fread(&num_channels, sizeof(uint16_t), 1, file);
  std::fread(&sample_rate, sizeof(uint32_t), 1, file);
  std::fread(&byte_rate, sizeof(uint32_t), 1, file);
  std::fread(&block_align, sizeof(uint16_t), 1, file);
  std::fread(&bits_per_sample, sizeof(uint16_t), 1, file);
  // Skip any extra fmt bytes
  if (chunk_size > 16) std::fseek(file, chunk_size - 16, SEEK_CUR);

  if (audio_format != 1 || bits_per_sample != 16) {
    std::fclose(file);
    throw std::runtime_error("Only 16-bit PCM WAV files are supported");
  }

  // Find the "data" chunk
  bool found_data = false;
  while (std::fread(chunk_id, 1, 4, file) == 4) {
    if (std::fread(&chunk_size, 4, 1, file) != 1) break;
    if (std::strncmp(chunk_id, "data", 4) == 0) {
      found_data = true;
      break;
    }
    std::fseek(file, chunk_size, SEEK_CUR);
  }
  if (!found_data) {
    std::fclose(file);
    throw std::runtime_error("No data chunk found");
  }

  // Read PCM data
  size_t num_samples = chunk_size / (bits_per_sample / 8);
  if (num_samples == 0) {
    std::fclose(file);
    throw std::runtime_error("No samples found");
  }
  audio_data_.resize(num_samples);
  for (size_t i = 0; i < num_samples; ++i) {
    int16_t sample = 0;
    if (std::fread(&sample, sizeof(int16_t), 1, file) != 1) {
      num_samples = i;
      break;
    }
    audio_data_[i] = static_cast<float>(sample) / 32768.0f;
  }
  std::fclose(file);
  sample_rate_ = sample_rate;
  return;
}
