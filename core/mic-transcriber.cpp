#include <alsa/asoundlib.h>
#include <csignal>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include <atomic>
#include <thread>
#include <chrono>

#include "moonshine-cpp.h"

namespace {

std::atomic<bool> g_running(true);

void signalHandler(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    std::cerr << "\nStopping..." << std::endl;
    g_running = false;
  }
}

class TerminalListener : public moonshine::TranscriptEventListener {
 public:
  TerminalListener() : last_line_text_length_(0) {}

  void onLineStarted(const moonshine::LineStarted &) override {
    last_line_text_length_ = 0;
  }

  void onLineTextChanged(const moonshine::LineTextChanged &event) override {
    updateLastTerminalLine(event.line);
  }

  void onLineCompleted(const moonshine::LineCompleted &event) override {
    updateLastTerminalLine(event.line);
    std::cout << std::endl;
  }

 private:
  void updateLastTerminalLine(const moonshine::TranscriptLine &line) {
    std::string speaker_prefix = "";
    if (line.hasSpeakerId) {
      speaker_prefix = "Speaker #" + std::to_string(line.speakerIndex) + ": ";
    }
    std::string new_text = speaker_prefix + line.text;
    
    // Carriage return to overwrite the line
    std::cout << "\r" << new_text << std::flush;
    
    // If new text is shorter, pad with spaces
    if (new_text.length() < last_line_text_length_) {
      size_t diff = last_line_text_length_ - new_text.length();
      std::cout << std::string(diff, ' ') << std::flush;
    }
    
    last_line_text_length_ = new_text.length();
  }

  size_t last_line_text_length_;
};

class MicrophoneCapture {
 public:
  MicrophoneCapture(const std::string &device_name = "default",
                    unsigned int sample_rate = 16000,
                    unsigned int channels = 1,
                    unsigned int frames_per_period = 1024)
      : device_name_(device_name),
        sample_rate_(sample_rate),
        channels_(channels),
        frames_per_period_(frames_per_period),
        pcm_handle_(nullptr) {}

  ~MicrophoneCapture() {
    close();
  }

  bool open() {
    int err;
    
    // Open PCM device for recording
    err = snd_pcm_open(&pcm_handle_, device_name_.c_str(),
                       SND_PCM_STREAM_CAPTURE, 0);
    if (err < 0) {
      std::cerr << "Unable to open PCM device: " << snd_strerror(err)
                << std::endl;
      return false;
    }

    // Allocate hardware parameters object
    snd_pcm_hw_params_t *params;
    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(pcm_handle_, params);

    // Set hardware parameters
    snd_pcm_hw_params_set_access(pcm_handle_, params,
                                  SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(pcm_handle_, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(pcm_handle_, params, channels_);
    snd_pcm_hw_params_set_rate_near(pcm_handle_, params, &sample_rate_, 0);
    snd_pcm_hw_params_set_period_size_near(pcm_handle_, params,
                                            &frames_per_period_, 0);

    // Write parameters to device
    err = snd_pcm_hw_params(pcm_handle_, params);
    if (err < 0) {
      std::cerr << "Unable to set hw parameters: " << snd_strerror(err)
                << std::endl;
      snd_pcm_close(pcm_handle_);
      pcm_handle_ = nullptr;
      return false;
    }

    // Get actual parameters
    snd_pcm_hw_params_get_rate(params, &sample_rate_, 0);
    snd_pcm_hw_params_get_period_size(params, &frames_per_period_, 0);

    std::cerr << "Audio capture initialized: " << sample_rate_ << " Hz, "
              << channels_ << " channel(s), " << frames_per_period_
              << " frames per period" << std::endl;

    return true;
  }

  void close() {
    if (pcm_handle_) {
      snd_pcm_drain(pcm_handle_);
      snd_pcm_close(pcm_handle_);
      pcm_handle_ = nullptr;
    }
  }

  bool read(std::vector<float> &audio_buffer) {
    if (!pcm_handle_) {
      return false;
    }

    // Allocate buffer for int16 samples
    std::vector<int16_t> buffer(frames_per_period_ * channels_);
    
    int err = snd_pcm_readi(pcm_handle_, buffer.data(), frames_per_period_);
    
    if (err == -EPIPE) {
      // Overrun occurred
      std::cerr << "Overrun occurred" << std::endl;
      snd_pcm_prepare(pcm_handle_);
      return false;
    } else if (err < 0) {
      std::cerr << "Error reading from PCM device: " << snd_strerror(err)
                << std::endl;
      return false;
    } else if (err != (int)frames_per_period_) {
      std::cerr << "Short read, expected " << frames_per_period_ << " got "
                << err << std::endl;
    }

    // Convert int16 to float32 [-1.0, 1.0]
    audio_buffer.resize(err * channels_);
    for (int i = 0; i < err * (int)channels_; ++i) {
      audio_buffer[i] = buffer[i] / 32768.0f;
    }

    return true;
  }

  unsigned int getSampleRate() const { return sample_rate_; }

 private:
  std::string device_name_;
  unsigned int sample_rate_;
  unsigned int channels_;
  snd_pcm_uframes_t frames_per_period_;
  snd_pcm_t *pcm_handle_;
};

}  // namespace

int main(int argc, char *argv[]) {
  std::string model_path = "";
  moonshine::ModelArch model_arch = moonshine::ModelArch::TINY;
  std::string device_name = "default";
  unsigned int sample_rate = 16000;
  float update_interval = 0.5f;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-m" || arg == "--model-path") && i + 1 < argc) {
      model_path = argv[++i];
    } else if ((arg == "-a" || arg == "--model-arch") && i + 1 < argc) {
      model_arch = static_cast<moonshine::ModelArch>(std::stoi(argv[++i]));
    } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
      device_name = argv[++i];
    } else if ((arg == "-r" || arg == "--sample-rate") && i + 1 < argc) {
      sample_rate = std::stoi(argv[++i]);
    } else if ((arg == "-u" || arg == "--update-interval") && i + 1 < argc) {
      update_interval = std::stof(argv[++i]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                << "Options:\n"
                << "  -m, --model-path PATH       Path to model directory (required)\n"
                << "  -a, --model-arch ARCH       Model architecture (0=TINY, 1=BASE, 2=SMALL, 3=MEDIUM)\n"
                << "  -d, --device DEVICE         ALSA device name (default: 'default')\n"
                << "  -r, --sample-rate RATE      Sample rate in Hz (default: 16000)\n"
                << "  -u, --update-interval SEC   Update interval in seconds (default: 0.5)\n"
                << "  -h, --help                  Show this help message\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      std::cerr << "Use --help for usage information" << std::endl;
      return 1;
    }
  }

  if (model_path.empty()) {
    std::cerr << "Error: Model path is required (use -m or --model-path)"
              << std::endl;
    std::cerr << "Use --help for usage information" << std::endl;
    return 1;
  }

  // Set up signal handlers
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  try {
    // Initialize microphone
    MicrophoneCapture mic(device_name, sample_rate);
    if (!mic.open()) {
      std::cerr << "Failed to open microphone" << std::endl;
      return 1;
    }

    // Initialize transcriber
    std::cerr << "Loading model from: " << model_path << std::endl;
    moonshine::Transcriber transcriber(model_path, model_arch);
    
    // Create stream with update interval
    moonshine::Stream stream = transcriber.createStream(update_interval);
    
    // Add listener
    TerminalListener listener;
    stream.addListener(&listener);

    std::cerr << "Listening to the microphone, press Ctrl+C to stop..."
              << std::endl;

    // Start transcription
    stream.start();

    // Main capture loop
    std::vector<float> audio_buffer;
    while (g_running) {
      if (mic.read(audio_buffer)) {
        stream.addAudio(audio_buffer, mic.getSampleRate());
      } else {
        // Small delay on error to avoid busy loop
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    }

    // Stop transcription
    stream.stop();
    mic.close();

    std::cerr << "Transcription stopped." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}