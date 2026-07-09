#include <atomic>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "moonshine-cpp.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "mmdevapi.lib")

// Helper class to manage COM initialization
class COMInitializer {
 public:
  COMInitializer() {
    HRESULT hr = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (FAILED(hr)) {
      throw std::runtime_error("Failed to initialize COM");
    }
  }
  ~COMInitializer() { CoUninitialize(); }
};

// Listener class to handle transcription events
class TranscriptionListener : public moonshine::TranscriptEventListener {
 public:
  void onLineStarted(const moonshine::LineStarted &event) override {
    std::lock_guard<std::mutex> lock(output_mutex_);
    std::cout << "\r[Started] " << event.line.text << std::flush;
    last_line_length_ = event.line.text.length();
  }

  void onLineTextChanged(const moonshine::LineTextChanged &event) override {
    std::lock_guard<std::mutex> lock(output_mutex_);
    // Clear the previous line and print the new one
    std::cout << "\r" << std::string(last_line_length_ * 2, ' ') << "\r";
    std::cout << event.line.text << std::flush;
    last_line_length_ = event.line.text.length();
  }

  void onLineCompleted(const moonshine::LineCompleted &event) override {
    std::lock_guard<std::mutex> lock(output_mutex_);
    // Clear the previous line and print the completed line
    std::cout << "\r" << std::string(last_line_length_, ' ') << "\r";
    std::cout << event.line.text << std::endl;
    last_line_length_ = 0;
  }

  void onError(const moonshine::Error &event) override {
    std::lock_guard<std::mutex> lock(output_mutex_);
    std::cerr << "\nError: " << event.errorMessage << std::endl;
  }

 private:
  std::mutex output_mutex_;
  size_t last_line_length_ = 0;
};

// WASAPI microphone capture class
class MicrophoneCapture {
 public:
  MicrophoneCapture() : is_capturing_(false), sample_rate_(16000) {}

  ~MicrophoneCapture() {
    Stop();
    if (capture_client_) {
      capture_client_->Release();
      capture_client_ = nullptr;
    }
    if (audio_client_) {
      audio_client_->Release();
      audio_client_ = nullptr;
    }
    if (device_) {
      device_->Release();
      device_ = nullptr;
    }
    if (device_enumerator_) {
      device_enumerator_->Release();
      device_enumerator_ = nullptr;
    }
  }

  bool Initialize() {
    HRESULT hr;

    // Get the default audio capture device
    hr = CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
                          __uuidof(IMMDeviceEnumerator),
                          (void **)&device_enumerator_);
    if (FAILED(hr)) {
      std::cerr << "Failed to create device enumerator: " << std::hex << hr
                << std::endl;
      return false;
    }

    hr = device_enumerator_->GetDefaultAudioEndpoint(eCapture, eConsole,
                                                     &device_);
    if (FAILED(hr)) {
      std::cerr << "Failed to get default audio endpoint: " << std::hex << hr
                << std::endl;
      return false;
    }

    hr = device_->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr,
                           (void **)&audio_client_);
    if (FAILED(hr)) {
      std::cerr << "Failed to activate audio client: " << std::hex << hr
                << std::endl;
      return false;
    }

    // Get the mix format
    WAVEFORMATEX *pwfx = nullptr;
    hr = audio_client_->GetMixFormat(&pwfx);
    if (FAILED(hr)) {
      std::cerr << "Failed to get mix format: " << std::hex << hr << std::endl;
      return false;
    }

    // We need 16kHz, float32, mono
    WAVEFORMATEX desired_format = {};
    desired_format.wFormatTag = WAVE_FORMAT_IEEE_FLOAT;
    desired_format.nChannels = 1;
    desired_format.nSamplesPerSec = sample_rate_;
    desired_format.wBitsPerSample = 32;
    desired_format.nBlockAlign =
        desired_format.nChannels * (desired_format.wBitsPerSample / 8);
    desired_format.nAvgBytesPerSec =
        desired_format.nSamplesPerSec * desired_format.nBlockAlign;
    desired_format.cbSize = 0;

    // Initialize the audio client with desired format
    hr = audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 0, 0,
                                   &desired_format, nullptr);
    if (hr == AUDCLNT_E_UNSUPPORTED_FORMAT) {
      // Try with the device's native format and resample later
      // We already have pwfx from GetMixFormat above
      hr = audio_client_->Initialize(AUDCLNT_SHAREMODE_SHARED, 0, 0, 0, pwfx,
                                     nullptr);
      if (FAILED(hr)) {
        std::cerr << "Failed to initialize audio client with native format: "
                  << std::hex << hr << std::endl;
        CoTaskMemFree(pwfx);
        return false;
      }
      // Store the actual format for resampling
      actual_sample_rate_ = pwfx->nSamplesPerSec;
      actual_channels_ = pwfx->nChannels;
      CoTaskMemFree(pwfx);
    } else if (FAILED(hr)) {
      std::cerr << "Failed to initialize audio client: " << std::hex << hr
                << std::endl;
      CoTaskMemFree(pwfx);
      return false;
    } else {
      actual_sample_rate_ = sample_rate_;
      actual_channels_ = 1;
      CoTaskMemFree(pwfx);
    }

    // Get the capture client
    hr = audio_client_->GetService(__uuidof(IAudioCaptureClient),
                                   (void **)&capture_client_);
    if (FAILED(hr)) {
      std::cerr << "Failed to get capture client: " << std::hex << hr
                << std::endl;
      return false;
    }

    // Get buffer size
    hr = audio_client_->GetBufferSize(&buffer_frame_count_);
    if (FAILED(hr)) {
      std::cerr << "Failed to get buffer size: " << std::hex << hr << std::endl;
      return false;
    }

    return true;
  }

  void Start() {
    if (is_capturing_) {
      return;
    }

    HRESULT hr = audio_client_->Start();
    if (FAILED(hr)) {
      std::cerr << "Failed to start audio client: " << std::hex << hr
                << std::endl;
      return;
    }

    is_capturing_ = true;
    capture_thread_ = std::thread(&MicrophoneCapture::CaptureLoop, this);
  }

  void Stop() {
    if (!is_capturing_) {
      return;
    }

    is_capturing_ = false;
    if (capture_thread_.joinable()) {
      capture_thread_.join();
    }

    if (audio_client_) {
      audio_client_->Stop();
    }
  }

  void SetAudioCallback(
      std::function<void(const std::vector<float> &, int32_t)> callback) {
    audio_callback_ = callback;
  }

 private:
  void CaptureLoop() {
    UINT32 num_frames_available;
    BYTE *pData;
    DWORD flags;
    HRESULT hr;

    while (is_capturing_) {
      // Sleep for half the buffer duration
      Sleep(
          (DWORD)((1000.0 * buffer_frame_count_) / (2 * actual_sample_rate_)));

      hr = capture_client_->GetBuffer(&pData, &num_frames_available, &flags,
                                      nullptr, nullptr);
      if (FAILED(hr)) {
        continue;
      }

      if (num_frames_available == 0) {
        capture_client_->ReleaseBuffer(num_frames_available);
        continue;
      }

      if (flags & AUDCLNT_BUFFERFLAGS_SILENT) {
        // Buffer is silent, we can skip it or send zeros
        capture_client_->ReleaseBuffer(num_frames_available);
        continue;
      }

      // Convert the audio data to float32
      std::vector<float> audio_data;
      if (actual_sample_rate_ == sample_rate_ && actual_channels_ == 1) {
        // Direct conversion if format matches
        float *float_data = reinterpret_cast<float *>(pData);
        audio_data.assign(float_data, float_data + num_frames_available);
      } else {
        // Simple resampling: just take every Nth sample or duplicate
        // For simplicity, we'll do a basic downsampling
        // In production, you'd want proper resampling
        float *float_data = reinterpret_cast<float *>(pData);
        size_t total_samples = num_frames_available * actual_channels_;

        if (actual_channels_ > 1) {
          // Convert to mono by averaging channels
          for (size_t i = 0; i < num_frames_available; ++i) {
            float sum = 0.0f;
            for (UINT32 ch = 0; ch < actual_channels_; ++ch) {
              sum += float_data[i * actual_channels_ + ch];
            }
            audio_data.push_back(sum / actual_channels_);
          }
        } else {
          audio_data.assign(float_data, float_data + num_frames_available);
        }

        // Simple resampling: if sample rate differs, we'll just decimate
        // This is a very basic approach - for production use proper resampling
        if (actual_sample_rate_ != sample_rate_) {
          double ratio =
              static_cast<double>(sample_rate_) / actual_sample_rate_;
          std::vector<float> resampled;
          for (size_t i = 0; i < audio_data.size(); ++i) {
            size_t target_index = static_cast<size_t>(i * ratio);
            if (target_index < resampled.size()) {
              resampled[target_index] = audio_data[i];
            } else {
              resampled.push_back(audio_data[i]);
            }
          }
          audio_data = std::move(resampled);
        }
      }

      if (audio_callback_ && !audio_data.empty()) {
        audio_callback_(audio_data, sample_rate_);
      }

      capture_client_->ReleaseBuffer(num_frames_available);
    }
  }

  IMMDeviceEnumerator *device_enumerator_ = nullptr;
  IMMDevice *device_ = nullptr;
  IAudioClient *audio_client_ = nullptr;
  IAudioCaptureClient *capture_client_ = nullptr;
  UINT32 buffer_frame_count_ = 0;
  std::atomic<bool> is_capturing_;
  std::thread capture_thread_;
  std::function<void(const std::vector<float> &, int32_t)> audio_callback_;
  int32_t sample_rate_;
  UINT32 actual_sample_rate_ = 0;
  UINT16 actual_channels_ = 0;
};

namespace {

class WavFileProducer {
 public:
  explicit WavFileProducer(std::string wav_path,
                           float chunk_duration_seconds = 0.1f)
      : current_index_(0) {
    loadWavData(std::move(wav_path));
    chunk_size_ = static_cast<size_t>(chunk_duration_seconds * sample_rate_);
    if (chunk_size_ == 0) {
      chunk_size_ = 1;
    }
  }

  bool getNextAudio(std::vector<float> &out_audio_data) {
    if (current_index_ >= audio_data_.size()) {
      return false;
    }
    const size_t end_index =
        std::min(current_index_ + chunk_size_, audio_data_.size());
    out_audio_data.assign(audio_data_.begin() + current_index_,
                          audio_data_.begin() + end_index);
    current_index_ = end_index;
    return true;
  }

  int32_t sampleRate() const { return sample_rate_; }

 private:
  void loadWavData(const std::string &wav_path) {
    audio_data_.clear();
    sample_rate_ = 0;

    FILE *file = std::fopen(wav_path.c_str(), "rb");
    if (!file) {
      throw std::runtime_error("Failed to open WAV file: " + wav_path);
    }

    char riff_header[4];
    if (std::fread(riff_header, 1, 4, file) != 4 ||
        std::strncmp(riff_header, "RIFF", 4) != 0) {
      std::fclose(file);
      throw std::runtime_error("Not a RIFF file: " + wav_path);
    }

    std::fseek(file, 4, SEEK_CUR);
    char wave_header[4];
    if (std::fread(wave_header, 1, 4, file) != 4 ||
        std::strncmp(wave_header, "WAVE", 4) != 0) {
      std::fclose(file);
      throw std::runtime_error("Not a WAVE file: " + wav_path);
    }

    char chunk_id[4];
    uint32_t chunk_size = 0;
    bool found_fmt = false;
    while (std::fread(chunk_id, 1, 4, file) == 4) {
      if (std::fread(&chunk_size, 4, 1, file) != 1) {
        break;
      }
      if (std::strncmp(chunk_id, "fmt ", 4) == 0) {
        found_fmt = true;
        break;
      }
      std::fseek(file, chunk_size, SEEK_CUR);
    }
    if (!found_fmt || chunk_size < 16) {
      std::fclose(file);
      throw std::runtime_error("Invalid fmt chunk in WAV file: " + wav_path);
    }

    uint16_t audio_format = 0;
    uint16_t num_channels = 0;
    uint32_t sample_rate = 0;
    uint16_t bits_per_sample = 0;
    std::fread(&audio_format, sizeof(uint16_t), 1, file);
    std::fread(&num_channels, sizeof(uint16_t), 1, file);
    std::fread(&sample_rate, sizeof(uint32_t), 1, file);
    std::fseek(file, 6, SEEK_CUR);
    std::fread(&bits_per_sample, sizeof(uint16_t), 1, file);
    if (chunk_size > 16) {
      std::fseek(file, chunk_size - 16, SEEK_CUR);
    }

    if (audio_format != 1 || bits_per_sample != 16) {
      std::fclose(file);
      throw std::runtime_error("Only 16-bit PCM WAV files are supported");
    }

    bool found_data = false;
    while (std::fread(chunk_id, 1, 4, file) == 4) {
      if (std::fread(&chunk_size, 4, 1, file) != 1) {
        break;
      }
      if (std::strncmp(chunk_id, "data", 4) == 0) {
        found_data = true;
        break;
      }
      std::fseek(file, chunk_size, SEEK_CUR);
    }
    if (!found_data) {
      std::fclose(file);
      throw std::runtime_error("No data chunk found in WAV file: " + wav_path);
    }

    const size_t num_samples = chunk_size / (bits_per_sample / 8);
    std::vector<int16_t> pcm_data(num_samples);
    if (std::fread(pcm_data.data(), bits_per_sample / 8, num_samples, file) !=
        num_samples) {
      std::fclose(file);
      throw std::runtime_error("Failed to read WAV PCM data: " + wav_path);
    }
    std::fclose(file);

    sample_rate_ = static_cast<int32_t>(sample_rate);
    audio_data_.reserve(num_samples / num_channels);
    for (size_t i = 0; i < num_samples; i += num_channels) {
      int32_t sample = pcm_data[i];
      if (num_channels > 1) {
        for (uint16_t ch = 1; ch < num_channels; ++ch) {
          sample += pcm_data[i + ch];
        }
        sample /= num_channels;
      }
      audio_data_.push_back(static_cast<float>(sample) / 32768.0f);
    }
  }

  size_t chunk_size_ = 0;
  int32_t sample_rate_ = 0;
  size_t current_index_ = 0;
  std::vector<float> audio_data_;
};

int runWavTranscription(const std::string &model_path,
                        moonshine::ModelArch model_arch,
                        const std::string &wav_path) {
  WavFileProducer audio_producer(wav_path);
  moonshine::Transcriber transcriber(model_path, model_arch, 0.5f);

  TranscriptionListener listener;
  transcriber.addListener(&listener);
  transcriber.start();

  std::vector<float> chunk_audio_data;
  const float transcription_interval_seconds = 0.481f;
  const int32_t samples_between_transcriptions = static_cast<int32_t>(
      transcription_interval_seconds * audio_producer.sampleRate());
  int32_t samples_since_last_transcription = 0;

  while (audio_producer.getNextAudio(chunk_audio_data)) {
    transcriber.addAudio(chunk_audio_data, audio_producer.sampleRate());
    samples_since_last_transcription +=
        static_cast<int32_t>(chunk_audio_data.size());
    if (samples_since_last_transcription < samples_between_transcriptions) {
      continue;
    }
    samples_since_last_transcription = 0;
    transcriber.updateTranscription();
  }

  transcriber.stop();
  return 0;
}

}  // namespace

int main(int argc, char *argv[]) {
  std::string model_path = "models/medium-streaming-en";
  moonshine::ModelArch model_arch = moonshine::ModelArch::MEDIUM_STREAMING;
  std::string wav_path;

  // Parse command line arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-m" || arg == "--model-path") && i + 1 < argc) {
      model_path = argv[++i];
    } else if ((arg == "-a" || arg == "--model-arch") && i + 1 < argc) {
      int arch = std::stoi(argv[++i]);
      model_arch = static_cast<moonshine::ModelArch>(arch);
    } else if ((arg == "-w" || arg == "--wav-path") && i + 1 < argc) {
      wav_path = argv[++i];
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: cli-transcriber [options]\n"
                << "Options:\n"
                << "  -m, --model-path PATH    Path to model directory "
                   "(default: models/medium-streaming-en)\n"
                << "  -a, --model-arch ARCH    Model architecture: 0=TINY, "
                   "1=BASE, 2=TINY_STREAMING, 3=BASE_STREAMING, "
                   "4=SMALL_STREAMING, 5=MEDIUM_STREAMING "
                   "(default: 5)\n"
                << "  -w, --wav-path PATH      Transcribe a WAV file and exit "
                   "(default: microphone mode)\n"
                << "  -h, --help               Show this help message\n";
      return 0;
    } else {
      std::cerr << "Unknown argument: " << arg << std::endl;
      return 1;
    }
  }

  try {
    if (!wav_path.empty()) {
      std::cout << "Loading transcriber from: " << model_path << std::endl;
      std::cout << "Transcribing: " << wav_path << std::endl;
      return runWavTranscription(model_path, model_arch, wav_path);
    }

    // Initialize COM
    COMInitializer com_init;

    // Initialize microphone capture
    MicrophoneCapture mic;
    if (!mic.Initialize()) {
      std::cerr << "Failed to initialize microphone" << std::endl;
      return 1;
    }

    // Initialize transcriber
    std::cout << "Loading transcriber from: " << model_path << std::endl;
    moonshine::Transcriber transcriber(model_path, model_arch, 0.5);

    // Set up listener
    TranscriptionListener listener;
    transcriber.addListener(&listener);

    // Start transcription
    std::cout << "Starting transcription... Press Ctrl+C to stop." << std::endl;
    transcriber.start();

    // Set up audio callback
    mic.SetAudioCallback([&transcriber](const std::vector<float> &audio_data,
                                        int32_t sample_rate) {
      try {
        transcriber.addAudio(audio_data, sample_rate);
      } catch (const moonshine::MoonshineException &e) {
        std::cerr << "\nTranscription error: " << e.what() << std::endl;
      }
    });

    // Start capturing
    mic.Start();

    // Wait for user interrupt (Ctrl+C)
    std::cout << "Listening to microphone..." << std::endl;

    // Main loop - wait for Ctrl+C
    // Note: Pressing Ctrl+C will terminate the process, and destructors will
    // handle cleanup
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

  } catch (const moonshine::MoonshineException &e) {
    std::cerr << "Moonshine error: " << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
