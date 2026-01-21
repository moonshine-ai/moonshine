

#include <chrono>
#include <cmath>  // for std::rint
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#if __cplusplus < 201703L
#include <memory>
#endif

#include "onnxruntime_c_api.h"

class SileroVad {
 private:
  // ONNX Runtime C API resources
  const OrtApi *ort_api;
  OrtEnv *env;
  OrtSessionOptions *session_options;
  OrtSession *session;
  OrtAllocator *allocator;
  OrtMemoryInfo *memory_info;

  // ----- Context-related additions -----
  static const int context_samples =
      64;                       // For 16kHz, 64 samples are added as context.
  std::vector<float> _context;  // Holds the last 64 samples from the previous
                                // chunk (initialized to zero).

  // Original window size (e.g., 32ms corresponds to 512 samples)
  int window_size_samples;
  // Effective window size = window_size_samples + context_samples
  int effective_window_size;

  // Additional declaration: samples per millisecond
  int sr_per_ms;

  // ONNX Runtime input/output buffers
  std::vector<OrtValue *> ort_inputs;
  std::vector<const char *> input_node_names = {"input", "state", "sr"};
  std::vector<float> input;
  unsigned int size_state = 2 * 1 * 128;
  std::vector<float> _state;
  int64_t sr;  // scalar sample rate
  int64_t input_node_dims[2] = {};
  const int64_t state_node_dims[3] = {2, 1, 128};
  std::vector<OrtValue *> ort_outputs;
  std::vector<const char *> output_node_names = {"output", "stateN"};

  // Model configuration parameters
  float threshold;
  int min_silence_samples;
  int min_silence_samples_at_max_speech;
  int min_speech_samples;
  float max_speech_samples;
  int speech_pad_samples;

  // Initializes the common ONNX runtime environment (env, session_options,
  // memory_info, allocator).
  void init_onnx_env();

  // Initializes threading settings.
  void init_engine_threads(int inter_threads, int intra_threads);

 public:
  SileroVad(
      int sample_rate = 16000, int windows_frame_size = 32,
      float threshold = 0.5, int min_silence_duration_ms = 100,
      int speech_pad_ms = 30, int min_speech_duration_ms = 250,
      float max_speech_duration_s = std::numeric_limits<float>::infinity());

  ~SileroVad();

  // Load model from memory buffer.
  int load_from_memory(const uint8_t *model_data, size_t model_data_size);

  bool is_loaded() const { return session != nullptr; }

  void predict(const std::vector<float> &data_chunk, float *out_probability,
               int *out_flag);
};
