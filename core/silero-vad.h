

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

// Everything Silero carries from one frame to the next: the LSTM cells, and the
// 64 samples of the previous frame that the model expects prepended to the next
// one. It belongs to the caller rather than to SileroVad because one model
// instance serves every stream in the process, so the state cannot live beside
// the session without streams overwriting each other's.
//
// Sharing it is not a subtle error. A stream that inherits another's state
// scores its opening second against a conversation it never heard, and a fresh
// stream in a long-lived process inherits whatever the last one was saying.
// Both show up as speech detected where the audio is silent.
struct SileroVadState {
  static constexpr size_t state_size = 2 * 1 * 128;
  // For 16 kHz, 64 samples are prepended as context.
  static constexpr size_t context_samples = 64;

  std::vector<float> state = std::vector<float>(state_size, 0.0f);
  std::vector<float> context = std::vector<float>(context_samples, 0.0f);

  // Back to "no audio has been seen". Every stream boundary needs this.
  void reset() {
    state.assign(state_size, 0.0f);
    context.assign(context_samples, 0.0f);
  }
};

class SileroVad {
 private:
  // ONNX Runtime C API resources
  const OrtApi *ort_api;
  OrtEnv *env;
  OrtSessionOptions *session_options;
  OrtSession *session;
  OrtAllocator *allocator;
  OrtMemoryInfo *memory_info;

  static const int context_samples =
      static_cast<int>(SileroVadState::context_samples);

  // Original window size (e.g., 32ms corresponds to 512 samples)
  int window_size_samples;
  // Effective window size = window_size_samples + context_samples
  int effective_window_size;

  // Additional declaration: samples per millisecond
  int sr_per_ms;

  // ONNX Runtime input/output buffers. The per-call scratch that used to live
  // here -- the input window and the tensor array -- is local to predict now,
  // so that two streams scoring a frame at the same time cannot tread on it.
  std::vector<const char *> input_node_names = {"input", "state", "sr"};
  unsigned int size_state = SileroVadState::state_size;
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

  // Scores one frame, advancing `state`. The caller owns that state and must
  // give each stream its own; see SileroVadState.
  void predict(const std::vector<float> &data_chunk, SileroVadState &state,
               float *out_probability, int *out_flag);
};
