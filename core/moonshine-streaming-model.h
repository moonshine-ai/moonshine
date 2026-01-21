#ifndef MOONSHINE_STREAMING_MODEL_H
#define MOONSHINE_STREAMING_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include <mutex>
#include <string>
#include <vector>

#include "bin-tokenizer.h"
#include "moonshine-ort-allocator.h"
#include "onnxruntime_c_api.h"

/* Streaming model configuration (matches streaming_config.json) */
struct MoonshineStreamingConfig {
  int encoder_dim;      /* Encoder hidden dimension (320) */
  int decoder_dim;      /* Decoder hidden dimension (320) */
  int depth;            /* Number of decoder layers (6) */
  int nheads;           /* Number of attention heads (8) */
  int head_dim;         /* Dimension per head (40) */
  int vocab_size;       /* Vocabulary size (32768) */
  int bos_id;           /* Beginning of sequence token ID (1) */
  int eos_id;           /* End of sequence token ID (2) */
  int frame_len;        /* Audio samples per frame (80) */
  int total_lookahead;  /* Encoder lookahead frames (16) */
  int d_model_frontend; /* Frontend linear output dim (320) */
  int c1;               /* Conv1 output channels (640) */
  int c2;               /* Conv2 output channels (320) */
  int max_seq_len;      /* Maximum sequence length for decoder (448) */
};

/* Internal state for streaming inference */
struct MoonshineStreamingState {
  // Frontend state
  std::vector<float> sample_buffer;  // [79]
  int64_t sample_len;
  std::vector<float> conv1_buffer;  // [d_model, 4]
  std::vector<float> conv2_buffer;  // [c1, 4]
  int64_t frame_count;

  // Feature accumulator (for encoder Path A)
  std::vector<float> accumulated_features;  // [T, encoder_dim]
  int accumulated_feature_count;

  // Encoder output tracking
  int encoder_frames_emitted;

  // Adapter position tracking
  int64_t adapter_pos_offset;

  // Memory accumulator
  std::vector<float> memory;  // [T, decoder_dim]
  int memory_len;

  // Decoder self-attention KV cache
  std::vector<float> k_self;
  std::vector<float> v_self;
  int cache_seq_len;

  // Cross-attention KV cache (precomputed from memory)
  // Used with decoder_kv.onnx for more efficient decoding
  std::vector<float> k_cross;
  std::vector<float> v_cross;
  int cross_len;
  bool cross_kv_valid;  // True if k_cross/v_cross are valid for current memory

  void reset(const MoonshineStreamingConfig &cfg);
};

struct MoonshineStreamingModel {
  const OrtApi *ort_api;
  OrtEnv *ort_env;
  OrtSessionOptions *ort_session_options;
  OrtMemoryInfo *ort_memory_info;
  MoonshineOrtAllocator *ort_allocator;

  OrtSession *frontend_session;
  OrtSession *encoder_session;
  OrtSession *adapter_session;

  // Optimized decoding sessions (cross_kv.onnx/decoder_kv.onnx)
  OrtSession *cross_kv_session;    // Computes cross-attention K/V from memory
  OrtSession *decoder_kv_session;  // Decoder that takes precomputed cross K/V

  BinTokenizer *tokenizer;
  std::mutex processing_mutex;

  MoonshineStreamingConfig config;

  // Memory-mapped data (if loaded from files)
  const char *frontend_mmapped_data = nullptr;
  size_t frontend_mmapped_data_size = 0;
  const char *encoder_mmapped_data = nullptr;
  size_t encoder_mmapped_data_size = 0;
  const char *adapter_mmapped_data = nullptr;
  size_t adapter_mmapped_data_size = 0;

  std::string last_result;

  bool log_ort_run = false;

  MoonshineStreamingModel(bool log_ort_run = false);
  ~MoonshineStreamingModel();

  int load(const char *model_dir, const char *tokenizer_path,
           int32_t model_type);

  int load_from_memory(
      const uint8_t *frontend_model_data, size_t frontend_model_data_size,
      const uint8_t *encoder_model_data, size_t encoder_model_data_size,
      const uint8_t *adapter_model_data, size_t adapter_model_data_size,
      const uint8_t *cross_kv_model_data, size_t cross_kv_model_data_size,
      const uint8_t *decoder_kv_model_data, size_t decoder_kv_model_data_size,
      const uint8_t *tokenizer_data, size_t tokenizer_data_size,
      const MoonshineStreamingConfig &config, int32_t model_type);

#if defined(ANDROID)
  int load_from_assets(const char *model_dir, const char *tokenizer_path,
                       int32_t model_type, AAssetManager *assetManager);
#endif

  /* Batch transcription - processes all audio at once */
  int transcribe(const float *input_audio_data, size_t input_audio_data_size,
                 char **out_text);

  /* Streaming inference methods */
  int process_audio_chunk(MoonshineStreamingState *state,
                          const float *audio_chunk, size_t chunk_len,
                          int *features_out);

  int encode(MoonshineStreamingState *state, bool is_final,
             int *new_frames_out);

  /* Single-token decode step (auto-regressive) */
  int decode_step(MoonshineStreamingState *state, int token, float *logits_out);

  /* Multi-token decode step - processes multiple tokens at once, returns logits
   * for each position. Useful for speculative decoding verification. logits_out
   * must have space for (tokens_len * config.vocab_size) floats. Returns logits
   * for ALL token positions, not just the last one. */
  int decode_tokens(MoonshineStreamingState *state, const int *tokens,
                    int tokens_len, float *logits_out);

  /* Full decode with optional speculative tokens.
   * If speculative_tokens is provided, verifies them against new predictions
   * and continues from divergence point.
   * tokens_out is allocated by this function - caller must free with free().
   * Returns 0 on success. */
  int decode_full(MoonshineStreamingState *state, const int *speculative_tokens,
                  int speculative_len, int **tokens_out, int *tokens_len_out);

  void decoder_reset(MoonshineStreamingState *state);

  /* Create a new streaming state */
  MoonshineStreamingState *create_state();

  /* Decode tokens to text using the tokenizer */
  std::string tokens_to_text(const std::vector<int64_t> &tokens);

 private:
  int load_config(const char *config_path);
  int load_config_from_string(const std::string &json);

  /* Internal helper that uses precomputed cross K/V */
  int run_decoder_with_cross_kv(MoonshineStreamingState *state,
                                const std::vector<int64_t> &tokens,
                                std::vector<float> &logits_out);

  /* Compute cross-attention K/V from current memory state */
  int compute_cross_kv(MoonshineStreamingState *state);
};

#endif
