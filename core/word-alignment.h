#ifndef WORD_ALIGNMENT_H
#define WORD_ALIGNMENT_H

#include <string>
#include <vector>

#include "bin-tokenizer/bin-tokenizer.h"

struct TranscriberWord {
  std::string text;
  float start;      // seconds, absolute from audio start
  float end;        // seconds
  float confidence;  // 0.0-1.0
};

// Dynamic Time Warping on a cost matrix [N x M]
// Returns aligned (text_indices, time_indices) arrays
void dtw(const std::vector<float>& cost_matrix, int N, int M,
     std::vector<int>& text_indices_out, std::vector<int>& time_indices_out);

// Apply median filter along the last axis of a 3D array [C x H x W]
// filter_width should be odd
void median_filter(std::vector<float>& data, int channels, int height, int width, int filter_width);

// Main entry point: given cross-attention weights and token info, produce word timings.
//
// cross_attention_data: flattened [num_layers * num_heads, num_tokens, encoder_frames]
// tokens: the generated token IDs (including BOS and EOS)
// num_layers, num_heads, num_tokens, encoder_frames: dimensions
// time_per_frame: seconds per encoder frame (e.g., audio_duration / encoder_frames)
// tokenizer: pointer to BinTokenizer for decoding tokens and detecting word boundaries
//
// Returns vector of TranscriberWord with absolute timestamps.
std::vector<TranscriberWord> align_words(
  const float* cross_attention_data,
  int num_layers, int num_heads, int num_tokens, int encoder_frames,
  const std::vector<int>& tokens,
  float time_per_frame,
  BinTokenizer* tokenizer);

#endif // WORD_ALIGNMENT_H
