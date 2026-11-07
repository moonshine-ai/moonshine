#ifndef WORD_ALIGNMENT_H
#define WORD_ALIGNMENT_H

#include <string>
#include <vector>

#include "bin-tokenizer/bin-tokenizer.h"

struct TranscriberWord {
  std::string text;
  float start;  // seconds, absolute from audio start
  float end;    // seconds
  // 0.0-1.0: the lowest softmax probability the decoder gave any of the word's
  // tokens when it chose them, before key-term biasing, leaving out tokens that
  // are only punctuation unless the word has nothing else. 1.0 if the decoder's
  // probabilities were not supplied to align_words.
  float confidence;
};

// Softmax probability of ``logits[token_id]`` over ``vocab_size`` logits.
// Subtracts the row's maximum first, so large logits cannot overflow. Returns
// 0.0 for a token outside the row.
float token_probability(const float* logits, int vocab_size, int token_id);

// Dynamic Time Warping on a cost matrix [N x M]
// Returns aligned (text_indices, time_indices) arrays
void dtw(const std::vector<float>& cost_matrix, int N, int M,
         std::vector<int>& text_indices_out,
         std::vector<int>& time_indices_out);

// Apply median filter along the last axis of a 3D array [C x H x W]
// filter_width should be odd
void median_filter(std::vector<float>& data, int channels, int height,
                   int width, int filter_width);

// Main entry point: given cross-attention weights and token info, produce word
// timings.
//
// cross_attention_data: flattened [num_layers * num_heads, num_tokens,
// encoder_frames] tokens: the generated token IDs (including BOS and EOS)
// token_probabilities: the probability the decoder gave each entry of
// ``tokens`` when it chose it (see token_probability), index for index, so the
// BOS entry is a placeholder. Each word's confidence is the minimum over its
// tokens, not counting tokens that are only punctuation (see
// TranscriberWord::confidence). Pass an empty vector, or one whose length
// differs from ``tokens``, and every confidence is 1.0.
// num_layers, num_heads, num_tokens, encoder_frames: dimensions
// time_per_frame: seconds per encoder frame (e.g., audio_duration /
// encoder_frames) tokenizer: pointer to BinTokenizer for decoding tokens and
// detecting word boundaries
//
// Returns vector of TranscriberWord with absolute timestamps.
std::vector<TranscriberWord> align_words(
    const float* cross_attention_data, int num_layers, int num_heads,
    int num_tokens, int encoder_frames, const std::vector<int>& tokens,
    const std::vector<float>& token_probabilities, float time_per_frame,
    BinTokenizer* tokenizer);

#endif  // WORD_ALIGNMENT_H
