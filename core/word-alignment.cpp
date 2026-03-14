#include "word-alignment.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

// ============================================================================
// DTW (Dynamic Time Warping)
// ============================================================================

void dtw(const std::vector<float>& cost_matrix, int N, int M,
         std::vector<int>& text_indices_out, std::vector<int>& time_indices_out) {
    // Cumulative cost matrix D of size (N+1) x (M+1), initialized to infinity
    std::vector<float> D((N + 1) * (M + 1), std::numeric_limits<float>::infinity());
    D[0 * (M + 1) + 0] = 0.0f;

    // Trace matrix of size N x M, stores which predecessor was chosen (0, 1, or 2)
    std::vector<int> trace(N * M, 0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            // Three candidates:
            //   0: diagonal (i, j)     -> D[i][j]
            //   1: vertical (i, j+1)   -> D[i][j+1]
            //   2: horizontal (i+1, j) -> D[i+1][j]
            float c0 = D[i * (M + 1) + j];         // diagonal
            float c1 = D[i * (M + 1) + (j + 1)];   // vertical (text advances, time stays)
            float c2 = D[(i + 1) * (M + 1) + j];   // horizontal (time advances, text stays)

            int argmin;
            float min_val;
            if (c0 <= c1 && c0 <= c2) {
                argmin = 0;
                min_val = c0;
            } else if (c1 <= c0 && c1 <= c2) {
                argmin = 1;
                min_val = c1;
            } else {
                argmin = 2;
                min_val = c2;
            }

            trace[i * M + j] = argmin;
            D[(i + 1) * (M + 1) + (j + 1)] = cost_matrix[i * M + j] + min_val;
        }
    }

    // Backtrace from (N-1, M-1) to (0, 0)
    int i = N - 1;
    int j = M - 1;
    std::vector<int> text_indices_rev;
    std::vector<int> time_indices_rev;

    while (i >= 0 || j >= 0) {
        text_indices_rev.push_back(i);
        time_indices_rev.push_back(j);

        if (i == 0 && j == 0) {
            break;
        }

        int direction = trace[i * M + j];
        if (direction == 0) {
            // diagonal
            i--;
            j--;
        } else if (direction == 1) {
            // vertical (text retreats)
            i--;
        } else {
            // horizontal (time retreats)
            j--;
        }
    }

    // Reverse to get forward order
    text_indices_out.resize(text_indices_rev.size());
    time_indices_out.resize(time_indices_rev.size());
    for (size_t k = 0; k < text_indices_rev.size(); k++) {
        text_indices_out[k] = text_indices_rev[text_indices_rev.size() - 1 - k];
        time_indices_out[k] = time_indices_rev[time_indices_rev.size() - 1 - k];
    }
}

// ============================================================================
// Median filter (along last axis of 3D array)
// ============================================================================

static float compute_median(std::vector<float>& window) {
    size_t n = window.size();
    std::nth_element(window.begin(), window.begin() + n / 2, window.end());
    return window[n / 2];
}

void median_filter(std::vector<float>& data, int channels, int height, int width, int filter_width) {
    if (filter_width <= 1) {
        return;
    }

    // Ensure filter_width is odd
    if (filter_width % 2 == 0) {
        filter_width += 1;
    }

    int pad = filter_width / 2;
    int padded_width = width + 2 * pad;

    // Work buffer for one row (padded with reflected values)
    std::vector<float> padded(padded_width);
    std::vector<float> window(filter_width);
    std::vector<float> result_row(width);

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            int row_offset = (c * height + h) * width;

            // Fill padded buffer with reflect padding
            // Left pad: reflect
            for (int p = 0; p < pad; p++) {
                int src_idx = pad - p;  // reflect index
                if (src_idx >= width) src_idx = width - 1;
                padded[p] = data[row_offset + src_idx];
            }
            // Center: copy original data
            for (int w = 0; w < width; w++) {
                padded[pad + w] = data[row_offset + w];
            }
            // Right pad: reflect
            for (int p = 0; p < pad; p++) {
                int src_idx = width - 2 - p;  // reflect index
                if (src_idx < 0) src_idx = 0;
                padded[pad + width + p] = data[row_offset + src_idx];
            }

            // Apply median filter
            for (int w = 0; w < width; w++) {
                for (int k = 0; k < filter_width; k++) {
                    window[k] = padded[w + k];
                }
                result_row[w] = compute_median(window);
            }

            // Write back
            for (int w = 0; w < width; w++) {
                data[row_offset + w] = result_row[w];
            }
        }
    }
}

// ============================================================================
// Helper: check if a token's raw bytes start with the SentencePiece word
// boundary marker (UTF-8 encoding of U+2581 LOWER ONE EIGHTH BLOCK).
// ============================================================================

static bool token_starts_new_word(BinTokenizer* tokenizer, int token_id) {
    if (token_id < 0 || token_id >= (int)tokenizer->tokens_to_bytes.size()) {
        return false;
    }
    const std::vector<uint8_t>& bytes = tokenizer->tokens_to_bytes[token_id];
    // The UTF-8 encoding of U+2581 is 0xE2 0x96 0x81 (3 bytes)
    if (bytes.size() >= 3 &&
        bytes[0] == 0xE2 &&
        bytes[1] == 0x96 &&
        bytes[2] == 0x81) {
        return true;
    }
    return false;
}

// ============================================================================
// Helper: decode a list of token IDs to text.
// ============================================================================

static std::string decode_tokens(BinTokenizer* tokenizer, const std::vector<int>& token_ids) {
    return tokenizer->tokens_to_text(token_ids, true);
}

// ============================================================================
// align_words: main entry point
// ============================================================================

std::vector<TranscriberWord> align_words(
    const float* cross_attention_data,
    int num_layers, int num_heads, int num_tokens, int encoder_frames,
    const std::vector<int>& tokens,
    float time_per_frame,
    BinTokenizer* tokenizer) {

    if (!cross_attention_data || num_tokens <= 0 || encoder_frames <= 0) {
        return {};
    }

    int total_heads = num_layers * num_heads;
    int n_steps = num_tokens;  // number of decode steps (rows in attention matrix)

    // -----------------------------------------------------------------------
    // Step 1: Copy cross_attention_data into a working buffer
    //         Shape: [total_heads, n_steps, encoder_frames]
    // -----------------------------------------------------------------------
    size_t total_size = (size_t)total_heads * n_steps * encoder_frames;
    std::vector<float> weights(total_size);
    std::memcpy(weights.data(), cross_attention_data, total_size * sizeof(float));

    // -----------------------------------------------------------------------
    // Step 2: Z-score normalize per head (along the time/encoder_frames axis)
    //
    // For each head h, for each token position t:
    //   Compute mean and std across encoder_frames, then normalize.
    // Actually the Python code normalizes per head with axis=-1 and keepdims,
    // which means for each (head, token_position), normalize across encoder_frames.
    // -----------------------------------------------------------------------
    for (int h = 0; h < total_heads; h++) {
        for (int t = 0; t < n_steps; t++) {
            int offset = (h * n_steps + t) * encoder_frames;

            // Compute mean
            float sum = 0.0f;
            for (int f = 0; f < encoder_frames; f++) {
                sum += weights[offset + f];
            }
            float mean = sum / encoder_frames;

            // Compute std
            float sq_sum = 0.0f;
            for (int f = 0; f < encoder_frames; f++) {
                float diff = weights[offset + f] - mean;
                sq_sum += diff * diff;
            }
            float stddev = std::sqrt(sq_sum / encoder_frames);
            if (stddev == 0.0f) {
                stddev = 1e-10f;
            }

            // Normalize
            for (int f = 0; f < encoder_frames; f++) {
                weights[offset + f] = (weights[offset + f] - mean) / stddev;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Median filter (width=7) along the last axis
    //         Shape is [total_heads, n_steps, encoder_frames]
    // -----------------------------------------------------------------------
    median_filter(weights, total_heads, n_steps, encoder_frames, 7);

    // -----------------------------------------------------------------------
    // Step 4: Average across all heads/layers -> [n_steps, encoder_frames]
    // -----------------------------------------------------------------------
    std::vector<float> matrix(n_steps * encoder_frames, 0.0f);
    for (int h = 0; h < total_heads; h++) {
        for (int t = 0; t < n_steps; t++) {
            int src_offset = (h * n_steps + t) * encoder_frames;
            int dst_offset = t * encoder_frames;
            for (int f = 0; f < encoder_frames; f++) {
                matrix[dst_offset + f] += weights[src_offset + f];
            }
        }
    }
    // Divide by total_heads to get the average
    float inv_heads = 1.0f / total_heads;
    for (size_t i = 0; i < matrix.size(); i++) {
        matrix[i] *= inv_heads;
    }

    // -----------------------------------------------------------------------
    // Step 5: Run DTW on the negated matrix
    //         (DTW minimizes cost; we want to maximize attention)
    // -----------------------------------------------------------------------
    std::vector<float> neg_matrix(matrix.size());
    for (size_t i = 0; i < matrix.size(); i++) {
        neg_matrix[i] = -matrix[i];
    }

    std::vector<int> text_indices, time_indices;
    dtw(neg_matrix, n_steps, encoder_frames, text_indices, time_indices);

    // -----------------------------------------------------------------------
    // Step 6: Group tokens into words using SentencePiece word boundaries
    //
    // tokens = [BOS, tok1, tok2, ..., tokN, EOS]
    // The cross-attention matrix has n_steps rows corresponding to decode steps.
    // Step i produced tokens[i+1] (step 0 consumed BOS and produced tok1).
    // DTW text_indices are row indices in [0, n_steps).
    // We map step i -> tokens[i+1].
    //
    // text_tokens = tokens[1:-1] (exclude BOS and EOS)
    // n_text_steps = len(text_tokens)
    // -----------------------------------------------------------------------

    // Extract text tokens (exclude BOS at index 0 and EOS at last index)
    std::vector<int> text_tokens;
    if (tokens.size() >= 2) {
        text_tokens.assign(tokens.begin() + 1, tokens.end() - 1);
    }

    int n_text_steps = (int)text_tokens.size();
    if (n_text_steps == 0) {
        return {};
    }

    // Group tokens into words
    // Each word is a pair: (list of token IDs, list of step indices)
    struct WordGroup {
        std::vector<int> token_ids;
        std::vector<int> step_indices;
    };

    std::vector<WordGroup> words;
    WordGroup current_word;

    for (int i = 0; i < n_text_steps; i++) {
        int tok_id = text_tokens[i];
        int step_idx = i;  // row index in DTW matrix

        bool starts_new = token_starts_new_word(tokenizer, tok_id);

        if (starts_new && !current_word.token_ids.empty()) {
            words.push_back(current_word);
            current_word = WordGroup();
        }

        current_word.token_ids.push_back(tok_id);
        current_word.step_indices.push_back(step_idx);
    }

    if (!current_word.token_ids.empty()) {
        words.push_back(current_word);
    }

    // -----------------------------------------------------------------------
    // Step 7: Map DTW alignment to word start/end times
    // -----------------------------------------------------------------------
    std::vector<TranscriberWord> word_timings;

    for (const auto& word_group : words) {
        std::string word_text = decode_tokens(tokenizer, word_group.token_ids);
        // Trim whitespace
        size_t start_pos = word_text.find_first_not_of(" \t\n\r");
        size_t end_pos = word_text.find_last_not_of(" \t\n\r");
        if (start_pos != std::string::npos && end_pos != std::string::npos) {
            word_text = word_text.substr(start_pos, end_pos - start_pos + 1);
        }

        if (word_text.empty()) {
            continue;
        }

        // Find all DTW path entries where text_indices matches any of this word's step indices
        int min_frame = encoder_frames;  // will track minimum time frame
        int max_frame = -1;              // will track maximum time frame

        for (size_t p = 0; p < text_indices.size(); p++) {
            int ti = text_indices[p];
            for (int si : word_group.step_indices) {
                if (ti == si) {
                    int frame = time_indices[p];
                    if (frame < min_frame) min_frame = frame;
                    if (frame > max_frame) max_frame = frame;
                    break;
                }
            }
        }

        TranscriberWord tw;
        tw.text = word_text;
        tw.confidence = 1.0f;  // default confidence

        if (max_frame < 0) {
            // No DTW match found for this word
            tw.start = 0.0f;
            tw.end = 0.0f;
        } else {
            tw.start = min_frame * time_per_frame;
            tw.end = (max_frame + 1) * time_per_frame;
        }

        word_timings.push_back(tw);
    }

    // -----------------------------------------------------------------------
    // Step 8: Fix overlapping word boundaries (snap to midpoint)
    // -----------------------------------------------------------------------
    for (size_t i = 1; i < word_timings.size(); i++) {
        if (word_timings[i - 1].end > word_timings[i].start) {
            float midpoint = (word_timings[i - 1].end + word_timings[i].start) * 0.5f;
            word_timings[i - 1].end = midpoint;
            word_timings[i].start = midpoint;
        }
    }

    return word_timings;
}
