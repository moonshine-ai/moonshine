// SPDX-License-Identifier: MIT

#include "filter_train.h"

#include <cmath>
#include <vector>

namespace cppannote::filter_train {

void filter_embeddings_train(int num_chunks, int num_frames, int num_speakers,
                             int dim, const float* embeddings,
                             const float* binarized, double min_active_ratio,
                             std::vector<int>& chunk_idx,
                             std::vector<int>& spk_idx,
                             Eigen::MatrixXd& train) {
  chunk_idx.clear();
  spk_idx.clear();
  const double thresh = min_active_ratio * static_cast<double>(num_frames);
  for (int c = 0; c < num_chunks; ++c) {
    for (int s = 0; s < num_speakers; ++s) {
      double clean = 0.0;
      bool nan_emb = false;
      for (int f = 0; f < num_frames; ++f) {
        const float* row = &binarized[((c * num_frames) + f) * num_speakers];
        double row_sum = 0.0;
        for (int k = 0; k < num_speakers; ++k) {
          row_sum += static_cast<double>(row[k]);
        }
        // Match ``(segmentations.sum(axis=2, keepdims=True) == 1)`` on
        // binarized floats.
        if (row_sum != 1.0) {
          continue;
        }
        clean += static_cast<double>(row[s]);
      }
      for (int t = 0; t < dim; ++t) {
        const float v = embeddings[((c * num_speakers) + s) * dim + t];
        if (std::isnan(v)) {
          nan_emb = true;
        }
      }
      if (clean >= thresh && !nan_emb) {
        chunk_idx.push_back(c);
        spk_idx.push_back(s);
      }
    }
  }
  const int T = static_cast<int>(chunk_idx.size());
  train.resize(T, dim);
  for (int i = 0; i < T; ++i) {
    const int c = chunk_idx[static_cast<std::size_t>(i)];
    const int s = spk_idx[static_cast<std::size_t>(i)];
    for (int t = 0; t < dim; ++t) {
      train(i, t) =
          static_cast<double>(embeddings[((c * num_speakers) + s) * dim + t]);
    }
  }
}

}  // namespace cppannote::filter_train
