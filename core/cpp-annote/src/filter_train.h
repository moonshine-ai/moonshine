// SPDX-License-Identifier: MIT
// ``VBxClustering.filter_embeddings`` (same rules as ``clustering.py``).

#ifndef FILTER_TRAIN_H_
#define FILTER_TRAIN_H_

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

namespace cppannote::filter_train {

/// Row-major ``embeddings`` length ``num_chunks * num_speakers * dim``;
/// ``binarized`` length ``num_chunks * num_frames * num_speakers``.
void filter_embeddings_train(int num_chunks, int num_frames, int num_speakers,
                             int dim, const float* embeddings,
                             const float* binarized, double min_active_ratio,
                             std::vector<int>& chunk_idx,
                             std::vector<int>& spk_idx, Eigen::MatrixXd& train);

}  // namespace cppannote::filter_train

#endif  // FILTER_TRAIN_H_
