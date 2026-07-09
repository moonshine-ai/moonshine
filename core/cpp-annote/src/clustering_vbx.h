// SPDX-License-Identifier: MIT
// ``VBxClustering.__call__`` (filter → AHC → PLDA → VBx → optional KMeans →
// cdist → assignment).

#ifndef CLUSTERING_VBX_H_
#define CLUSTERING_VBX_H_

#include <cstdint>
#include <vector>

#include "plda_vbx.h"

namespace cppannote::clustering_vbx {

struct VbxClusteringParams {
  double threshold = 0.6;
  double Fa = 0.07;
  double Fb = 0.8;
  int lda_dimension = 128;
  int max_vbx_iters = 20;
  double init_smoothing = 7.0;
  double min_active_ratio = 0.2;
  bool constrained_assignment = true;
  /// ``metric`` from Python (``cosine`` only supported here).
  bool metric_is_cosine = true;
  int min_clusters = 1;
  int max_clusters = 1000000000;
  int num_clusters =
      -1;  // optional forced count (``num_speakers``); -1 = unset
};

/// ``embeddings`` row-major ``(num_chunks * num_speakers * dim)``;
/// ``binarized`` ``(num_chunks * num_frames * num_speakers)``.
void vbx_clustering_hard(const plda_vbx::PldaModel& plda,
                         const VbxClusteringParams& pr, int num_chunks,
                         int num_frames, int num_speakers, int dim,
                         const float* embeddings, const float* binarized,
                         std::vector<std::int8_t>& hard_clusters);

}  // namespace cppannote::clustering_vbx

#endif  // CLUSTERING_VBX_H_
