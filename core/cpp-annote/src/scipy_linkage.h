// SPDX-License-Identifier: MIT
// SciPy-compatible helpers: condensed pdist (Euclidean), naive centroid
// linkage, fcluster(..., criterion='distance').

#ifndef SCIPY_LINKAGE_H_
#define SCIPY_LINKAGE_H_

#include <cmath>
#include <cstddef>
#include <vector>

namespace cppannote::scipy_linkage {

inline std::size_t condensed_index(int n, int i, int j) {
  if (i > j) {
    std::swap(i, j);
  }
  return static_cast<std::size_t>(n * i - (i * (i + 1)) / 2 + (j - i - 1));
}

/// Row-major `X`: `n` rows, `d` cols → condensed pairwise Euclidean distances
/// (length n*(n-1)/2).
void pdist_euclidean(const std::vector<double>& X, int n, int d,
                     std::vector<double>& dist);

/// SciPy `linkage(..., method='centroid')` via the generic O(n³) SciPy C
/// implementation (small `n` only). `Z` is written as `(n-1)*4` row-major
/// doubles: for each merge: idx_a, idx_b, dist, count.
void linkage_centroid_naive(const std::vector<double>& dist, int n,
                            std::vector<double>& Z);

/// ``fcluster(Z, t, criterion='distance')`` → labels in **SciPy convention**
/// (1..K inclusive). Implemented like SciPy: ``get_max_dist_for_each_cluster``
/// + ``cluster_monocrit`` (not pairwise cophenetic union).
void fcluster_distance(const std::vector<double>& Z, int n, double cutoff,
                       std::vector<int>& T);

/// `np.unique(labels - 1, return_inverse=True)[1]` — contiguous 0..K-1.
void remap_labels_contiguous(const std::vector<int>& labels_one_based,
                             std::vector<int>& out);

}  // namespace cppannote::scipy_linkage

#endif  // SCIPY_LINKAGE_H_
