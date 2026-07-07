// SPDX-License-Identifier: MIT

#include "scipy_linkage.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace cppannote::scipy_linkage {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

inline double centroid_update(double d_xi, double d_yi, double d_xy, int size_x,
                              int size_y, int /*size_i*/) {
  const double sx = static_cast<double>(size_x);
  const double sy = static_cast<double>(size_y);
  const double num = (sx * d_xi * d_xi + sy * d_yi * d_yi) -
                     (sx * sy * d_xy * d_xy) / (sx + sy);
  const double den = sx + sy;
  if (!(num >= 0.0) || !(den > 0.0)) {
    return 0.0;
  }
  return std::sqrt(num / den);
}

// SciPy ``fcluster(..., criterion='distance')`` uses
// ``get_max_dist_for_each_cluster`` then
// ``cluster_monocrit`` (see ``scipy/cluster/_hierarchy.pyx``), not pairwise
// cophenetic union.

inline bool is_visited(const std::vector<unsigned char>& bitset, int i) {
  return (bitset[static_cast<std::size_t>(i >> 3)] &
          static_cast<unsigned char>(1U << (i & 7))) != 0;
}

inline void set_visited(std::vector<unsigned char>& bitset, int i) {
  bitset[static_cast<std::size_t>(i >> 3)] |=
      static_cast<unsigned char>(1U << (i & 7));
}

inline int visited_bytes(int n) { return (((n * 2) - 1) >> 3) + 1; }

void get_max_dist_for_each_cluster(const double* Z, int n,
                                   std::vector<double>& MD) {
  MD.assign(static_cast<std::size_t>(n), 0.0);
  std::vector<int> curr_node(static_cast<std::size_t>(n));
  std::vector<unsigned char> visited(static_cast<std::size_t>(visited_bytes(n)),
                                     0);
  auto Zat = [&](int row, int col) -> double {
    return Z[static_cast<std::size_t>(row) * 4 + static_cast<std::size_t>(col)];
  };
  int k = 0;
  curr_node[0] = 2 * n - 2;
  while (k >= 0) {
    const int root = curr_node[static_cast<std::size_t>(k)] - n;
    const int i_lc = static_cast<int>(Zat(root, 0));
    const int i_rc = static_cast<int>(Zat(root, 1));

    if (i_lc >= n && !is_visited(visited, i_lc)) {
      set_visited(visited, i_lc);
      ++k;
      curr_node[static_cast<std::size_t>(k)] = i_lc;
      continue;
    }
    if (i_rc >= n && !is_visited(visited, i_rc)) {
      set_visited(visited, i_rc);
      ++k;
      curr_node[static_cast<std::size_t>(k)] = i_rc;
      continue;
    }

    double max_dist = Zat(root, 2);
    if (i_lc >= n) {
      const double max_l = MD[static_cast<std::size_t>(i_lc - n)];
      if (max_l > max_dist) {
        max_dist = max_l;
      }
    }
    if (i_rc >= n) {
      const double max_r = MD[static_cast<std::size_t>(i_rc - n)];
      if (max_r > max_dist) {
        max_dist = max_r;
      }
    }
    MD[static_cast<std::size_t>(root)] = max_dist;
    --k;
  }
}

void cluster_monocrit(const double* Z, int n, const std::vector<double>& MC,
                      double cutoff, std::vector<int>& T) {
  T.assign(static_cast<std::size_t>(n), 0);
  std::vector<int> curr_node(static_cast<std::size_t>(n));
  std::vector<unsigned char> visited(static_cast<std::size_t>(visited_bytes(n)),
                                     0);
  auto Zat = [&](int row, int col) -> double {
    return Z[static_cast<std::size_t>(row) * 4 + static_cast<std::size_t>(col)];
  };
  int n_cluster = 0;
  int cluster_leader = -1;
  int k = 0;
  curr_node[0] = 2 * n - 2;
  while (k >= 0) {
    const int root = curr_node[static_cast<std::size_t>(k)] - n;
    const int i_lc = static_cast<int>(Zat(root, 0));
    const int i_rc = static_cast<int>(Zat(root, 1));

    if (cluster_leader == -1 && MC[static_cast<std::size_t>(root)] <= cutoff) {
      cluster_leader = root;
      ++n_cluster;
    }

    if (i_lc >= n && !is_visited(visited, i_lc)) {
      set_visited(visited, i_lc);
      ++k;
      curr_node[static_cast<std::size_t>(k)] = i_lc;
      continue;
    }
    if (i_rc >= n && !is_visited(visited, i_rc)) {
      set_visited(visited, i_rc);
      ++k;
      curr_node[static_cast<std::size_t>(k)] = i_rc;
      continue;
    }

    if (i_lc < n) {
      if (cluster_leader == -1) {
        ++n_cluster;
      }
      T[static_cast<std::size_t>(i_lc)] = n_cluster;
    }
    if (i_rc < n) {
      if (cluster_leader == -1) {
        ++n_cluster;
      }
      T[static_cast<std::size_t>(i_rc)] = n_cluster;
    }

    if (cluster_leader == root) {
      cluster_leader = -1;
    }
    --k;
  }
}

}  // namespace

void pdist_euclidean(const std::vector<double>& X, int n, int d,
                     std::vector<double>& dist) {
  const std::size_t m = static_cast<std::size_t>(n * (n - 1) / 2);
  dist.assign(m, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = i + 1; j < n; ++j) {
      double s = 0.0;
      for (int t = 0; t < d; ++t) {
        const double a = X[static_cast<std::size_t>(i * d + t)];
        const double b = X[static_cast<std::size_t>(j * d + t)];
        const double di = a - b;
        s += di * di;
      }
      dist[condensed_index(n, i, j)] = std::sqrt(s);
    }
  }
}

void linkage_centroid_naive(const std::vector<double>& dist, int n,
                            std::vector<double>& Z) {
  Z.assign(static_cast<std::size_t>(n - 1) * 4, 0.0);
  std::vector<double> D = dist;
  std::vector<int> id_map(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    id_map[static_cast<std::size_t>(i)] = i;
  }

  for (int merge = 0; merge < n - 1; ++merge) {
    double current_min = kInf;
    int x = 0;
    int y = 1;
    for (int i = 0; i < n - 1; ++i) {
      if (id_map[static_cast<std::size_t>(i)] == -1) {
        continue;
      }
      const std::size_t i_start = condensed_index(n, i, i + 1);
      for (int j = 0; j < n - i - 1; ++j) {
        const double v = D[i_start + static_cast<std::size_t>(j)];
        if (v < current_min) {
          current_min = v;
          x = i;
          y = i + j + 1;
        }
      }
    }

    const int id_x = id_map[static_cast<std::size_t>(x)];
    const int id_y = id_map[static_cast<std::size_t>(y)];
    int nx = 1;
    if (id_x >= n) {
      nx = static_cast<int>(Z[static_cast<std::size_t>(id_x - n) * 4 + 3]);
    }
    int ny = 1;
    if (id_y >= n) {
      ny = static_cast<int>(Z[static_cast<std::size_t>(id_y - n) * 4 + 3]);
    }

    Z[static_cast<std::size_t>(merge) * 4 + 0] =
        static_cast<double>(std::min(id_x, id_y));
    Z[static_cast<std::size_t>(merge) * 4 + 1] =
        static_cast<double>(std::max(id_x, id_y));
    Z[static_cast<std::size_t>(merge) * 4 + 2] = current_min;
    Z[static_cast<std::size_t>(merge) * 4 + 3] = static_cast<double>(nx + ny);

    id_map[static_cast<std::size_t>(x)] = -1;
    id_map[static_cast<std::size_t>(y)] = n + merge;

    for (int i = 0; i < n; ++i) {
      const int id_i = id_map[static_cast<std::size_t>(i)];
      if (id_i == -1 || id_i == n + merge) {
        continue;
      }
      int ni = 1;
      if (id_i >= n) {
        ni = static_cast<int>(Z[static_cast<std::size_t>(id_i - n) * 4 + 3]);
      }
      D[condensed_index(n, i, y)] =
          centroid_update(D[condensed_index(n, i, x)],
                          D[condensed_index(n, i, y)], current_min, nx, ny, ni);
      if (i < x) {
        D[condensed_index(n, i, x)] = kInf;
      }
    }
  }
}

void fcluster_distance(const std::vector<double>& Z, int n, double cutoff,
                       std::vector<int>& T) {
  std::vector<double> MD;
  get_max_dist_for_each_cluster(Z.data(), n, MD);
  cluster_monocrit(Z.data(), n, MD, cutoff, T);
}

void remap_labels_contiguous(const std::vector<int>& labels_one_based,
                             std::vector<int>& out) {
  const int n = static_cast<int>(labels_one_based.size());
  std::vector<int> z(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    z[static_cast<std::size_t>(i)] =
        labels_one_based[static_cast<std::size_t>(i)] - 1;
  }
  std::vector<int> uniq = z;
  std::sort(uniq.begin(), uniq.end());
  uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
  out.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int v = z[static_cast<std::size_t>(i)];
    auto it = std::lower_bound(uniq.begin(), uniq.end(), v);
    out[static_cast<std::size_t>(i)] = static_cast<int>(it - uniq.begin());
  }
}

}  // namespace cppannote::scipy_linkage
