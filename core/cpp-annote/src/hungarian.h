// SPDX-License-Identifier: MIT
// Min-cost assignment for a rectangular cost matrix (rows <= cols), CC0-style
// Hungarian.

#ifndef HUNGARIAN_H_
#define HUNGARIAN_H_

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cppannote::hungarian {

inline std::pair<double, std::vector<int>> min_cost_assignment(
    const std::vector<std::vector<double>>& cost) {
  const int n = static_cast<int>(cost.size());
  if (n == 0) {
    return {0.0, {}};
  }
  const int m = static_cast<int>(cost[0].size());
  if (m < n) {
    throw std::runtime_error("hungarian: need cols >= rows");
  }
  const int big_n = n + 1;
  const int big_m = m + 1;
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<double> u(big_n, 0.0);
  std::vector<double> v(big_m, 0.0);
  std::vector<int> p(big_m, 0);
  std::vector<int> way(big_m, 0);
  for (int i = 1; i < big_n; ++i) {
    p[0] = i;
    int j0 = 0;
    std::vector<double> minv(big_m, inf);
    std::vector<char> used(big_m, 0);
    do {
      used[static_cast<std::size_t>(j0)] = 1;
      const int i0 = p[static_cast<std::size_t>(j0)];
      double delta = inf;
      int j1 = 0;
      for (int j = 1; j < big_m; ++j) {
        if (!used[static_cast<std::size_t>(j)]) {
          const double cur = cost[static_cast<std::size_t>(i0 - 1)]
                                 [static_cast<std::size_t>(j - 1)] -
                             u[static_cast<std::size_t>(i0)] -
                             v[static_cast<std::size_t>(j)];
          if (cur < minv[static_cast<std::size_t>(j)]) {
            minv[static_cast<std::size_t>(j)] = cur;
            way[static_cast<std::size_t>(j)] = j0;
          }
          if (minv[static_cast<std::size_t>(j)] < delta) {
            delta = minv[static_cast<std::size_t>(j)];
            j1 = j;
          }
        }
      }
      for (int j = 0; j < big_m; ++j) {
        if (used[static_cast<std::size_t>(j)]) {
          u[static_cast<std::size_t>(p[static_cast<std::size_t>(j)])] += delta;
          v[static_cast<std::size_t>(j)] -= delta;
        } else {
          minv[static_cast<std::size_t>(j)] -= delta;
        }
      }
      j0 = j1;
    } while (p[static_cast<std::size_t>(j0)] != 0);
    do {
      const int j1 = way[static_cast<std::size_t>(j0)];
      p[static_cast<std::size_t>(j0)] = p[static_cast<std::size_t>(j1)];
      j0 = j1;
    } while (j0);
  }
  std::vector<int> assignment(static_cast<std::size_t>(n), -1);
  for (int j = 1; j < big_m; ++j) {
    if (p[static_cast<std::size_t>(j)] != 0) {
      assignment[static_cast<std::size_t>(p[static_cast<std::size_t>(j)] - 1)] =
          j - 1;
    }
  }
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    const int j = assignment[static_cast<std::size_t>(i)];
    if (j >= 0) {
      total += cost[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    }
  }
  return {-v[0], assignment};
}

}  // namespace cppannote::hungarian

#endif  // HUNGARIAN_H_
