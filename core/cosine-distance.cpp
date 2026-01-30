#include "cosine-distance.h"

#include <cmath>
#include <stdexcept>

float cosine_distance(const std::vector<float>& a,
                      const std::vector<float>& b) {
  if (a.size() != b.size()) {
    throw std::invalid_argument(
        "cosine distance: vectors must have the same length");
  }

  float dot = 0.0f;
  float norm_a_sq = 0.0f;
  float norm_b_sq = 0.0f;

  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    norm_a_sq += a[i] * a[i];
    norm_b_sq += b[i] * b[i];
  }

  float norm_a = std::sqrt(norm_a_sq);
  float norm_b = std::sqrt(norm_b_sq);

  if (norm_a == 0.0f || norm_b == 0.0f) {
    return 0.0f;  // scipy returns 0 when either vector has zero norm
  }

  float similarity = dot / (norm_a * norm_b);
  return 1.0f - similarity;
}
