#ifndef COSINE_DISTANCE_H_
#define COSINE_DISTANCE_H_

#include <vector>

// Computes cosine distance between two vectors: 1 - (a·b)/(||a||*||b||).
// Matches scipy.spatial.distance.cdist(..., metric="cosine")[0,0].
// Throws std::invalid_argument if a.size() != b.size().
float cosine_distance(const std::vector<float>& a, const std::vector<float>& b);

#endif  // COSINE_DISTANCE_H_
