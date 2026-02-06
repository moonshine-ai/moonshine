#include "online-clusterer.h"

#include <random>
#include <stdexcept>

#include "cosine-distance.h"
#include "debug-utils.h"

// Implementation of the Sequential Leader Algorithm with centroid updating.
//
// This is a streaming algorithm that can be used to cluster speaker embeddings
// in real-time. as follows:
//
// 1) For every speech segment coming in, an embedding is generated.
//
// 2) For the first segment in a session, that embedding is marked as speaker 1.
//
// 3) When the next segment comes in, I calculate a distance measure from the
// first embedding for speaker 1.
//
// 4) If the distance is below a threshold, I
// append the embedding to a list for speaker 1.
//
// 5) If the distance is above a
// threshold, I mark the segment as speaker 2, and start a list for that speaker
// with the current embedding as a single member.
//
// 6) As more segments arrive, I loop through the known speakers and calculate
// distances between the new embedding and the average of the list of embeddings
// collected for each speaker. This average is effectively the centroid of the
// cluster associated with each speaker.
//
// 7) As before, if the new embedding's distance from the average of a speaker
// cluster is below a threshold with any speakers, we choose the closest cluster
// as the speaker of the line.
//
// 8) If it's not close enough to any existing cluster's centroids, it's
// assigned to a new speaker.
//
// This approach was chosen because it is a simple and efficient way to cluster
// embeddings that arrive in a streaming fashion. Other algorithms, such as
// DBSCAN, can provide more accurate clustering, but at the cost of increased
// complexity and computational overhead.
//
// Strengths:
//
// - O(n) time complexity, single pass through data
// - Low memory footprint (only store centroids + counts)
// - Simple to implement and debug
// - Works well when clusters are reasonably well-separated
//
// Weaknesses:
//
// - Order-dependent: The clusters you get depend heavily on which points arrive
// first.
// - Threshold sensitivity: The threshold is doing a lot of work. Too tight and
// it'll over-segment (same speaker becomes multiple); too loose and it'll
// under-segment.
// - No cluster merging: If it accidentally creates two clusters for the same
// speaker early on, they'll never merge.
// - Centroid drift: As it adds points, the centroid moves, which can cause
// inconsistent assignment decisions over time.
//
// Update 1: Added a threshold scaling factor to the threshold so that short
// segments are placed in the nearest cluster instead of creating a new one.
// Dropped confusion from 30.67% to 26.44% (using scripts/eval-speaker-id.py).
//
// Update 2: Added a previous cluster bias to the distance calculation so that
// the speaker identified for the previous segment is given more weight than
// others. Dropped DER from 26.44% to

OnlineClusterer::OnlineClusterer(const OnlineClustererOptions &options)
    : options(options) {}

OnlineClusterer::~OnlineClusterer() {}

uint64_t OnlineClusterer::embed_and_cluster(const std::vector<float> &embedding,
                                            float audio_duration) {
  if (embedding.size() != options.embedding_size) {
    throw std::invalid_argument("embedding size " +
                                std::to_string(embedding.size()) +
                                " must match the options embedding size " +
                                std::to_string(options.embedding_size));
  }
  // Find the cluster that is closest to the embedding
  float min_distance = std::numeric_limits<float>::max();
  uint64_t closest_cluster_id = 0;
  bool found_cluster = false;
  for (const auto &cluster : clusters) {
    float distance = cosine_distance(embedding, cluster.second.centroid);
    if (distance < min_distance) {
      min_distance = distance;
      closest_cluster_id = cluster.first;
      found_cluster = true;
    }
  }
  // Linearly scale the threshold so that segments shorter than 2 seconds
  // are placed in the nearest cluster instead of creating a new one, and
  // between 2 and 3 seconds are subject to a proportional threshold.
  constexpr float scale_min = 2.0f;
  constexpr float scale_max = 3.0f;
  constexpr float duration_min = 1.0f;
  constexpr float scale_range = scale_max - scale_min;
  constexpr float threshold_max = 1.5f;
  float current_threshold;
  if (audio_duration > scale_max) {
    current_threshold = options.threshold;
  } else if (audio_duration > scale_min) {
    const float scale_factor = (audio_duration - scale_min) / scale_range;
    current_threshold = (options.threshold * scale_factor) +
                        (threshold_max * (1.0f - scale_factor));
  } else if (audio_duration > duration_min) {
    current_threshold = threshold_max;
  } else if (has_previous_cluster) {
    return previous_cluster_id;
  } else {
    current_threshold = threshold_max;
  }
  uint64_t result_cluster_id = 0;
  if (found_cluster && min_distance < current_threshold) {
    // If a cluster is found, update the centroid and sample count.
    Cluster &cluster = clusters[closest_cluster_id];
    const size_t n = cluster.sample_count;
    const float scale_old = static_cast<float>(n) / static_cast<float>(n + 1);
    const float scale_new = 1.0f / static_cast<float>(n + 1);
    for (size_t i = 0; i < cluster.centroid.size(); ++i) {
      cluster.centroid[i] =
          scale_old * cluster.centroid[i] + scale_new * embedding[i];
    }
    cluster.sample_count++;
    result_cluster_id = closest_cluster_id;
  } else {
    // If no cluster is found, create a new cluster with a random id.
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist;
    uint64_t new_cluster_id = dist(rng);
    clusters[new_cluster_id] = {new_cluster_id, embedding, 1};
    result_cluster_id = new_cluster_id;
  }
  previous_cluster_id = result_cluster_id;
  has_previous_cluster = true;
  return result_cluster_id;
}
