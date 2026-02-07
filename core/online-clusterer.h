#ifndef ONLINE_CLUSTERER_H
#define ONLINE_CLUSTERER_H

#include <cstdint>
#include <cstddef>

#include <map>
#include <vector>

struct OnlineClustererOptions {
  size_t embedding_size = 512;
  float threshold = 0.8f;
};

struct Cluster {
  uint64_t id;
  std::vector<float> centroid;
  size_t sample_count = 0;
};

class OnlineClusterer {
  std::map<uint64_t, Cluster> clusters;
  OnlineClustererOptions options;

  uint64_t previous_cluster_id = 0;
  bool has_previous_cluster = false;

 public:
  OnlineClusterer(const OnlineClustererOptions &options);
  ~OnlineClusterer();
  uint64_t embed_and_cluster(const std::vector<float> &embedding,
                             float audio_duration);
};

#endif