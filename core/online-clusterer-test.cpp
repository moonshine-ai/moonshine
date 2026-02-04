#include "online-clusterer.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("online-clusterer") {
    SUBCASE("embed_and_cluster") {
        OnlineClustererOptions options;
        options.embedding_size = 3;
        options.threshold = 0.8f;
        OnlineClusterer clusterer(options);
        const uint64_t first_cluster_id = clusterer.embed_and_cluster({1.0f, 2.0f, 3.0f}, 5.0f);
        const uint64_t second_cluster_id = clusterer.embed_and_cluster({0.0f, -1.0f, -2.0f}, 5.0f);
        const uint64_t third_cluster_id = clusterer.embed_and_cluster({2.0f, 4.0f, 6.0f}, 5.0f);
        const uint64_t fourth_cluster_id = clusterer.embed_and_cluster({1.0f, -1.0f, 0.0f}, 5.0f);
        CHECK(first_cluster_id != second_cluster_id);
        CHECK(first_cluster_id != fourth_cluster_id);
        CHECK(first_cluster_id == third_cluster_id);
        CHECK(third_cluster_id != fourth_cluster_id);
    }
}