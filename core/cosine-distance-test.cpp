#include "cosine-distance.h"

#include <stdexcept>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("cosine-distance") {
  SUBCASE("identical vectors give zero distance") {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    CHECK(cosine_distance(a, b) == doctest::Approx(0.0f));
  }

  SUBCASE("orthogonal vectors give distance one") {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};
    CHECK(cosine_distance(a, b) == doctest::Approx(1.0f));
  }

  SUBCASE("opposite vectors give distance two") {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {-1.0f, 0.0f, 0.0f};
    CHECK(cosine_distance(a, b) == doctest::Approx(2.0f));
  }

  SUBCASE("mismatched length throws") {
    std::vector<float> a = {1.0f, 2.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    CHECK_THROWS_AS(cosine_distance(a, b), std::invalid_argument);
  }

  SUBCASE("zero vector gives zero distance") {
    std::vector<float> a = {0.0f, 0.0f, 0.0f};
    std::vector<float> b = {1.0f, 2.0f, 3.0f};
    CHECK(cosine_distance(a, b) == doctest::Approx(0.0f));
  }

  SUBCASE("matches scipy implementation") {
    // Values created using:
    // from scipy.spatial.distance import cdist
    // a = np.random.uniform(-10.0, 10.0, (1, 10))
    // b = np.random.uniform(-10.0, 10.0, (1, 10))
    // distance = cdist(a, b, metric="cosine")[0,0]
    std::vector<float> a = {
        4.06762777f, -6.04896662f, -1.49120807f, -0.82805242f, -2.61263022,
        3.86128271f, -6.88300617f, -1.15056214f, -7.53303174f, -2.55405438f};
    std::vector<float> b = {
        -8.83947805f, 7.62308151f, -2.37085764f, -7.13739351f, 9.36701334f,
        -3.04214464f, 9.59137477f, 7.63300308f,  -8.06447383f, -6.8891267f};
    const float actual_distance = cosine_distance(a, b);
    const float expected_distance = 1.325184768493097f;
    REQUIRE(actual_distance == doctest::Approx(expected_distance));
  }
}
