#include "ort-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

TEST_CASE("ort-utils") {
  SUBCASE("ort_session_from_path") {
    REQUIRE(ort_session_from_path(nullptr, nullptr, nullptr, "model.onnx",
                                  nullptr, nullptr, nullptr) < 0);
  }
  SUBCASE("ort_session_from_memory") {
    REQUIRE(ort_session_from_memory(nullptr, nullptr, nullptr, nullptr, 0,
                                    nullptr) < 0);
  }
}