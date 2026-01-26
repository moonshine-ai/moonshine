#include "string-utils.h"

#include <cstdio>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("string-utils") {
  SUBCASE("replace_all") {
    CHECK(replace_all("hello world", "world", "hello") == "hello hello");
  }
  SUBCASE("trim") { CHECK(trim("  hello world  ") == "hello world"); }
  SUBCASE("split") {
    CHECK(split("hello world", " ") ==
          std::vector<std::string>{"hello", "world"});
  }
  SUBCASE("starts_with") { CHECK(starts_with("hello world", "hello") == true); }
  SUBCASE("starts_with_invalid") {
    CHECK(starts_with("hello world", "world") == false);
  }
  SUBCASE("ends_with") { CHECK(ends_with("hello world", "world") == true); }
  SUBCASE("ends_with_invalid") {
    CHECK(ends_with("hello world", "hello") == false);
  }
  SUBCASE("name_to_index") {
    CHECK(name_to_index({"hello", "world"}) ==
          std::map<std::string, int64_t>{{"hello", 0}, {"world", 1}});
  }
  SUBCASE("append_path_component") {
    CHECK(append_path_component("hello", "world") == "hello/world");
    CHECK(append_path_component("hello/", "world") == "hello/world");
    CHECK(append_path_component("hello", "/world") == "hello/world");
    CHECK(append_path_component("hello/", "/world") == "hello/world");
  }
  SUBCASE("to_lowercase") {
    CHECK(to_lowercase("Hello World") == "hello world");
    CHECK(to_lowercase("123") == "123");
    CHECK(to_lowercase("abc") == "abc");
  }
  SUBCASE("bool_from_string") {
    CHECK(bool_from_string("true") == true);
    CHECK(bool_from_string("false") == false);
    CHECK(bool_from_string("1") == true);
    CHECK(bool_from_string("0") == false);
  }
  SUBCASE("bool_from_string_invalid") {
    CHECK_THROWS(bool_from_string("invalid"));
    CHECK_THROWS(bool_from_string(""));
    CHECK_THROWS(bool_from_string(nullptr));
  }
  SUBCASE("float_from_string") {
    CHECK(float_from_string("1.0") == doctest::Approx(1.0f));
    CHECK(float_from_string("2027.89") == doctest::Approx(2027.89f));
    CHECK(float_from_string("0.0000001") == doctest::Approx(0.0000001f));
  }
  SUBCASE("float_from_string_invalid") {
    CHECK_THROWS(float_from_string("invalid"));
    CHECK_THROWS(float_from_string(""));
    CHECK_THROWS(float_from_string(nullptr));
  }
  SUBCASE("int32_from_string") {
    CHECK(int32_from_string("1") == 1);
    CHECK(int32_from_string("2027") == 2027);
    CHECK(int32_from_string("0") == 0);
  }
  SUBCASE("int32_from_string_invalid") {
    CHECK_THROWS(int32_from_string("invalid"));
    CHECK_THROWS(int32_from_string(""));
    CHECK_THROWS(int32_from_string(nullptr));
  }
  SUBCASE("size_t_from_string") {
    CHECK(size_t_from_string("1") == 1);
    CHECK(size_t_from_string("2027") == 2027);
    CHECK(size_t_from_string("0") == 0);
  }
  SUBCASE("size_t_from_string_invalid") {
    CHECK_THROWS(size_t_from_string("invalid"));
    CHECK_THROWS(size_t_from_string(""));
    CHECK_THROWS(size_t_from_string(nullptr));
  }
}