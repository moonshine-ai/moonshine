#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "heteronym-context.h"
#include "utf8-utils.h"

using namespace moonshine_tts;

TEST_CASE("heteronym_centered_context_window_cells short pad") {
  const auto cells = utf8_split_codepoints("ab");
  const auto w = heteronym_centered_context_window_cells(cells, 0, 2, 4);
  REQUIRE(w);
  const auto& [text, ws, we] = *w;
  CHECK(text.size() == 4);
  CHECK(ws == 1);
  CHECK(we == 3);
}

TEST_CASE("heteronym_centered_context_window_cells crop") {
  std::string s;
  for (int i = 0; i < 34; ++i) {
    s.push_back(static_cast<char>('a' + (i % 26)));
  }
  const auto cells = utf8_split_codepoints(s);
  REQUIRE(static_cast<int>(cells.size()) == 34);
  const auto w = heteronym_centered_context_window_cells(cells, 30, 32, 32);
  REQUIRE(w);
  const auto& [text, ws, we] = *w;
  CHECK(utf8_split_codepoints(text).size() == 32);
  CHECK(ws >= 0);
  CHECK(we <= 32);
  CHECK(we > ws);
}
