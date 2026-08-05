#include "clone-clip.h"

#include <string>
#include <vector>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

TEST_CASE("clone-clip-refine") {
  SUBCASE("extends past requested duration to finish the last word") {
    const std::vector<CloneClipWord> words = {
        {"Ever", 0.10f, 0.40f},
        {"tried", 0.45f, 0.90f},
        {"Ever", 1.00f, 1.30f},
        {"failed", 1.35f, 3.90f},  // ends after the 3.5s window
    };
    const CloneClipBounds bounds =
        refine_clone_clip_bounds(/*window_start=*/0.0f, /*duration=*/3.5f,
                                 words, /*max_extension=*/1.5f,
                                 /*end_pad=*/0.05f);
    CHECK(bounds.start_seconds == doctest::Approx(0.10f));
    CHECK(bounds.end_seconds == doctest::Approx(3.95f));
    CHECK(bounds.transcript == "Ever tried Ever failed");
  }

  SUBCASE("drops a trailing word that exceeds the extension budget") {
    const std::vector<CloneClipWord> words = {
        {"one", 0.0f, 0.4f},
        {"two", 0.5f, 1.0f},
        {"three", 1.1f, 5.5f},  // would need >1.5s past a 4s window
    };
    const CloneClipBounds bounds =
        refine_clone_clip_bounds(0.0f, 4.0f, words, 1.5f, 0.05f);
    CHECK(bounds.start_seconds == doctest::Approx(0.0f));
    CHECK(bounds.end_seconds == doctest::Approx(1.05f));
    CHECK(bounds.transcript == "one two");
  }

  SUBCASE("drops a partial leading word that started before the window") {
    const std::vector<CloneClipWord> words = {
        {"leading", 0.0f, 0.8f},  // starts before window at 0.5
        {"kept", 0.9f, 1.4f},
        {"words", 1.5f, 2.0f},
    };
    const CloneClipBounds bounds =
        refine_clone_clip_bounds(0.5f, 2.0f, words, 1.5f, 0.05f);
    CHECK(bounds.start_seconds == doctest::Approx(0.9f));
    CHECK(bounds.end_seconds == doctest::Approx(2.05f));
    CHECK(bounds.transcript == "kept words");
  }

  SUBCASE("falls back to the exact VAD window when there are no words") {
    const CloneClipBounds bounds =
        refine_clone_clip_bounds(1.25f, 4.0f, {}, 1.5f, 0.05f);
    CHECK(bounds.start_seconds == doctest::Approx(1.25f));
    CHECK(bounds.end_seconds == doctest::Approx(5.25f));
    CHECK(bounds.transcript.empty());
  }

  SUBCASE("ignores words that begin at or after the requested end") {
    const std::vector<CloneClipWord> words = {
        {"inside", 0.2f, 0.8f},
        {"after", 4.1f, 4.6f},  // starts past a 4s window
    };
    const CloneClipBounds bounds =
        refine_clone_clip_bounds(0.0f, 4.0f, words, 1.5f, 0.05f);
    CHECK(bounds.transcript == "inside");
    CHECK(bounds.end_seconds == doctest::Approx(0.85f));
  }

  SUBCASE("extends to finish a word that overlaps past the requested window") {
    // Last in-window word ends at 3.9; the next word overlaps it and ends past
    // the 4.0s window. Refine should extend to finish that next word.
    const std::vector<CloneClipWord> words = {
        {"times", 3.0f, 3.9f},
        {"It", 3.7f, 4.4f},
    };
    const CloneClipBounds bounds =
        refine_clone_clip_bounds(0.0f, 4.0f, words, 1.5f, 0.05f);
    CHECK(bounds.end_seconds == doctest::Approx(4.45f));
    CHECK(bounds.transcript == "times It");
  }
}
