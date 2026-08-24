#include "moonshine-model-catalog.h"

#include <optional>
#include <string>

#include "moonshine-c-api.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest.h>

// The C API logs these strings through LOGF; capturing stderr is not portable,
// so the wording is checked here against the catalog helper the C API calls.

TEST_CASE("stt-missing-dependencies-message") {
  SUBCASE("unknown-language") {
    CHECK(moonshine::stt_missing_dependencies_message("zzz_not_a_language",
                                                      std::nullopt) ==
          "unknown language \"zzz_not_a_language\"");
    // An architecture does not change the diagnosis when the language itself
    // is unpublished.
    CHECK(moonshine::stt_missing_dependencies_message(
              "zzz_not_a_language", MOONSHINE_MODEL_ARCH_TINY) ==
          "unknown language \"zzz_not_a_language\"");
  }

  SUBCASE("unknown-arch-lists-supported") {
    // BASE_STREAMING is defined in the C API but unpublished for every
    // language; the message must not pretend the language is unknown.
    const std::string en = moonshine::stt_missing_dependencies_message(
        "en", MOONSHINE_MODEL_ARCH_BASE_STREAMING);
    CHECK(en.find("unknown language") == std::string::npos);
    CHECK(en.find("has no model_arch 3 (BASE_STREAMING)") != std::string::npos);
    CHECK(en.find("supported architectures:") != std::string::npos);
    CHECK(en.find("5 (MEDIUM_STREAMING)") != std::string::npos);
    CHECK(en.find("4 (SMALL_STREAMING)") != std::string::npos);
    CHECK(en.find("1 (BASE)") != std::string::npos);
    CHECK(en.find("2 (TINY_STREAMING)") != std::string::npos);
    CHECK(en.find("0 (TINY)") != std::string::npos);

    const std::string es = moonshine::stt_missing_dependencies_message(
        "es", MOONSHINE_MODEL_ARCH_BASE_STREAMING);
    CHECK(es.find("has no model_arch 3 (BASE_STREAMING)") != std::string::npos);
    CHECK(es.find("4 (SMALL_STREAMING)") != std::string::npos);
    CHECK(es.find("2 (TINY_STREAMING)") != std::string::npos);
    CHECK(es.find("1 (BASE)") != std::string::npos);
    // Spanish does not publish medium streaming; the list is per-language.
    CHECK(es.find("5 (MEDIUM_STREAMING)") == std::string::npos);

    CHECK(moonshine::stt_missing_dependencies_message("en", std::nullopt)
              .empty());
    CHECK(moonshine::stt_missing_dependencies_message("en",
                                                      MOONSHINE_MODEL_ARCH_TINY)
              .empty());
  }
}
