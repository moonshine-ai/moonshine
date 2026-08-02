#include "ort-utils.h"

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN

#include <doctest.h>

#include <filesystem>
#include <string>

TEST_CASE("ort_parse_provider_names") {
  SUBCASE("empty string") {
    CHECK(ort_parse_provider_names("").empty());
    CHECK(ort_parse_provider_names("   ").empty());
  }

  SUBCASE("comma-separated aliases") {
    const auto names = ort_parse_provider_names("CoreML, CPU");
    REQUIRE(names.size() == 2);
    CHECK(names[0] == "coreml");
    CHECK(names[1] == "cpu");
  }

  SUBCASE("execution provider suffix aliases") {
    const auto names = ort_parse_provider_names(
        "CoreMLExecutionProvider,CPUExecutionProvider");
    REQUIRE(names.size() == 2);
    CHECK(names[0] == "coreml");
    CHECK(names[1] == "cpu");
  }

  SUBCASE("empty token is rejected") {
    CHECK_THROWS_AS(ort_parse_provider_names("CoreML,,CPU"),
                    std::invalid_argument);
  }
}

TEST_CASE("ort_append_execution_providers") {
  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  REQUIRE(api != nullptr);

  OrtEnv *env = nullptr;
  REQUIRE(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "ort-utils-ep-test", &env) ==
          nullptr);

  OrtSessionOptions *opts = nullptr;
  REQUIRE(api->CreateSessionOptions(&opts) == nullptr);

  SUBCASE("unknown provider returns error status") {
    OrtStatus *status = ort_append_execution_providers(
        api, opts, {"not_a_real_provider"}, nullptr);
    REQUIRE(status != nullptr);
    CHECK(std::string(api->GetErrorMessage(status)).find("Unknown") !=
          std::string::npos);
    api->ReleaseStatus(status);
  }

  SUBCASE("cpu provider appends successfully") {
    OrtStatus *status =
        ort_append_execution_providers(api, opts, {"cpu"}, nullptr);
    CHECK(status == nullptr);
  }

  // A compiling provider either appends or is absent from this library, and
  // both are expected: macOS has CoreML, iOS and Android ship without either
  // (docs/execution-providers.md). What must hold in the absent case is that
  // the caller is told where to look rather than handed ORT's "not supported
  // in this build".
  //
  // Guarded by the same condition as its only callers below: on a platform
  // that is neither, the lambda is dead and GCC's -Wunused-but-set-variable
  // fails the build under -Werror.
#if defined(__APPLE__) || defined(__ANDROID__)
  auto check_optional_provider = [&](const char *name) {
    OrtStatus *status = ort_append_execution_providers(api, opts, {name},
                                                       nullptr);
    if (status == nullptr) {
      return;
    }
    const std::string message = api->GetErrorMessage(status);
    CHECK(message.find("docs/execution-providers.md") != std::string::npos);
    api->ReleaseStatus(status);
  };
#endif

#if defined(__APPLE__)
  SUBCASE("coreml provider appends or explains its absence") {
    check_optional_provider("coreml");
  }
#endif

#if defined(__ANDROID__)
  SUBCASE("nnapi provider appends or explains its absence") {
    check_optional_provider("nnapi");
  }
#endif

  api->ReleaseSessionOptions(opts);
  api->ReleaseEnv(env);
}

TEST_CASE("ort session with execution providers") {
  // path::c_str() yields ORTCHAR_T (wchar_t on Windows, char elsewhere), as
  // required by OrtApi::CreateSession.
  const std::filesystem::path model_path =
      "../test-assets/tiny-en/encoder_model.ort";
  if (!std::filesystem::exists(model_path)) {
    return;
  }

  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  REQUIRE(api != nullptr);

  OrtEnv *env = nullptr;
  REQUIRE(api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "ort-utils-ep-test", &env) ==
          nullptr);

  SUBCASE("cpu-only session creation") {
    OrtSessionOptions *opts = nullptr;
    REQUIRE(api->CreateSessionOptions(&opts) == nullptr);
    REQUIRE(ort_append_execution_providers(api, opts, {"cpu"}, nullptr) ==
            nullptr);
    OrtSession *session = nullptr;
    OrtStatus *status =
        api->CreateSession(env, model_path.c_str(), opts, &session);
    CHECK(status == nullptr);
    if (session != nullptr) {
      api->ReleaseSession(session);
    }
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    api->ReleaseSessionOptions(opts);
  }

  // Same reasoning as above, including the guard: run the session only where
  // the provider exists.
#if defined(__APPLE__) || defined(__ANDROID__)
  auto check_optional_provider_session = [&](const char *name) {
    OrtSessionOptions *opts = nullptr;
    REQUIRE(api->CreateSessionOptions(&opts) == nullptr);
    OrtStatus *appended =
        ort_append_execution_providers(api, opts, {name, "cpu"}, nullptr);
    if (appended != nullptr) {
      api->ReleaseStatus(appended);
      api->ReleaseSessionOptions(opts);
      return;
    }
    OrtSession *session = nullptr;
    OrtStatus *status =
        api->CreateSession(env, model_path.c_str(), opts, &session);
    CHECK(status == nullptr);
    if (session != nullptr) {
      api->ReleaseSession(session);
    }
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    api->ReleaseSessionOptions(opts);
  };
#endif

#if defined(__APPLE__)
  SUBCASE("coreml session creation") {
    check_optional_provider_session("coreml");
  }
#endif

#if defined(__ANDROID__)
  SUBCASE("nnapi session creation") { check_optional_provider_session("nnapi"); }
#endif

  api->ReleaseEnv(env);
}
