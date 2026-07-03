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
    const auto names =
        ort_parse_provider_names("CoreMLExecutionProvider,CPUExecutionProvider");
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

#if defined(__APPLE__)
  SUBCASE("coreml provider appends successfully") {
    OrtStatus *status =
        ort_append_execution_providers(api, opts, {"coreml"}, nullptr);
    CHECK(status == nullptr);
  }
#endif

#if defined(__ANDROID__)
  SUBCASE("nnapi provider appends successfully") {
    OrtStatus *status =
        ort_append_execution_providers(api, opts, {"nnapi"}, nullptr);
    CHECK(status == nullptr);
  }
#endif

  api->ReleaseSessionOptions(opts);
  api->ReleaseEnv(env);
}

TEST_CASE("ort session with execution providers") {
  const char *model_path = "../test-assets/tiny-en/encoder_model.ort";
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
    OrtStatus *status = api->CreateSession(env, model_path, opts, &session);
    CHECK(status == nullptr);
    if (session != nullptr) {
      api->ReleaseSession(session);
    }
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    api->ReleaseSessionOptions(opts);
  }

#if defined(__APPLE__)
  SUBCASE("coreml session creation") {
    OrtSessionOptions *opts = nullptr;
    REQUIRE(api->CreateSessionOptions(&opts) == nullptr);
    REQUIRE(ort_append_execution_providers(api, opts, {"coreml", "cpu"},
                                           nullptr) == nullptr);
    OrtSession *session = nullptr;
    OrtStatus *status = api->CreateSession(env, model_path, opts, &session);
    CHECK(status == nullptr);
    if (session != nullptr) {
      api->ReleaseSession(session);
    }
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    api->ReleaseSessionOptions(opts);
  }
#endif

#if defined(__ANDROID__)
  SUBCASE("nnapi session creation") {
    OrtSessionOptions *opts = nullptr;
    REQUIRE(api->CreateSessionOptions(&opts) == nullptr);
    REQUIRE(ort_append_execution_providers(api, opts, {"nnapi", "cpu"},
                                           nullptr) == nullptr);
    OrtSession *session = nullptr;
    OrtStatus *status = api->CreateSession(env, model_path, opts, &session);
    CHECK(status == nullptr);
    if (session != nullptr) {
      api->ReleaseSession(session);
    }
    if (status != nullptr) {
      api->ReleaseStatus(status);
    }
    api->ReleaseSessionOptions(opts);
  }
#endif

  api->ReleaseEnv(env);
}
