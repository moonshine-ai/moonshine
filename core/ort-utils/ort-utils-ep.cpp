#include "cpu_provider_factory.h"
#include "ort-utils.h"

#if defined(__ANDROID__)
#include "nnapi_provider_factory.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string trim_copy(const std::string &s) {
  size_t lo = 0;
  size_t hi = s.size();
  while (lo < hi && std::isspace(static_cast<unsigned char>(s[lo]))) {
    ++lo;
  }
  while (hi > lo && std::isspace(static_cast<unsigned char>(s[hi - 1]))) {
    --hi;
  }
  return s.substr(lo, hi - lo);
}

std::string lowercase_copy(std::string s) {
  for (char &c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return s;
}

std::string normalize_provider_name(const std::string &name) {
  const std::string lower = lowercase_copy(trim_copy(name));
  if (lower == "cpu" || lower == "cpuexecutionprovider") {
    return "cpu";
  }
  if (lower == "coreml" || lower == "coremlexecutionprovider") {
    return "coreml";
  }
  if (lower == "nnapi" || lower == "nnapiexecutionprovider" ||
      lower == "nnapiprovider") {
    return "nnapi";
  }
  return lower;
}

OrtStatus *make_invalid_argument_status(const OrtApi *ort_api,
                                        const std::string &message) {
  return ort_api->CreateStatus(ORT_INVALID_ARGUMENT, message.c_str());
}

OrtStatus *append_one_provider(const OrtApi *ort_api,
                               OrtSessionOptions *session_options,
                               const std::string &normalized,
                               const OrtExecutionProviderOptions *config) {
  if (normalized == "cpu") {
    return OrtSessionOptionsAppendExecutionProvider_CPU(session_options, 0);
  }

  if (normalized == "coreml") {
#if defined(__APPLE__)
    const char *provider_name = "CoreMLExecutionProvider";
    const char *cache_dir =
        (config != nullptr && config->coreml_cache_dir != nullptr &&
         config->coreml_cache_dir[0] != '\0')
            ? config->coreml_cache_dir
            : nullptr;
    if (cache_dir != nullptr) {
      const char *keys[] = {"ModelCacheDirectory"};
      const char *values[] = {cache_dir};
      return ort_api->SessionOptionsAppendExecutionProvider(
          session_options, provider_name, keys, values, 1);
    }
    return ort_api->SessionOptionsAppendExecutionProvider(
        session_options, provider_name, nullptr, nullptr, 0);
#else
    return make_invalid_argument_status(
        ort_api, "CoreML execution provider is not available on this platform");
#endif
  }

  if (normalized == "nnapi") {
#if defined(__ANDROID__)
    return OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options, 0);
#else
    return make_invalid_argument_status(
        ort_api, "NNAPI execution provider is not available on this platform");
#endif
  }

  std::ostringstream oss;
  oss << "Unknown ONNX Runtime execution provider: " << normalized;
  return make_invalid_argument_status(ort_api, oss.str());
}

}  // namespace

std::vector<std::string> ort_parse_provider_names(const std::string &csv) {
  std::vector<std::string> out;
  if (trim_copy(csv).empty()) {
    return out;
  }

  size_t start = 0;
  while (start <= csv.size()) {
    const size_t comma = csv.find(',', start);
    const std::string token = trim_copy(csv.substr(
        start, comma == std::string::npos ? std::string::npos : comma - start));
    if (token.empty()) {
      throw std::invalid_argument(
          "ort_providers contains an empty provider name");
    }
    out.push_back(normalize_provider_name(token));
    if (comma == std::string::npos) {
      break;
    }
    start = comma + 1;
  }
  return out;
}

OrtStatus *ort_append_execution_providers(
    const OrtApi *ort_api, OrtSessionOptions *session_options,
    const std::vector<std::string> &provider_names,
    const OrtExecutionProviderOptions *config) {
  if (ort_api == nullptr || session_options == nullptr) {
    return nullptr;
  }
  for (const std::string &name : provider_names) {
    OrtStatus *status =
        append_one_provider(ort_api, session_options, name, config);
    if (status != nullptr) {
      return status;
    }
  }
  return nullptr;
}
