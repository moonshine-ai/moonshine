#ifndef ORT_UTILS_CXX_H
#define ORT_UTILS_CXX_H

#include "onnxruntime_cxx_api.h"
#include "ort-utils.h"

inline void ort_append_execution_providers(
    Ort::SessionOptions &opts, const std::vector<std::string> &names,
    const OrtExecutionProviderOptions &config = {}) {
  const OrtApi *api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  OrtStatus *status = ort_append_execution_providers(api, opts, names, &config);
  Ort::ThrowOnError(status);
}

#endif
