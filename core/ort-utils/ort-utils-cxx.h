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

// C++ counterpart of ort_create_env(): on the multithreaded WebAssembly build
// the env must own the global thread pools (ORT defaults
// use_per_session_threads=false there), otherwise a plain per-session Env is
// correct. Use for every Ort::Env so the wasm-threaded build can create
// sessions.
inline Ort::Env make_ort_env(OrtLoggingLevel logging_level,
                             const char *logid) {
#if defined(__EMSCRIPTEN__) && defined(__EMSCRIPTEN_PTHREADS__)
  Ort::ThreadingOptions threading_options;
  return Ort::Env(threading_options, logging_level, logid);
#else
  return Ort::Env(logging_level, logid);
#endif
}

#endif
