#include "ort-session-options.h"

#include "ort-utils-cxx.h"

namespace moonshine_tts {

Ort::SessionOptions make_ort_session_options(
    const std::vector<std::string>& provider_names,
    const std::string& coreml_cache_dir, int intra_op_num_threads,
    int inter_op_num_threads) {
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  opts.SetIntraOpNumThreads(intra_op_num_threads);
  opts.SetInterOpNumThreads(inter_op_num_threads);
  // Match STT and the spelling model, which have always been ORT-only. On a
  // full ORT build this is what makes a stray .onnx fail here rather than
  // loading fine on desktop and failing later in the browser.
  opts.AddConfigEntry("session.load_model_format", "ORT");
  if (!provider_names.empty()) {
    OrtExecutionProviderOptions ep_config{};
    if (!coreml_cache_dir.empty()) {
      ep_config.coreml_cache_dir = coreml_cache_dir.c_str();
    }
    ort_append_execution_providers(opts, provider_names, ep_config);
  }
  return opts;
}

}  // namespace moonshine_tts
