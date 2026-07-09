#ifndef MOONSHINE_TTS_ORT_SESSION_OPTIONS_H
#define MOONSHINE_TTS_ORT_SESSION_OPTIONS_H

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

#include "ort-utils.h"

namespace moonshine_tts {

Ort::SessionOptions make_ort_session_options(
    const std::vector<std::string>& provider_names,
    const std::string& coreml_cache_dir = {}, int intra_op_num_threads = 0,
    int inter_op_num_threads = 0);

inline Ort::SessionOptions make_g2p_ort_session_options(
    const std::vector<std::string>& provider_names,
    const std::string& coreml_cache_dir = {}) {
  return make_ort_session_options(provider_names, coreml_cache_dir, 1, 1);
}

}  // namespace moonshine_tts

#endif
