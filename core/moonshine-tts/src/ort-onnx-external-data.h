#ifndef MOONSHINE_TTS_ORT_ONNX_EXTERNAL_DATA_H
#define MOONSHINE_TTS_ORT_ONNX_EXTERNAL_DATA_H

#include "file-information.h"

#include <onnxruntime_cxx_api.h>
#include <string_view>

namespace moonshine_tts {

/// If ``files`` contains an in-memory ``<stem>.onnx.data`` companion for ``model_map_key`` (either
/// ``…/model.onnx`` or a canonical ``…/model.ort`` key), register it with ORT so
/// ``CreateSessionFromArray`` can resolve external initializers when the process cwd has no data tree.
void ort_add_external_initializer_files_for_onnx_model_buffer(Ort::SessionOptions& opts,
                                                                const FileInformationMap& files,
                                                                std::string_view model_map_key);

}  // namespace moonshine_tts

#endif
