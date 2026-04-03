#ifndef MOONSHINE_TTS_BUILTIN_CPP_DATA_ROOT_H
#define MOONSHINE_TTS_BUILTIN_CPP_DATA_ROOT_H

#include <filesystem>

namespace moonshine_tts {

/// ``cpp/data`` next to this library’s sources (lexicons, JA/ZH/AR ONNX bundles, …).
std::filesystem::path builtin_cpp_data_root();

}  // namespace moonshine_tts

#endif
