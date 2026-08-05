#ifndef MOONSHINE_TTS_ORT_MODEL_FORMAT_H
#define MOONSHINE_TTS_ORT_MODEL_FORMAT_H

#include <cstddef>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <string_view>

#include "g2p-path.h"

namespace moonshine_tts {

/// Rejecting ONNX models, and saying why.
///
/// Moonshine loads ORT-format models only. The wasm and mobile runtimes are
/// minimal ONNX Runtime builds with no ONNX parser compiled in at all, so a
/// ``.onnx`` cannot be read there whatever we do. Checking here rather than
/// letting the load fail keeps one behaviour across platforms, and turns what
/// would be an opaque parse error into a message saying how to fix it.
///
/// Header-only and free of any ONNX Runtime dependency, so the option-parsing
/// code can use it without linking the runtime.

namespace detail {

inline std::string ort_only_message(std::string_view what) {
  return std::string(what) +
         ": Moonshine loads ORT-format models only. Convert it with "
         "scripts/convert-models-to-ort.py.";
}

}  // namespace detail

/// Throws if *path* names a ``.onnx``, naming *what* in the message.
inline void require_ort_model_path(const std::filesystem::path& path,
                                   std::string_view what) {
  if (is_onnx_model_name(path.filename().string())) {
    throw std::runtime_error(detail::ort_only_message(what) +
                             " Got: " + path.string());
  }
}

/// Throws if the buffer does not hold an ORT model, naming *what*.
inline void require_ort_model_bytes(const void* data, size_t length,
                                    std::string_view what) {
  // An ORT flatbuffer carries "ORTM" at offset 4. A serialised ONNX model is
  // protobuf and will not, so this separates the two without parsing either.
  static constexpr size_t kIdentifierEnd = 8;
  if (data == nullptr || length < kIdentifierEnd) {
    return;
  }
  const char* bytes = static_cast<const char*>(data);
  if (std::memcmp(bytes + 4, "ORTM", 4) != 0) {
    throw std::runtime_error(detail::ort_only_message(what) +
                             " The supplied bytes are not an ORT model.");
  }
}

}  // namespace moonshine_tts

#endif
