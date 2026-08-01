#ifndef MOONSHINE_TTS_G2P_PATH_H
#define MOONSHINE_TTS_G2P_PATH_H

#include <filesystem>
#include <string>
#include <string_view>

namespace moonshine_tts {

/// If ``path`` is absolute, returns it unchanged. If ``root`` is empty, returns
/// ``path`` (relative to the process working directory). Otherwise returns
/// ``root / path``.
inline std::filesystem::path resolve_path_under_root(
    const std::filesystem::path& root, const std::filesystem::path& path) {
  if (path.empty()) {
    return path;
  }
  if (path.is_absolute()) {
    return path;
  }
  if (root.empty()) {
    return path;
  }
  return root / path;
}

/// True when ``name`` ends in ``.onnx``.
inline bool is_onnx_model_name(std::string_view name) {
  return name.size() >= 5 && name.compare(name.size() - 5, 5, ".onnx") == 0;
}

/// The ORT model in ``dir`` for a model named ``basename``.
///
/// Models ship exclusively in ORT format, but some names reaching here still
/// say ``.onnx``: the ``onnx_model_file`` field in a bundle's ``meta.json``,
/// and directory-shorthand options that build a filename themselves. Both are
/// mapped onto the ``.ort`` we actually ship. A name with any other extension
/// is returned unchanged, for the caller to fail on.
inline std::filesystem::path ort_model_path(const std::filesystem::path& dir,
                                            std::string_view basename) {
  if (is_onnx_model_name(basename)) {
    const std::string stem(basename.substr(0, basename.size() - 5));
    return dir / (stem + ".ort");
  }
  return dir / std::string(basename);
}

}  // namespace moonshine_tts

#endif
