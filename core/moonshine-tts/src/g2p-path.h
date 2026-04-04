#ifndef MOONSHINE_TTS_G2P_PATH_H
#define MOONSHINE_TTS_G2P_PATH_H

#include <filesystem>
#include <string>
#include <string_view>

namespace moonshine_tts {

/// If ``path`` is absolute, returns it unchanged. If ``root`` is empty, returns ``path`` (relative to
/// the process working directory). Otherwise returns ``root / path``.
inline std::filesystem::path resolve_path_under_root(const std::filesystem::path& root,
                                                      const std::filesystem::path& path) {
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

/// Prefer ``stem.ort`` when present, else ``stem.onnx`` (``basename`` may end with ``.ort`` or ``.onnx``).
/// If neither exists, returns ``dir / basename`` (caller may use this in error messages).
inline std::filesystem::path resolve_prefer_ort_model(const std::filesystem::path& dir,
                                                      std::string_view basename) {
  namespace fs = std::filesystem;
  const std::string b(basename);
  std::string stem;
  if (b.size() >= 4 && b.compare(b.size() - 4, 4, ".ort") == 0) {
    stem = b.substr(0, b.size() - 4);
  } else if (b.size() >= 5 && b.compare(b.size() - 5, 5, ".onnx") == 0) {
    stem = b.substr(0, b.size() - 5);
  } else {
    return dir / b;
  }
  const fs::path ort = dir / (stem + ".ort");
  const fs::path onnx = dir / (stem + ".onnx");
  if (fs::is_regular_file(ort)) {
    return ort;
  }
  if (fs::is_regular_file(onnx)) {
    return onnx;
  }
  return dir / b;
}

/// For a path whose basename ends with ``.ort`` or ``.onnx``, set ``path`` to the existing sibling
/// preferring ``stem.ort`` over ``stem.onnx`` when both are present.
inline void resolve_disk_model_file_path(std::filesystem::path& path) {
  namespace fs = std::filesystem;
  if (path.empty()) {
    return;
  }
  const std::string fn = path.filename().string();
  std::string stem;
  if (fn.size() >= 5 && fn.compare(fn.size() - 5, 5, ".onnx") == 0) {
    stem = fn.substr(0, fn.size() - 5);
  } else if (fn.size() >= 4 && fn.compare(fn.size() - 4, 4, ".ort") == 0) {
    stem = fn.substr(0, fn.size() - 4);
  } else {
    return;
  }
  const fs::path parent = path.parent_path();
  const fs::path ort = parent / (stem + ".ort");
  const fs::path onnx = parent / (stem + ".onnx");
  if (fs::is_regular_file(ort)) {
    path = ort;
    return;
  }
  if (fs::is_regular_file(onnx)) {
    path = onnx;
  }
}

}  // namespace moonshine_tts

#endif
