#ifndef MOONSHINE_TTS_G2P_PATH_H
#define MOONSHINE_TTS_G2P_PATH_H

#include <filesystem>

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

}  // namespace moonshine_tts

#endif
