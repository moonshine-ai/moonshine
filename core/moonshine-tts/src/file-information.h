#ifndef MOONSHINE_TTS_FILE_INFORMATION_H
#define MOONSHINE_TTS_FILE_INFORMATION_H

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace moonshine_tts {

/// Describes a bundled asset: optional on-disk ``path`` (relative to a caller root unless absolute),
/// optional client-owned ``memory`` / ``memory_size``, or bytes read from disk by ``load()``.
struct FileInformation {
  std::filesystem::path path{};
  const uint8_t* memory = nullptr;
  size_t memory_size = 0;

  FileInformation() = default;
  FileInformation(std::filesystem::path p, const uint8_t* mem, size_t sz)
      : path(std::move(p)), memory(mem), memory_size(sz) {}

  FileInformation(const FileInformation& o);
  FileInformation& operator=(const FileInformation& o);
  FileInformation(FileInformation&& o) noexcept = default;
  FileInformation& operator=(FileInformation&& o) noexcept = default;

  /// If ``memory`` / ``memory_size`` are set (client buffer), returns them. Otherwise reads ``path``
  /// into internal storage and returns that. Throws if neither is available or the file cannot be read.
  void load(const uint8_t** out_memory, size_t* out_size);

  /// Drops bytes read by ``load()`` from disk. Does not free client-supplied ``memory``; clears only
  /// this object's view when it pointed at internally loaded data.
  void free();

 private:
  std::vector<uint8_t> owned_storage_{};
};

/// Maps canonical asset keys (typically default relative paths such as ``fr/dict.tsv``) to
/// ``FileInformation``. Optional overrides use the same keys as ``MoonshineG2POptions::parse_options``
/// where no single bundle default exists (e.g. ``portuguese_dict_path``, ``oov_onnx_override``).
struct FileInformationMap {
  std::map<std::string, FileInformation> entries;

  void set_path(std::string_view key, std::filesystem::path path) {
    entries[std::string(key)] = FileInformation{std::move(path), nullptr, 0};
  }

  /// Client-owned bytes; ``path`` defaults to ``key`` for layout resolution (e.g. parent directories).
  void set_memory(std::string_view key, const uint8_t* mem, size_t sz,
                  std::filesystem::path path_for_resolve = {});

  void erase_key(std::string_view key) { entries.erase(std::string(key)); }

  bool contains(std::string_view key) const {
    return entries.find(std::string(key)) != entries.end();
  }

  /// Fills ``entries`` from ``(*key_list)[i].first`` → path ``root_path / (*key_list)[i].second``,
  /// with optional in-memory bytes per row.
  void parse_file_list(const std::vector<std::pair<std::string, std::string>>* key_list,
                       const std::vector<uint8_t*>* memory_pointers,
                       const std::vector<size_t>* memory_sizes,
                       const std::filesystem::path& root_path);
};

}  // namespace moonshine_tts

#endif  // MOONSHINE_TTS_FILE_INFORMATION_H
