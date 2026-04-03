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

struct FileInformation {
  std::filesystem::path path;
  const uint8_t* memory = nullptr;
  size_t memory_size = 0;
};

/// Maps canonical asset keys (typically default relative paths such as ``fr/dict.tsv``) to
/// ``FileInformation``. Optional overrides use the same keys as ``MoonshineG2POptions::parse_options``
/// where no single bundle default exists (e.g. ``portuguese_dict_path``, ``heteronym_onnx_override``).
struct FileInformationMap {
  std::map<std::string, FileInformation> entries;

  void set_path(std::string_view key, std::filesystem::path path) {
    entries[std::string(key)] = FileInformation{std::move(path), nullptr, 0};
  }

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
