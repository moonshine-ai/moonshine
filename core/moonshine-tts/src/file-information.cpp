#include "file-information.h"

#include <stdexcept>

namespace moonshine_tts {

void FileInformationMap::parse_file_list(const std::vector<std::pair<std::string, std::string>>* key_list,
                                       const std::vector<uint8_t*>* memory_pointers,
                                       const std::vector<size_t>* memory_sizes,
                                       const std::filesystem::path& root_path) {
  if (key_list == nullptr) {
    throw std::runtime_error("FileInformationMap::parse_file_list: key_list is null");
  }
  const bool have_mem = memory_pointers != nullptr && memory_sizes != nullptr;
  if (have_mem && (memory_pointers->size() != key_list->size() ||
                   memory_sizes->size() != key_list->size())) {
    throw std::runtime_error(
        "FileInformationMap::parse_file_list: memory vector sizes must match key_list");
  }
  for (size_t i = 0; i < key_list->size(); ++i) {
    const std::string& key = key_list->at(i).first;
    const std::string& rel = key_list->at(i).second;
    const std::filesystem::path full_path = root_path / rel;
    const uint8_t* memory = nullptr;
    size_t memory_size = 0;
    if (have_mem) {
      memory = memory_pointers->at(i);
      memory_size = memory_sizes->at(i);
    }
    entries[key] = FileInformation{full_path, memory, memory_size};
  }
}

}  // namespace moonshine_tts
