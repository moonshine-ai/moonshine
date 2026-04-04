#include "file-information.h"

#include <fstream>
#include <stdexcept>

namespace moonshine_tts {

FileInformation::FileInformation(const FileInformation& o) : path(o.path), owned_storage_(o.owned_storage_) {
  if (!owned_storage_.empty()) {
    memory = owned_storage_.data();
    memory_size = owned_storage_.size();
  } else {
    memory = o.memory;
    memory_size = o.memory_size;
  }
}

FileInformation& FileInformation::operator=(const FileInformation& o) {
  if (this == &o) {
    return *this;
  }
  path = o.path;
  owned_storage_ = o.owned_storage_;
  if (!owned_storage_.empty()) {
    memory = owned_storage_.data();
    memory_size = owned_storage_.size();
  } else {
    memory = o.memory;
    memory_size = o.memory_size;
  }
  return *this;
}

void FileInformation::load(const uint8_t** out_memory, size_t* out_size) {
  if (out_memory == nullptr || out_size == nullptr) {
    throw std::runtime_error("FileInformation::load: null output pointer");
  }
  if (memory != nullptr && memory_size > 0 && owned_storage_.empty()) {
    *out_memory = memory;
    *out_size = memory_size;
    return;
  }
  if (!owned_storage_.empty()) {
    *out_memory = owned_storage_.data();
    *out_size = owned_storage_.size();
    return;
  }
  if (path.empty()) {
    throw std::runtime_error("FileInformation::load: empty path and no memory buffer");
  }
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("FileInformation::load: cannot open " + path.string());
  }
  f.seekg(0, std::ios::end);
  const std::streamoff end = f.tellg();
  if (end < 0) {
    throw std::runtime_error("FileInformation::load: cannot size " + path.string());
  }
  f.seekg(0, std::ios::beg);
  owned_storage_.resize(static_cast<size_t>(end));
  if (!owned_storage_.empty()) {
    f.read(reinterpret_cast<char*>(owned_storage_.data()),
           static_cast<std::streamsize>(owned_storage_.size()));
  }
  if (!f) {
    owned_storage_.clear();
    throw std::runtime_error("FileInformation::load: failed reading " + path.string());
  }
  memory = owned_storage_.data();
  memory_size = owned_storage_.size();
  *out_memory = memory;
  *out_size = memory_size;
}

void FileInformation::free() {
  const bool had_owned = !owned_storage_.empty();
  owned_storage_.clear();
  if (had_owned) {
    memory = nullptr;
    memory_size = 0;
  }
}

void FileInformationMap::set_memory(std::string_view key, const uint8_t* mem, size_t sz,
                                    std::filesystem::path path_for_resolve) {
  std::filesystem::path p =
      path_for_resolve.empty() ? std::filesystem::path(std::string(key)) : std::move(path_for_resolve);
  entries[std::string(key)] = FileInformation{std::move(p), mem, sz};
}

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
    const std::string& map_key = key_list->at(i).first;
    const std::string& rel = key_list->at(i).second;
    const std::filesystem::path full_path = root_path / rel;
    const uint8_t* mem = nullptr;
    size_t mem_sz = 0;
    if (have_mem) {
      mem = memory_pointers->at(i);
      mem_sz = memory_sizes->at(i);
    }
    entries[map_key] = FileInformation{full_path, mem, mem_sz};
  }
}

}  // namespace moonshine_tts
