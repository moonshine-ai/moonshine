#include "ort-onnx-external-data.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace moonshine_tts {

void ort_add_external_initializer_files_for_onnx_model_buffer(Ort::SessionOptions& opts,
                                                              const FileInformationMap& files,
                                                              std::string_view model_map_key) {
  std::string data_key;
  if (model_map_key.size() >= 4 &&
      model_map_key.compare(model_map_key.size() - 4, 4, ".ort") == 0) {
    data_key.assign(model_map_key.begin(), model_map_key.end() - 4);
    data_key.append(".onnx.data");
  } else if (model_map_key.size() >= 5 &&
             model_map_key.compare(model_map_key.size() - 5, 5, ".onnx") == 0) {
    data_key.assign(model_map_key.begin(), model_map_key.end());
    data_key.append(".data");
  } else {
    return;
  }
  const auto it = files.entries.find(data_key);
  if (it == files.entries.end()) {
    return;
  }
  const FileInformation& fi = it->second;
  if (fi.memory == nullptr || fi.memory_size == 0) {
    return;
  }
  const std::string ort_basename = std::filesystem::path(data_key).filename().string();
  std::vector<std::basic_string<ORTCHAR_T>> names;
#ifdef _WIN32
  names.emplace_back(ort_basename.begin(), ort_basename.end());
#else
  names.emplace_back(ort_basename);
#endif
  char* buf = reinterpret_cast<char*>(const_cast<uint8_t*>(fi.memory));
  std::vector<char*> buffers = {buf};
  std::vector<size_t> lengths = {fi.memory_size};
  opts.AddExternalInitializersFromFilesInMemory(names, buffers, lengths);
}

}  // namespace moonshine_tts
