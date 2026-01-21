#ifndef MOONSHINE_TEST_UTILS_H
#define MOONSHINE_TEST_UTILS_H

#include <filesystem>

#include "debug-utils.h"

#define REQUIRE_FILE_EXISTS(filename)                                \
  do {                                                               \
    std::filesystem::path file_path = (filename);                    \
    if (!std::filesystem::exists(file_path)) {                       \
      std::string log_message =                                      \
          "No file found at '" + file_path.string() + "'. ";         \
      std::filesystem::path parent_path = file_path.parent_path();   \
      log_message +=                                                 \
          "Actual files found at '" + parent_path.string() + "': ("; \
      for (const auto &entry :                                       \
           std::filesystem::directory_iterator(parent_path)) {       \
        log_message += "'" + entry.path().string() + "', ";          \
      }                                                              \
      FAIL(log_message);                                             \
    }                                                                \
  } while (0)

#endif  // MOONSHINE_TEST_UTILS_H