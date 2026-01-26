#include "string-utils.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

// See
// https://stackoverflow.com/questions/2896600/how-to-replace-all-occurrences-of-a-character-in-string
std::string replace_all(std::string str, const std::string &from,
                        const std::string &to) {
  size_t start_pos = 0;
  while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return str;
}

// See
// https://stackoverflow.com/questions/1798112/removing-leading-and-trailing-spaces-from-a-string
std::string trim(const std::string &str, const std::string &whitespace) {
  const auto strBegin = str.find_first_not_of(whitespace);
  if (strBegin == std::string::npos) return "";  // no content

  const auto strEnd = str.find_last_not_of(whitespace);
  const auto strRange = strEnd - strBegin + 1;

  return str.substr(strBegin, strRange);
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delimiter) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t end = 0;
  while ((end = str.find(delimiter, start)) != std::string::npos) {
    result.push_back(str.substr(start, end - start));
    start = end + delimiter.length();
  }
  result.push_back(str.substr(start));
  return result;
}

bool starts_with(const std::string &str, const std::string &prefix) {
  return str.size() >= prefix.size() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string &str, const std::string &suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

std::map<std::string, int64_t> name_to_index(
    const std::vector<const char *> &names) {
  std::map<std::string, int64_t> name_to_index;
  for (size_t i = 0; i < names.size(); i++) {
    name_to_index[std::string(names[i])] = i;
  }
  return name_to_index;
}

std::string append_path_component(const std::string &path,
                                  const std::string &component) {
  std::string normalized_path = path;
  if (ends_with(normalized_path, "/")) {
    normalized_path = normalized_path.substr(0, normalized_path.size() - 1);
  }

  std::string normalized_component = component;
  if (starts_with(normalized_component, "/")) {
    normalized_component = normalized_component.substr(1);
  }
  if (normalized_component.empty() && normalized_path.empty()) {
    return "";
  }
  if (normalized_component.empty()) {
    return normalized_path;
  }
  if (normalized_path.empty()) {
    return normalized_component;
  }
  return normalized_path + '/' + normalized_component;
}

std::string to_lowercase(const std::string &str) {
  std::string result = str;
  std::transform(result.begin(), result.end(), result.begin(), ::tolower);
  return result;
}

bool bool_from_string(const char *input) {
  if (input == nullptr) {
    throw std::runtime_error("Input is null");
  }
  const std::string input_string = to_lowercase(input);
  if (input_string == "true" || input_string == "1") {
    return true;
  }
  if (input_string == "false" || input_string == "0") {
    return false;
  }
  throw std::runtime_error("Invalid boolean string: '" + std::string(input) +
                           "'");
  return false;
}

float float_from_string(const char *input) {
  if (input == nullptr) {
    throw std::runtime_error("Input is null");
  }
  float result = 0.0f;
  try {
    result = std::stof(input);
  } catch (const std::exception &e) {
    throw std::runtime_error("Invalid float string: '" + std::string(input) +
                             "': " + e.what());
  }
  return result;
}

int32_t int32_from_string(const char *input) {
  if (input == nullptr) {
    throw std::runtime_error("Input is null");
  }
  int32_t result = 0;
  try {
    result = std::stoi(input, nullptr, 10);
  } catch (const std::exception &e) {
    throw std::runtime_error("Invalid int32_t string: '" + std::string(input) +
                             "': " + e.what());
  }
  return result;
}

size_t size_t_from_string(const char *input) {
  if (input == nullptr) {
    throw std::runtime_error("Input is null");
  }
  size_t result = 0;
  try {
    result = std::stoul(input, nullptr, 10);
  } catch (const std::exception &e) {
    throw std::runtime_error("Invalid size_t string: '" + std::string(input) +
                             "': " + e.what());
  }
  return result;
}
