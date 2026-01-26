#ifndef STRING_UTILS_H
#define STRING_UTILS_H

#include <map>
#include <string>
#include <vector>

std::string replace_all(std::string str, const std::string &from,
                        const std::string &to);

std::string trim(const std::string &str, const std::string &whitespace = " \t");

std::vector<std::string> split(const std::string &str,
                               const std::string &delimiter);

bool starts_with(const std::string &str, const std::string &prefix);

bool ends_with(const std::string &str, const std::string &suffix);

std::map<std::string, int64_t> name_to_index(
    const std::vector<const char *> &names);

std::string append_path_component(const std::string &path,
                                  const std::string &component);

std::string to_lowercase(const std::string &str);

bool bool_from_string(const char *input);

float float_from_string(const char *input);

int32_t int32_from_string(const char *input);

size_t size_t_from_string(const char *input);

#endif