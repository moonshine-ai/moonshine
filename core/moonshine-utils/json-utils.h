#ifndef JSON_UTILS_H
#define JSON_UTILS_H

#include <string>

/**
 * Read the entire contents of a file into a std::string.
 * Returns an empty string if the file cannot be opened.
 */
std::string read_file_to_string(const std::string &path);

/**
 * Extract an integer value for a given key from a simple JSON object.
 * E.g. for key "depth" in {"depth": 6, ...} returns 6.
 * Returns 0 if the key is not found.
 */
int json_get_int(const std::string &json, const char *key);

/**
 * Extract a JSON string value for a given key from a simple JSON object.
 * E.g. for key "pad" in {"pad": "_", ...} returns "_".
 * Handles escaped characters (\", \\, \n, \t, \uXXXX basic-BMP → UTF-8).
 * Returns an empty string if the key is not found.
 */
std::string json_get_string(const std::string &json, const char *key);

/**
 * Iterate over a UTF-8 string one code-point at a time, calling `fn` with each
 * code-point as a std::string.
 */
template <typename Fn>
void for_each_utf8_char(const std::string &s, Fn fn) {
  size_t i = 0;
  while (i < s.size()) {
    size_t len = 1;
    uint8_t c = static_cast<uint8_t>(s[i]);
    if ((c & 0x80) == 0)
      len = 1;
    else if ((c & 0xE0) == 0xC0)
      len = 2;
    else if ((c & 0xF0) == 0xE0)
      len = 3;
    else if ((c & 0xF8) == 0xF0)
      len = 4;
    fn(s.substr(i, len));
    i += len;
  }
}

#endif
