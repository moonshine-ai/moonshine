#include "json-utils.h"

#include <cstdlib>
#include <fstream>
#include <sstream>

std::string read_file_to_string(const std::string &path) {
  std::ifstream f(path);
  if (!f.good()) return "";
  std::stringstream buf;
  buf << f.rdbuf();
  return buf.str();
}

int json_get_int(const std::string &json, const char *key) {
  std::string search = std::string("\"") + key + "\":";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return 0;
  pos += search.length();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
  int val = 0;
  bool negative = false;
  if (json[pos] == '-') {
    negative = true;
    pos++;
  }
  while (pos < json.size() && json[pos] >= '0' && json[pos] <= '9') {
    val = val * 10 + (json[pos] - '0');
    pos++;
  }
  return negative ? -val : val;
}

std::string json_get_string(const std::string &json, const char *key) {
  std::string search = std::string("\"") + key + "\"";
  size_t pos = json.find(search);
  if (pos == std::string::npos) return "";

  // Skip past the key, colon, and whitespace to the opening quote.
  pos += search.length();
  while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
                                json[pos] == ':' || json[pos] == '\n' ||
                                json[pos] == '\r'))
    pos++;

  if (pos >= json.size() || json[pos] != '"') return "";
  pos++;  // skip opening quote

  std::string result;
  while (pos < json.size() && json[pos] != '"') {
    if (json[pos] == '\\' && pos + 1 < json.size()) {
      pos++;
      switch (json[pos]) {
        case '"':
          result += '"';
          break;
        case '\\':
          result += '\\';
          break;
        case 'n':
          result += '\n';
          break;
        case 't':
          result += '\t';
          break;
        case 'u': {
          // Basic \uXXXX → UTF-8 (BMP only, sufficient for IPA symbols).
          if (pos + 4 < json.size()) {
            char hex[5] = {json[pos + 1], json[pos + 2], json[pos + 3],
                           json[pos + 4], '\0'};
            uint32_t cp = static_cast<uint32_t>(strtoul(hex, nullptr, 16));
            pos += 4;
            if (cp < 0x80) {
              result += static_cast<char>(cp);
            } else if (cp < 0x800) {
              result += static_cast<char>(0xC0 | (cp >> 6));
              result += static_cast<char>(0x80 | (cp & 0x3F));
            } else {
              result += static_cast<char>(0xE0 | (cp >> 12));
              result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
              result += static_cast<char>(0x80 | (cp & 0x3F));
            }
          }
          break;
        }
        default:
          result += json[pos];
          break;
      }
    } else {
      result += json[pos];
    }
    pos++;
  }
  return result;
}
