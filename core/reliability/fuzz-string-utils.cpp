// libFuzzer harness for the string utilities.
//
// Feeds arbitrary bytes through the trimming, splitting, path, and typed-parser
// helpers. The typed parsers throw on invalid input by design, so those calls
// are wrapped; only sanitizer errors fail the run.

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>

#include "string-utils.h"

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  const std::string input(data, data + size);

  trim(input);
  to_lowercase(input);
  starts_with(input, "prefix");
  ends_with(input, "suffix");
  replace_all(input, " ", "_");
  split(input, ",");
  append_path_component(input, input);

  try {
    bool_from_string(input);
  } catch (const std::exception &) {
  }
  try {
    float_from_string(input);
  } catch (const std::exception &) {
  }
  try {
    int32_from_string(input);
  } catch (const std::exception &) {
  }
  try {
    size_t_from_string(input);
  } catch (const std::exception &) {
  }
  return 0;
}
