// Copyright 2024 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/micro/hexdump.h"

#include <algorithm>
#include <cctype>

#include "tensorflow/lite/micro/debug_log.h"
#include "tensorflow/lite/micro/static_vector.h"

namespace {

tflite::Span<char> output(const tflite::Span<char>& buf, const char* format,
                          ...) {
  // Writes formatted output, printf-style, to either a buffer or DebugLog.
  // Writes to DebugLog if the buffer data pointer is null. Does not exceed
  // the size of the buffer. Returns the unused remainder of the buffer, or a
  // buffer with a null data pointer in the case of printing to DebugLog.

  tflite::Span<char> result{nullptr, 0};

  va_list args;
  va_start(args, format);

  if (buf.data() == nullptr) {
    DebugLog(format, args);
    result = {nullptr, 0};
  } else {
    size_t len = DebugVsnprintf(buf.data(), buf.size(), format, args);
    // Returns the number of characters that would have been written if
    // there were enough room, so cap it at the size of the buffer in order to
    // know how much was actually written.
    size_t consumed = std::min(len, buf.size());
    result = {buf.data() + consumed, buf.size() - consumed};
  }

  va_end(args);
  return result;
}

}  // end anonymous namespace

tflite::Span<char> tflite::hexdump(const tflite::Span<const std::byte> region,
                                   const tflite::Span<char> out) {
  tflite::Span<char> buffer{out};
  std::size_t byte_nr = 0;
  constexpr int per_line = 16;
  const int lines = (region.size() + per_line - 1) / per_line;  // round up

  for (int line = 0; line < lines; ++line) {
    tflite::StaticVector<char, per_line> ascii;

    // print address
    buffer = output(buffer, "%08X:", line);

    for (int pos = 0; pos < per_line; ++pos) {
      if (byte_nr < region.size()) {
        // print byte
        int as_int = static_cast<int>(region[byte_nr++]);
        buffer = output(buffer, " %02X", as_int);

        // buffer an ascii printable value
        char c{'.'};
        if (std::isprint(as_int)) {
          c = static_cast<char>(as_int);
        }
        ascii.push_back(c);
      } else {
        buffer = output(buffer, "   ");
      }

      // print extra space in middle of the line
      if (pos == per_line / 2 - 1) {
        buffer = output(buffer, " ");
      }
    }

    // print the ascii value
    buffer = output(buffer, "  ");
    for (const auto& c : ascii) {
      buffer = output(buffer, "%c", c);
    }
    buffer = output(buffer, "%c", '\n');
  }

  return {out.data(), out.size() - buffer.size()};
}

void tflite::hexdump(const tflite::Span<const std::byte> region) {
  hexdump(region, {nullptr, 0});
}
