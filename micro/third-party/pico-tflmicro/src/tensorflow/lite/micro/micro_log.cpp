/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_log.h"

#include <cstdarg>
#include <cstdint>

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
#include "tensorflow/lite/micro/debug_log.h"
#endif

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
namespace {

void VDebugLog(const char* format, ...) {
  va_list args;
  va_start(args, format);
  DebugLog(format, args);
  va_end(args);
}

}  // namespace

void VMicroPrintf(const char* format, va_list args) {
  DebugLog(format, args);
  // TODO(b/290051015): remove "\r\n"
  VDebugLog("\r\n");
}

void MicroPrintf(const char* format, ...) {
  va_list args;
  va_start(args, format);
  VMicroPrintf(format, args);
  va_end(args);
}

int MicroSnprintf(char* buffer, size_t buf_size, const char* format, ...) {
  va_list args;
  va_start(args, format);
  int result = MicroVsnprintf(buffer, buf_size, format, args);
  va_end(args);
  return result;
}

int MicroVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist) {
  return DebugVsnprintf(buffer, buf_size, format, vlist);
}
#endif  // !defined(TF_LITE_STRIP_ERROR_STRINGS)
