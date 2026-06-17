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
#ifndef TENSORFLOW_LITE_MICRO_DEBUG_LOG_H_
#define TENSORFLOW_LITE_MICRO_DEBUG_LOG_H_

#ifdef __cplusplus
#include <cstdarg>
#include <cstddef>
#else
#include <stdarg.h>
#include <stddef.h>
#endif  // __cplusplus

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// These functions should be implemented by each target platform, and provide a
// way for strings to be output to some text stream. For more information, see
// the tensorflow/lite/micro/debug_log.cc file.  These functions should support
// standard C/C++ stdio style formatting operations.
void DebugLog(const char* format, va_list args);
int DebugVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // TENSORFLOW_LITE_MICRO_DEBUG_LOG_H_
