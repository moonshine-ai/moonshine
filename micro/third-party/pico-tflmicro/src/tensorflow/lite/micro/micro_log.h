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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_LOG_H_
#define TENSORFLOW_LITE_MICRO_MICRO_LOG_H_

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
#include <cstdarg>
#include <cstddef>
// These functions can be used independent of the MicroErrorReporter to get
// printf-like functionalitys and are common to all target platforms.
void MicroPrintf(const char* format, ...);
void VMicroPrintf(const char* format, va_list args);
int MicroSnprintf(char* buffer, size_t buf_size, const char* format, ...);
int MicroVsnprintf(char* buffer, size_t buf_size, const char* format,
                   va_list vlist);
#else
// We use a #define to ensure that the strings are completely stripped, to
// prevent an unnecessary increase in the binary size.
#define MicroPrintf(...) tflite::Unused(__VA_ARGS__)
#define VMicroPrintf(...) tflite::Unused(__VA_ARGS__)
#define MicroSnprintf(...) tflite::Unused<int>(__VA_ARGS__)
#define MicroVsnprintf(...) tflite::Unused<int>(__VA_ARGS__)
#endif

namespace tflite {

// From
// https://stackoverflow.com/questions/23235910/variadic-unused-function-macro
template <typename... Args>
void Unused(Args&&... args) {
  (void)(sizeof...(args));
}

template <typename T, typename... Args>
T Unused(Args&&... args) {
  (void)(sizeof...(args));
  return static_cast<T>(0);
}

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_LOG_H_
