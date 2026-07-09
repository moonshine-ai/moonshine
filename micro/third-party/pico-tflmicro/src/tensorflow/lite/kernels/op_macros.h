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
#ifndef TENSORFLOW_LITE_KERNELS_OP_MACROS_H_
#define TENSORFLOW_LITE_KERNELS_OP_MACROS_H_

#include "tensorflow/lite/micro/micro_log.h"

#if !defined(TF_LITE_MCU_DEBUG_LOG)
#include <cstdlib>
#define TFLITE_ABORT abort()
#else
inline void AbortImpl() {
  MicroPrintf("HALTED");
  while (1) {
  }
}
#define TFLITE_ABORT AbortImpl();
#endif

#if defined(ARDUINO)
#define TFLITE_ASSERT_FALSE (static_cast<void>(0))
#else
#define TFLITE_ASSERT_FALSE TFLITE_ABORT
#endif

#define TF_LITE_FATAL(msg)    \
  do {                        \
    MicroPrintf("%s", (msg)); \
    TFLITE_ABORT;             \
  } while (0)

#define TF_LITE_ASSERT(x)        \
  do {                           \
    if (!(x)) TF_LITE_FATAL(#x); \
  } while (0)

#endif  // TENSORFLOW_LITE_KERNELS_OP_MACROS_H_
