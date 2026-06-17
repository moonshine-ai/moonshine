/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef SIGNAL_SRC_SQUARE_ROOT_H_
#define SIGNAL_SRC_SQUARE_ROOT_H_

#include <stdint.h>

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above

// Square root
uint16_t Sqrt32(uint32_t num);
uint32_t Sqrt64(uint64_t num);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_SRC_SQUARE_ROOT_H_
