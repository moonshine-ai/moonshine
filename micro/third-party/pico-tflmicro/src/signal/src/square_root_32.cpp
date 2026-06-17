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

#include "signal/src/msb.h"
#include "signal/src/square_root.h"

namespace tflite {
namespace tflm_signal {

uint16_t Sqrt32(uint32_t num) {
  if (num == 0) {
    return 0;
  };
  uint32_t res = 0;
  int max_bit_number = 32 - MostSignificantBit32(num);
  max_bit_number |= 1;
  uint32_t bit = 1u << (31 - max_bit_number);
  int iterations = (31 - max_bit_number) / 2 + 1;
  while (iterations--) {
    if (num >= res + bit) {
      num -= res + bit;
      res = (res >> 1U) + bit;
    } else {
      res >>= 1U;
    }
    bit >>= 2U;
  }
  // Do rounding - if we have the bits.
  if (num > res && res != 0xFFFF) ++res;
  return res;
}

}  // namespace tflm_signal
}  // namespace tflite
