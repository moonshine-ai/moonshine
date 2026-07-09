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

uint32_t Sqrt64(uint64_t num) {
  // Take a shortcut and just use 32 bit operations if the upper word is all
  // clear. This will cause a slight off by one issue for numbers close to 2^32,
  // but it probably isn't going to matter (and gives us a big performance win).
  if ((num >> 32) == 0) {
    return Sqrt32(static_cast<uint32_t>(num));
  }
  uint64_t res = 0;
  int max_bit_number = 64 - MostSignificantBit64(num);
  max_bit_number |= 1;
  uint64_t bit = UINT64_C(1) << (63 - max_bit_number);
  int iterations = (63 - max_bit_number) / 2 + 1;
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
  if (num > res && res != 0xFFFFFFFFLL) ++res;
  return res;
}

}  // namespace tflm_signal
}  // namespace tflite
