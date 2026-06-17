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

#if defined(XTENSA)
#include <xtensa/tie/xt_misc.h>
#endif

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above

uint32_t MostSignificantBit64(uint64_t x) {
#if defined(XTENSA)
  // XT_NSAU returns the number of left shifts needed to put the MSB in the
  // leftmost position. Returns 32 if the argument is 0.
  uint32_t upper = 64 - XT_NSAU((uint32_t)(x >> 32));
  if (upper != 32) {
    return upper;
  }
  // Only if the upper bits are all clear do we want to look at the lower bits.
  return 32 - XT_NSAU((uint32_t)x);
#elif defined(__GNUC__)
  if (x) {
    return 64 - __builtin_clzll(x);
  }
  return 64;
#else
  uint32_t temp = 0;
  while (x) {
    x = x >> 1;
    ++temp;
  }
  return temp;
#endif
}

}  // namespace tflm_signal
}  // namespace tflite
