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

// TODO(b/291167350):  can allow __builtin_clz to be used in more cases here
uint32_t MostSignificantBit32(uint32_t x) {
#if defined(XTENSA)
  // XT_NSAU returns the number of left shifts needed to put the MSB in the
  // leftmost position. Returns 32 if the argument is 0.
  return 32 - XT_NSAU(x);
#elif defined(__GNUC__)
  if (x) {
    return 32 - __builtin_clz(x);
  }
  return 32;
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
