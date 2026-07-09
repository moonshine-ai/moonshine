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
#include "signal/src/energy.h"

#include "signal/src/complex.h"

namespace tflite {
namespace tflm_signal {
void SpectrumToEnergy(const Complex<int16_t>* input, int start_index,
                      int end_index, uint32_t* output) {
  for (int i = start_index; i < end_index; i++) {
    const int16_t real = input[i].real;  // 15 bits
    const int16_t imag = input[i].imag;  // 15 bits
    // 31 bits
    output[i] = (static_cast<int32_t>(real) * real) +
                (static_cast<int32_t>(imag) * imag);
  }
}

}  // namespace tflm_signal
}  // namespace tflite
