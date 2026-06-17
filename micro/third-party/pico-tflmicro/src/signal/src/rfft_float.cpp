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

#include <stddef.h>
#include <stdint.h>

#include "signal/src/complex.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_float.h"
#include "signal/src/rfft.h"

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

size_t RfftFloatGetNeededMemory(int32_t fft_length) {
  size_t state_size = 0;
  kiss_fft_float::kiss_fftr_alloc(fft_length, 0, nullptr, &state_size);
  return state_size;
}

void* RfftFloatInit(int32_t fft_length, void* state, size_t state_size) {
  return kiss_fft_float::kiss_fftr_alloc(fft_length, 0, state, &state_size);
}

void RfftFloatApply(void* state, const float* input, Complex<float>* output) {
  kiss_fft_float::kiss_fftr(
      static_cast<kiss_fft_float::kiss_fftr_cfg>(state),
      reinterpret_cast<const kiss_fft_scalar*>(input),
      reinterpret_cast<kiss_fft_float::kiss_fft_cpx*>(output));
}

}  // namespace tflm_signal
