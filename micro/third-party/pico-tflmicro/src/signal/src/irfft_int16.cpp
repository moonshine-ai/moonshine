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
#include "signal/src/irfft.h"
#include "signal/src/kiss_fft_wrappers/kiss_fft_int16.h"

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflite {
namespace tflm_signal {

size_t IrfftInt16GetNeededMemory(int32_t fft_length) {
  size_t state_size = 0;
  kiss_fft_fixed16::kiss_fftr_alloc(fft_length, 1, nullptr, &state_size);
  return state_size;
}

void* IrfftInt16Init(int32_t fft_length, void* state, size_t state_size) {
  return kiss_fft_fixed16::kiss_fftr_alloc(fft_length, 1, state, &state_size);
}

void IrfftInt16Apply(void* state, const Complex<int16_t>* input,
                     int16_t* output) {
  kiss_fft_fixed16::kiss_fftri(
      static_cast<kiss_fft_fixed16::kiss_fftr_cfg>(state),
      reinterpret_cast<const kiss_fft_fixed16::kiss_fft_cpx*>(input),
      reinterpret_cast<kiss_fft_scalar*>(output));
}

}  // namespace tflm_signal
}  // namespace tflite
