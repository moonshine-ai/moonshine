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
#include "signal/src/kiss_fft_wrappers/kiss_fft_float.h"

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflite {
namespace tflm_signal {

struct IrfftFloatState {
  int32_t fft_length;
  kiss_fft_float::kiss_fftr_cfg cfg;
};

size_t IrfftFloatGetNeededMemory(int32_t fft_length) {
  size_t cfg_size = 0;
  kiss_fft_float::kiss_fftr_alloc(fft_length, 1, nullptr, &cfg_size);
  return sizeof(IrfftFloatState) + cfg_size;
}

void* IrfftFloatInit(int32_t fft_length, void* state, size_t state_size) {
  IrfftFloatState* irfft_float_state = static_cast<IrfftFloatState*>(state);
  irfft_float_state->cfg =
      reinterpret_cast<kiss_fft_float::kiss_fftr_cfg>(irfft_float_state + 1);
  irfft_float_state->fft_length = fft_length;
  size_t cfg_size = state_size - sizeof(IrfftFloatState);
  return kiss_fft_float::kiss_fftr_alloc(fft_length, 1, irfft_float_state->cfg,
                                         &cfg_size);
}

void IrfftFloatApply(void* state, const Complex<float>* input, float* output) {
  IrfftFloatState* irfft_float_state = static_cast<IrfftFloatState*>(state);
  kiss_fft_float::kiss_fftri(
      static_cast<kiss_fft_float::kiss_fftr_cfg>(irfft_float_state->cfg),
      reinterpret_cast<const kiss_fft_float::kiss_fft_cpx*>(input),
      reinterpret_cast<kiss_fft_scalar*>(output));
  // KissFFT scales the IRFFT output by the FFT length.
  // KissFFT's nfft is the complex FFT length, which is half the real FFT's
  // length. Compensate.
  const int fft_length = irfft_float_state->fft_length;
  for (int i = 0; i < fft_length; i++) {
    output[i] /= fft_length;
  }
}

}  // namespace tflm_signal
}  // namespace tflite
