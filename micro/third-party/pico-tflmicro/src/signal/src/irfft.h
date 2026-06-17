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

#ifndef SIGNAL_SRC_IRFFT_H_
#define SIGNAL_SRC_IRFFT_H_

#include <stddef.h>
#include <stdint.h>

#include "signal/src/complex.h"

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflite {
namespace tflm_signal {

// IRFFT (Inverse Real Fast Fourier Transform)
// IFFT for real valued time domain outputs.

// 16-bit Integer input/output

// Returns the size of the memory that an IRFFT of `fft_length` needs
size_t IrfftInt16GetNeededMemory(int32_t fft_length);

// Initialize the state of an IRFFT of `fft_length`
// `state` points to an opaque state of size `state_size`, which
//  must be greater or equal to the value returned by
//  IrfftGetNeededMemory(fft_length). Fails if it isn't.
void* IrfftInt16Init(int32_t fft_length, void* state, size_t state_size);

// Applies IRFFT to `input` and writes the result to `output`
// * `input` must be of size `fft_length` elements (see IRfftInit)
// * `output` must be of size output
void IrfftInt16Apply(void* state, const Complex<int16_t>* input,
                     int16_t* output);

// 32-bit Integer input/output

// Returns the size of the memory that an IRFFT of `fft_length` needs
size_t IrfftInt32GetNeededMemory(int32_t fft_length);

// Initialize the state of an IRFFT of `fft_length`
// `state` points to an opaque state of size `state_size`, which
//  must be greater or equal to the value returned by
//  IrfftGetNeededMemory(fft_length). Fails if it isn't.
void* IrfftInt32Init(int32_t fft_length, void* state, size_t state_size);

// Applies IRFFT to `input` and writes the result to `output`
// * `input` must be of size `fft_length` elements (see IRfftInit)
// * `output` must be of size output
void IrfftInt32Apply(void* state, const Complex<int32_t>* input,
                     int32_t* output);

// Floating point input/output

// Returns the size of the memory that an IRFFT of `fft_length` needs
size_t IrfftFloatGetNeededMemory(int32_t fft_length);

// Initialize the state of an IRFFT of `fft_length`
// `state` points to an opaque state of size `state_size`, which
//  must be greater or equal to the value returned by
//  IrfftGetNeededMemory(fft_length). Fails if it isn't.
void* IrfftFloatInit(int32_t fft_length, void* state, size_t state_size);

// Applies IRFFT to `input` and writes the result to `output`
// * `input` must be of size `fft_length` elements (see IRfftInit)
// * `output` must be of size output
void IrfftFloatApply(void* state, const Complex<float>* input, float* output);

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_SRC_IRFFT_H_