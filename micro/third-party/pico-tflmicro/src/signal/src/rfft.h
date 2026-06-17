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

#ifndef SIGNAL_SRC_RFFT_H_
#define SIGNAL_SRC_RFFT_H_

#include <stddef.h>
#include <stdint.h>

#include "signal/src/complex.h"

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

// RFFT (Real Fast Fourier Transform)
// FFT for real valued time domain inputs.

// 16-bit Integer input/output

// Returns the size of the memory that an RFFT of `fft_length` needs
size_t RfftInt16GetNeededMemory(int32_t fft_length);

// Initialize the state of an RFFT of `fft_length`
// `state` points to an opaque state of size `state_size`, which
//  must be greater or equal to the value returned by
//  RfftGetNeededMemory(fft_length).
// Return the value of `state` on success or nullptr on failure
void* RfftInt16Init(int32_t fft_length, void* state, size_t state_size);

// Applies RFFT to `input` and writes the result to `output`
// * `input` must be of size `fft_length` elements (see RfftInit)
// * `output` must be of size (`fft_length` * 2) + 1 elements
void RfftInt16Apply(void* state, const int16_t* input,
                    Complex<int16_t>* output);

// 32-bit Integer input/output

// Returns the size of the memory that an RFFT of `fft_length` needs
size_t RfftInt32GetNeededMemory(int32_t fft_length);

// Initialize the state of an RFFT of `fft_length`
// `state` points to an opaque state of size `state_size`, which
//  must be greater or equal to the value returned by
//  RfftGetNeededMemory(fft_length).
// Return the value of `state` on success or nullptr on failure
void* RfftInt32Init(int32_t fft_length, void* state, size_t state_size);

// Applies RFFT to `input` and writes the result to `output`
// * `input` must be of size `fft_length` elements (see RfftInit)
// * `output` must be of size (`fft_length` * 2) + 1 elements
void RfftInt32Apply(void* state, const int32_t* input,
                    Complex<int32_t>* output);

// Floating point input/output

// Returns the size of the memory that an RFFT of `fft_length` needs
size_t RfftFloatGetNeededMemory(int32_t fft_length);

// Initialize the state of an RFFT of `fft_length`
// `state` points to an opaque state of size `state_size`, which
//  must be greater or equal to the value returned by
//  RfftGetNeededMemory(fft_length).
// Return the value of `state` on success or nullptr on failure
void* RfftFloatInit(int32_t fft_length, void* state, size_t state_size);

// Applies RFFT to `input` and writes the result to `output`
// * `input` must be of size `fft_length` elements (see RfftInit)
// * `output` must be of size (`fft_length` * 2) + 1 elements
void RfftFloatApply(void* state, const float* input, Complex<float>* output);

}  // namespace tflm_signal

#endif  // SIGNAL_SRC_RFFT_H_
