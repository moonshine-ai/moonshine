/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

// For the TfLite transpose_conv implementation, input tensor 0 corresponds to
// the OutputShapeTensor. However, since TFLM does not support dynamic tensors,
// the TFLM implementation ignores input tensor 0 and the only inputs we care
// about are kFilterTensor, kInputTensor and kBiasTensor.
constexpr int kTransposeConvFilterTensor = 1;
constexpr int kTransposeConvInputTensor = 2;
constexpr int kTransposeConvBiasTensor = 3;
constexpr int kTransposeConvOutputTensor = 0;

// Conv is quantized along dimension 0:
// https://www.tensorflow.org/lite/performance/quantization_spec
constexpr int kTransposeConvQuantizedDimension = 0;

// This is the most generic TFLMRegistration. The actual supported types
// may still be target dependent. The only requirement is that every
// implementation (reference or optimized) must define this function.
TFLMRegistration Register_TRANSPOSE_CONV();

#if defined(ARDUINO)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8.
TFLMRegistration Register_TRANSPOSE_CONV_INT8();

#else
// Note that while this block gets used for both reference and optimized kernels
// that do not have any specialized implementations, the only goal here is to
// define fallback implementation that allow reference kernels to still be used
// from applications that call a more specific kernel variant.

inline TFLMRegistration Register_TRANSPOSE_CONV_INT8() {
  return Register_TRANSPOSE_CONV();
}

#endif

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_TRANSPOSE_CONV_H_
