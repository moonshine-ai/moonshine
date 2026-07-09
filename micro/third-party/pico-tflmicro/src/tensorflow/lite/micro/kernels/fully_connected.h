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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

struct OpDataFullyConnected {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
  // Cached zero point values of tensors.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;

// TODO(b/258710417): enable by default once optimized fully-connected works for
// all targets.
#if !defined(HEXAGON)
  // A buffer used to store unpacked filter values. This is used if the source
  // tensor is of n-bit precision that cannot be easily processed by kernels.
  int filter_buffer_index;

  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
  bool is_per_channel;
#endif

#ifdef USE_TFLM_COMPRESSION

  // scratch buffers for compressed tensors
  int weights_scratch_index;
  int bias_scratch_index;

#endif  // USE_TFLM_COMPRESSION
};

extern const int kFullyConnectedInputTensor;
extern const int kFullyConnectedWeightsTensor;
extern const int kFullyConnectedBiasTensor;
extern const int kFullyConnectedOutputTensor;

// Returns a FullyConnectedParams struct with all the parameters needed for a
// float computation.
FullyConnectedParams FullyConnectedParamsFloat(
    TfLiteFusedActivation activation);

// Returns a FullyConnectedParams struct with all the parameters needed for a
// quantized computation.
FullyConnectedParams FullyConnectedParamsQuantized(
    const OpDataFullyConnected& op_data);

TfLiteStatus CalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output, OpDataFullyConnected* data);

// This is the most generic TFLMRegistration. The actual supported types
// may still be target dependent. The only requirement is that every
// implementation (reference or optimized) must define this function.
TFLMRegistration Register_FULLY_CONNECTED();

#if defined(ARDUINO) || defined(HEXAGON) || defined(XTENSA)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8.
TFLMRegistration Register_FULLY_CONNECTED_INT8();

#else
// Note that while this block gets used for both reference and optimized kernels
// that do not have any specialized implementations, the only goal here is to
// define fallback implementation that allow reference kernels to still be used
// from applications that call a more specific kernel variant.

inline TFLMRegistration Register_FULLY_CONNECTED_INT8() {
  return Register_FULLY_CONNECTED();
}

#endif

#if defined(ARDUINO)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int16.
TFLMRegistration Register_FULLY_CONNECTED_INT16();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 and int4 packed kernels.
TFLMRegistration Register_FULLY_CONNECTED_INT4();

#else
// Note that while this block gets used for both reference and optimized kernels
// that do not have any specialized implementations, the only goal here is to
// define fallback implementation that allow reference kernels to still be used
// from applications that call a more specific kernel variant.

inline TFLMRegistration Register_FULLY_CONNECTED_INT16() {
  return Register_FULLY_CONNECTED();
}

inline TFLMRegistration Register_FULLY_CONNECTED_INT4() {
  return Register_FULLY_CONNECTED();
}

#endif

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_FULLY_CONNECTED_H_
