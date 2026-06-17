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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/fully_connected.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

const int kFullyConnectedInputTensor = 0;
const int kFullyConnectedWeightsTensor = 1;
const int kFullyConnectedBiasTensor = 2;
const int kFullyConnectedOutputTensor = 0;

FullyConnectedParams FullyConnectedParamsQuantized(
    const OpDataFullyConnected& op_data) {
  FullyConnectedParams op_params;
  op_params.input_offset = -op_data.input_zero_point;
  op_params.weights_offset = -op_data.filter_zero_point;
  op_params.output_offset = op_data.output_zero_point;
  op_params.output_multiplier = op_data.output_multiplier;
  op_params.output_shift = op_data.output_shift;
  op_params.quantized_activation_min = op_data.output_activation_min;
  op_params.quantized_activation_max = op_data.output_activation_max;
  return op_params;
}

FullyConnectedParams FullyConnectedParamsFloat(
    TfLiteFusedActivation activation) {
  FullyConnectedParams op_params;
  CalculateActivationRange(activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

TfLiteStatus CalculateOpDataFullyConnected(
    TfLiteContext* context, TfLiteFusedActivation activation,
    TfLiteType data_type, const TfLiteTensor* input, const TfLiteTensor* filter,
    const TfLiteTensor* bias, TfLiteTensor* output,
    OpDataFullyConnected* data) {
#ifndef HEXAGON
  data->is_per_channel = false;
#endif

  if (data_type == kTfLiteFloat32) {
    return kTfLiteOk;
  }

  bool is_per_channel = false;
  if (filter->quantization.type == kTfLiteAffineQuantization &&
      filter->quantization.params != nullptr) {
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    is_per_channel = affine_quantization->scale->size > 1;
  }

  if (is_per_channel) {
// Hexagon currently does not support per-channel fully connected, and the
// existing hexagon support library is intolerant of data members being added to
// OpDataFullyConnected. As such, we have to be careful not to reference newer
// data members. This is why we use a local variable is_per_channel in common
// code, and only reference the data->is_per_channel in non-HEXAGON code.
#ifdef HEXAGON
    TF_LITE_ENSURE_MSG(
        context, !is_per_channel,
        "FullyConnected per-channel quantization not yet supported on Hexagon. "
        "Please set converter._experimental_disable_per_channel_quantization_"
        "for_dense_layers = True.");
#else
    data->is_per_channel = is_per_channel;
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    const int per_channel_quantization_size = affine_quantization->scale->size;

    //  Currently only Int8 is supported for per channel quantization.
    TF_LITE_ENSURE(context,
                   input->type == kTfLiteInt8 && filter->type != kTfLiteInt4);

    TF_LITE_ENSURE_EQ(
        context, per_channel_quantization_size,
        filter->dims->data[affine_quantization->quantized_dimension]);

    data->per_channel_output_multiplier =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, per_channel_quantization_size * sizeof(int32_t)));
    data->per_channel_output_shift =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, per_channel_quantization_size * sizeof(int32_t)));

    // Populate multiplier and shift using affine quantization.
    const float input_scale = input->params.scale;
    const float output_scale = output->params.scale;
    const float* filter_scales = affine_quantization->scale->data;

    for (int i = 0; i < per_channel_quantization_size; ++i) {
      const float scale = filter_scales[i];
      const double filter_scale = static_cast<double>(scale);
      const double effective_output_scale = static_cast<double>(input_scale) *
                                            filter_scale /
                                            static_cast<double>(output_scale);
      int32_t significand;
      int channel_shift;
      QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
      data->per_channel_output_multiplier[i] = significand;
      data->per_channel_output_shift[i] = channel_shift;
    }
#endif
  } else {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  // Filter weights will always be symmetric quantized since we only support
  // int8 quantization. See
  // https://github.com/tensorflow/tensorflow/issues/44912 for additional
  // context.
  TFLITE_DCHECK(filter->params.zero_point == 0);

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  return CalculateActivationRangeQuantized(context, activation, output,
                                           &data->output_activation_min,
                                           &data->output_activation_max);
}

}  // namespace tflite
