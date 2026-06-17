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

#include "tensorflow/lite/micro/kernels/conv.h"

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct OpData {
  OpDataConv reference_op_data;

  // Index to buffer for optimizations if applicable.
  int buffer_idx;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  int32_t buf_size = 0;
  const auto& params =
      *(static_cast<const TfLiteConvParams*>(node->builtin_data));
  OpData* data = static_cast<OpData*>(node->user_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kConvWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempOutputTensor(node, kConvBiasTensor);
  TfLiteType bias_type = bias != nullptr ? bias->type : kTfLiteNoType;

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context,
                     input->type == kTfLiteFloat32 ||
                         input->type == kTfLiteInt16 ||
                         input->type == kTfLiteInt8,
                     "Input data type not supported");
  TF_LITE_ENSURE_MSG(
      context,
      (input->type == kTfLiteFloat32 && filter->type == kTfLiteFloat32) ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 &&
           (filter->type == kTfLiteInt4 || filter->type == kTfLiteInt8)),
      "Hybrid models are not supported on TFLite Micro.");

  // Consistency check tensor dims
  // Dimensionality
  TF_LITE_ENSURE_EQ(context, input->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 4);
  TF_LITE_ENSURE_EQ(context, output->dims->size, 4);
  // Equal batch size in input and output
  TF_LITE_ENSURE_EQ(context, input->dims->data[0], output->dims->data[0]);
  // Input channels should be an even multiple of filter channels
  TF_LITE_ENSURE(context, filter->dims->data[3] > 0);
  TF_LITE_ENSURE_EQ(context, input->dims->data[3] % filter->dims->data[3], 0);
  // Output channels should be an even multiple of the number of groups
  const int groups = input->dims->data[3] / filter->dims->data[3];
  TFLITE_DCHECK_EQ(output->dims->data[3] % groups, 0);
  // Bias size equal to output channels
  if (bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, bias->dims->size, 4);
    const int bias_size = NumElements(bias->dims);
    TFLITE_DCHECK_EQ(bias_size, output->dims->data[3]);
  }

  // Initialize cmsis_nn dimensions
  cmsis_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input->dims->data[3];

  cmsis_nn_dims filter_dims;
  filter_dims.n = 1;
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3];

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    const int num_channels = filter->dims->data[kConvQuantizedDimension];
    data->reference_op_data.per_channel_output_multiplier =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, num_channels * sizeof(int32_t)));
    data->reference_op_data.per_channel_output_shift =
        static_cast<int32_t*>(context->AllocatePersistentBuffer(
            context, num_channels * sizeof(int32_t)));
  }

  TF_LITE_ENSURE_STATUS(CalculateOpDataConv(
      context, node, params, input_dims.w, input_dims.h, filter_dims.w,
      filter_dims.h, output_dims.w, output_dims.h, input->type,
      &data->reference_op_data));

  // CMSIS_NN allows INT64 or nullptr bias data pointer
  if (input->type == kTfLiteInt8 ||
      (input->type == kTfLiteInt16 &&
       (bias_type == kTfLiteInt64 || bias_type == kTfLiteNoType))) {
    // Initialize cmsis_nn convolution parameters
    cmsis_nn_conv_params conv_params;
    conv_params.input_offset = -input->params.zero_point;
    conv_params.output_offset = output->params.zero_point;
    conv_params.stride.h = params.stride_height;
    conv_params.stride.w = params.stride_width;
    conv_params.dilation.h = params.dilation_height_factor;
    conv_params.dilation.w = params.dilation_width_factor;
    conv_params.padding.h = data->reference_op_data.padding.height;
    conv_params.padding.w = data->reference_op_data.padding.width;
    conv_params.activation.min = data->reference_op_data.output_activation_min;
    conv_params.activation.max = data->reference_op_data.output_activation_max;

    if (input->type == kTfLiteInt8) {
      buf_size = arm_convolve_wrapper_s8_get_buffer_size(
          &conv_params, &input_dims, &filter_dims, &output_dims);
    } else if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
      buf_size = arm_convolve_wrapper_s16_get_buffer_size(
          &conv_params, &input_dims, &filter_dims, &output_dims);
    }

    if (buf_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, buf_size, &data->buffer_idx));
    } else {
      data->buffer_idx = -1;
    }
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  if (bias != nullptr) {
    micro_context->DeallocateTempTfLiteTensor(bias);
  }

  return kTfLiteOk;
}

template <class ActType, class BiasType, class WeigthsType>
arm_cmsis_nn_status convolve_wrapper(
    const cmsis_nn_context* ctx, const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const ActType* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const BiasType* bias,
    const cmsis_nn_dims* output_dims, ActType* output, WeigthsType weightsT) {
  return ARM_CMSIS_NN_ARG_ERROR;
}

template <>
arm_cmsis_nn_status convolve_wrapper(
    const cmsis_nn_context* ctx, const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int8_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int8_t* output, TfLiteType weightsT) {
  if (weightsT == kTfLiteInt8) {
    return arm_convolve_wrapper_s8(ctx, conv_params, quant_params, input_dims,
                                   input, filter_dims, filter, bias_dims, bias,
                                   output_dims, output);
  } else if (weightsT == kTfLiteInt4) {
    return arm_convolve_wrapper_s4(ctx, conv_params, quant_params, input_dims,
                                   input, filter_dims, filter, bias_dims, bias,
                                   output_dims, output);
  } else {
    return ARM_CMSIS_NN_ARG_ERROR;
  }
}

template <>
arm_cmsis_nn_status convolve_wrapper(
    const cmsis_nn_context* ctx, const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int16_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int64_t* bias,
    const cmsis_nn_dims* output_dims, int16_t* output, TfLiteType weightsT) {
  const cmsis_nn_bias_data bias_data = {bias, false};

  return arm_convolve_wrapper_s16(ctx, conv_params, quant_params, input_dims,
                                  input, filter_dims, filter, bias_dims,
                                  &bias_data, output_dims, output);
}

template <>
arm_cmsis_nn_status convolve_wrapper(
    const cmsis_nn_context* ctx, const cmsis_nn_conv_params* conv_params,
    const cmsis_nn_per_channel_quant_params* quant_params,
    const cmsis_nn_dims* input_dims, const int16_t* input,
    const cmsis_nn_dims* filter_dims, const int8_t* filter,
    const cmsis_nn_dims* bias_dims, const int32_t* bias,
    const cmsis_nn_dims* output_dims, int16_t* output, TfLiteType weightsT) {
  const cmsis_nn_bias_data bias_data = {bias, true};

  return arm_convolve_wrapper_s16(ctx, conv_params, quant_params, input_dims,
                                  input, filter_dims, filter, bias_dims,
                                  &bias_data, output_dims, output);
}

template <typename ActType, typename BiasType, TfLiteType type>
TfLiteStatus EvalQuantizedPerChannel(TfLiteContext* context, TfLiteNode* node,
                                     const TfLiteConvParams& params,
                                     const OpData& data,
                                     const TfLiteEvalTensor* input,
                                     const TfLiteEvalTensor* filter,
                                     const TfLiteEvalTensor* bias,
                                     TfLiteEvalTensor* output) {
  cmsis_nn_conv_params conv_params;
  conv_params.dilation.h = params.dilation_height_factor;
  conv_params.dilation.w = params.dilation_width_factor;

  // Initialize cmsis_nn convolution parameters
  conv_params.input_offset = -data.reference_op_data.input_zero_point;
  conv_params.output_offset = data.reference_op_data.output_zero_point;
  conv_params.stride.h = params.stride_height;
  conv_params.stride.w = params.stride_width;
  conv_params.padding.h = data.reference_op_data.padding.height;
  conv_params.padding.w = data.reference_op_data.padding.width;
  conv_params.activation.min = data.reference_op_data.output_activation_min;
  conv_params.activation.max = data.reference_op_data.output_activation_max;

  // Initialize cmsis_nn per channel quantization parameters
  cmsis_nn_per_channel_quant_params quant_params;
  quant_params.multiplier = const_cast<int32_t*>(
      data.reference_op_data.per_channel_output_multiplier);
  quant_params.shift =
      const_cast<int32_t*>(data.reference_op_data.per_channel_output_shift);

  // Initialize cmsis_nn dimension structs, consistency is checked in the
  // prepare stage
  cmsis_nn_dims input_dims;
  input_dims.n = input->dims->data[0];
  input_dims.h = input->dims->data[1];
  input_dims.w = input->dims->data[2];
  input_dims.c = input->dims->data[3];

  cmsis_nn_dims filter_dims;
  filter_dims.n = 1;
  filter_dims.h = filter->dims->data[1];
  filter_dims.w = filter->dims->data[2];
  filter_dims.c = filter->dims->data[3];

  cmsis_nn_dims bias_dims;
  bias_dims.n = 1;
  bias_dims.h = 1;
  bias_dims.w = 1;
  bias_dims.c = output->dims->data[3];

  cmsis_nn_dims output_dims;
  output_dims.n = output->dims->data[0];
  output_dims.h = output->dims->data[1];
  output_dims.w = output->dims->data[2];
  output_dims.c = output->dims->data[3];

  // Initialize cmsis_nn context
  cmsis_nn_context ctx;
  ctx.buf = nullptr;
  ctx.size = 0;

  if (data.buffer_idx > -1) {
    ctx.buf = context->GetScratchBuffer(context, data.buffer_idx);
    // Note: ctx.size is currently not used in cmsis_nn.
    // The buffer should be allocated in the prepare function through
    // the corresponding arm_convolve_wrapper_[type]_get_buffer_size
  }

  // arm_convolve_wrapper_[type] dispatches the optimized kernel accordingly
  // with the parameters passed
  TFLITE_DCHECK_EQ(
      convolve_wrapper(
          &ctx, &conv_params, &quant_params, &input_dims,
          tflite::micro::GetTensorData<ActType>(input), &filter_dims,
          tflite::micro::GetTensorData<int8_t>(filter), &bias_dims,
          tflite::micro::GetOptionalTensorData<BiasType>(bias), &output_dims,
          tflite::micro::GetTensorData<ActType>(output), type),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}

TfLiteStatus EvalInt4(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt4>(
      context, node, params, data, input, filter, bias, output);
}

TfLiteStatus EvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt8>(
      context, node, params, data, input, filter, bias, output);
}

TfLiteStatus EvalInt16x8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  if (bias == nullptr || bias->type == kTfLiteInt32) {
    return EvalQuantizedPerChannel<int16_t, int32_t, kTfLiteInt16>(
        context, node, params, data, input, filter, bias, output);
  } else if (bias->type == kTfLiteInt64) {
    return EvalQuantizedPerChannel<int16_t, int64_t, kTfLiteInt16>(
        context, node, params, data, input, filter, bias, output);
  } else {
    MicroPrintf("Bias type %s (%d) not supported.",
                TfLiteTypeGetName(bias->type), bias->type);
    return kTfLiteError;
  }
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kConvOutputTensor);

  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto& params =
      *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8) ||
          (input->type == kTfLiteInt8 && filter->type == kTfLiteInt4),
      "Hybrid models are not supported on TFLite Micro.");

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      tflite::reference_ops::Conv(
          ConvParamsFloat(params, data.reference_op_data),
          tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8: {
      switch (filter->type) {
        case kTfLiteInt4: {
          return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt4>(
              context, node, params, data, input, filter, bias, output);
        }
        case kTfLiteInt8: {
          return EvalQuantizedPerChannel<int8_t, int32_t, kTfLiteInt8>(
              context, node, params, data, input, filter, bias, output);
        }
        default: {
          MicroPrintf("Filter type %s (%d) not supported.",
                      TfLiteTypeGetName(filter->type), filter->type);
          return kTfLiteError;
        }
      }
      break;
    }
    case kTfLiteInt16: {
      if (bias == nullptr || bias->type == kTfLiteInt32) {
        return EvalQuantizedPerChannel<int16_t, int32_t, kTfLiteInt16>(
            context, node, params, data, input, filter, bias, output);
      } else if (bias->type == kTfLiteInt64) {
        return EvalQuantizedPerChannel<int16_t, int64_t, kTfLiteInt16>(
            context, node, params, data, input, filter, bias, output);
      } else {
        MicroPrintf("Bias type %s (%d) not supported.",
                    TfLiteTypeGetName(bias->type), bias->type);
        return kTfLiteError;
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONV_2D() {
  return tflite::micro::RegisterOp(Init, Prepare, Eval);
}

TFLMRegistration Register_CONV_2D_INT4() {
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt4);
}

TFLMRegistration Register_CONV_2D_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt8);
}

TFLMRegistration Register_CONV_2D_INT16() {
  return tflite::micro::RegisterOp(Init, Prepare, EvalInt16x8);
}

}  // namespace tflite
