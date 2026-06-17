/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/pooling.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

struct OpData {
  OpDataPooling reference_op_data;

  // Index to buffer for optimizations if applicable.
  int buffer_idx;
};

void PopulateCommonParams(
    TfLiteContext* const context, cmsis_nn_dims* const input_dims,
    cmsis_nn_dims* const output_dims, cmsis_nn_pool_params* const pool_params,
    cmsis_nn_context* const ctx, cmsis_nn_dims* const filter_dims,
    const OpData& data, const RuntimeShape& input_shape,
    const RuntimeShape& output_shape, const TfLitePoolParams* params) {
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);

  input_dims->n = 1;
  input_dims->h = input_shape.Dims(1);
  input_dims->w = input_shape.Dims(2);
  input_dims->c = depth;

  output_dims->n = 1;
  output_dims->h = output_shape.Dims(1);
  output_dims->w = output_shape.Dims(2);
  output_dims->c = depth;

  pool_params->stride.h = params->stride_height;
  pool_params->stride.w = params->stride_width;
  pool_params->padding.h = data.reference_op_data.padding.height;
  pool_params->padding.w = data.reference_op_data.padding.width;
  pool_params->activation.min = data.reference_op_data.activation_min;
  pool_params->activation.max = data.reference_op_data.activation_max;

  filter_dims->n = 1;
  filter_dims->h = params->filter_height;
  filter_dims->w = params->filter_width;
  filter_dims->c = 1;
  ctx->buf = nullptr;
  ctx->size = 0;
  if (data.buffer_idx > -1) {
    ctx->buf = context->GetScratchBuffer(context, data.buffer_idx);
  }
}

void AverageEvalQuantized(TfLiteContext* context, const TfLiteNode* node,
                          const TfLitePoolParams* params, const OpData& data,
                          const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output) {
  TFLITE_DCHECK((input->type == kTfLiteInt8) || (input->type == kTfLiteInt16));

  RuntimeShape input_shape = micro::GetTensorShape(input);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);

  RuntimeShape output_shape = micro::GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  cmsis_nn_dims input_dims;
  cmsis_nn_dims output_dims;
  cmsis_nn_pool_params pool_params;
  cmsis_nn_dims filter_dims;
  cmsis_nn_context ctx;

  PopulateCommonParams(context, &input_dims, &output_dims, &pool_params, &ctx,
                       &filter_dims, data, input_shape, output_shape, params);

  if (input->type == kTfLiteInt8) {
    TFLITE_DCHECK_EQ(
        arm_avgpool_s8(&ctx, &pool_params, &input_dims,
                       micro::GetTensorData<int8_t>(input), &filter_dims,
                       &output_dims, micro::GetTensorData<int8_t>(output)),
        ARM_CMSIS_NN_SUCCESS);
  } else {
    TFLITE_DCHECK_EQ(
        arm_avgpool_s16(&ctx, &pool_params, &input_dims,
                        micro::GetTensorData<int16_t>(input), &filter_dims,
                        &output_dims, micro::GetTensorData<int16_t>(output)),
        ARM_CMSIS_NN_SUCCESS);
  }
}

TfLiteStatus MaxEvalQuantized(TfLiteContext* context, const TfLiteNode* node,
                              const TfLitePoolParams* params,
                              const OpData& data, const TfLiteEvalTensor* input,
                              TfLiteEvalTensor* output) {
  TFLITE_DCHECK((input->type == kTfLiteInt8) || (input->type == kTfLiteInt16));

  RuntimeShape input_shape = micro::GetTensorShape(input);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);

  RuntimeShape output_shape = micro::GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  cmsis_nn_dims input_dims;
  cmsis_nn_dims output_dims;
  cmsis_nn_pool_params pool_params;
  cmsis_nn_dims filter_dims;
  cmsis_nn_context ctx;

  PopulateCommonParams(context, &input_dims, &output_dims, &pool_params, &ctx,
                       &filter_dims, data, input_shape, output_shape, params);

  if (input->type == kTfLiteInt8) {
    TFLITE_DCHECK_EQ(
        arm_max_pool_s8(&ctx, &pool_params, &input_dims,
                        micro::GetTensorData<int8_t>(input), &filter_dims,
                        &output_dims, micro::GetTensorData<int8_t>(output)),
        ARM_CMSIS_NN_SUCCESS);
  } else {
    TFLITE_DCHECK_EQ(
        arm_max_pool_s16(&ctx, &pool_params, &input_dims,
                         micro::GetTensorData<int16_t>(input), &filter_dims,
                         &output_dims, micro::GetTensorData<int16_t>(output)),
        ARM_CMSIS_NN_SUCCESS);
  }

  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus MaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PoolingPrepare(context, node));
  // Set buffer index to a reset value
  static_cast<OpData*>(node->user_data)->buffer_idx = -1;
  return kTfLiteOk;
}

TfLiteStatus AveragePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(PoolingPrepare(context, node));

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kPoolingInputTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kPoolingOutputTensor);

  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    RuntimeShape input_shape = GetTensorShape(input);
    TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);

    RuntimeShape output_shape = GetTensorShape(output);
    TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

    const int depth = MatchingDim(input_shape, 3, output_shape, 3);
    const int output_width = output_shape.Dims(2);

    const int32_t buffer_size =
        input->type == kTfLiteInt16
            ? arm_avgpool_s16_get_buffer_size(output_width, depth)
            : arm_avgpool_s8_get_buffer_size(output_width, depth);

    auto* data = static_cast<OpData*>(node->user_data);
    if (buffer_size > 0) {
      TF_LITE_ENSURE_STATUS(context->RequestScratchBufferInArena(
          context, buffer_size, &data->buffer_idx));
    } else {
      data->buffer_idx = -1;
    }
  }

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  return kTfLiteOk;
}

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  if (input->type == kTfLiteFloat32) {
    AveragePoolingEvalFloat(context, node, params, &data.reference_op_data,
                            input, output);
  } else if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    AverageEvalQuantized(context, node, params, data, input, output);
  } else {
    MicroPrintf("Input type %s is not currently supported",
                TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus AverageEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TFLITE_DCHECK(input->type == kTfLiteInt8);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  AverageEvalQuantized(context, node, params, data, input, output);

  return kTfLiteOk;
}

TfLiteStatus AverageEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TFLITE_DCHECK(input->type == kTfLiteInt16);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  AverageEvalQuantized(context, node, params, data, input, output);

  return kTfLiteOk;
}
TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  if (input->type == kTfLiteFloat32) {
    MaxPoolingEvalFloat(context, node, params, &data.reference_op_data, input,
                        output);
  } else if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    MaxEvalQuantized(context, node, params, data, input, output);
  } else {
    MicroPrintf("Input type %s is not currently supported",
                TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus MaxEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TFLITE_DCHECK(input->type == kTfLiteInt8);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  MaxEvalQuantized(context, node, params, data, input, output);
  return kTfLiteOk;
}

TfLiteStatus MaxEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kPoolingInputTensor);
  TFLITE_DCHECK(input->type == kTfLiteInt16);
  TfLiteEvalTensor* output =
      micro::GetEvalOutput(context, node, kPoolingOutputTensor);

  MaxEvalQuantized(context, node, params, data, input, output);
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_AVERAGE_POOL_2D_INT8() {
  return tflite::micro::RegisterOp(Init, AveragePrepare, AverageEvalInt8);
}

TFLMRegistration Register_AVERAGE_POOL_2D_INT16() {
  return tflite::micro::RegisterOp(Init, AveragePrepare, AverageEvalInt16);
}

TFLMRegistration Register_AVERAGE_POOL_2D() {
  return tflite::micro::RegisterOp(Init, AveragePrepare, AverageEval);
}

TFLMRegistration Register_MAX_POOL_2D_INT8() {
  return tflite::micro::RegisterOp(Init, MaxPrepare, MaxEvalInt8);
}

TFLMRegistration Register_MAX_POOL_2D_INT16() {
  return tflite::micro::RegisterOp(Init, MaxPrepare, MaxEvalInt16);
}

TFLMRegistration Register_MAX_POOL_2D() {
  return tflite::micro::RegisterOp(Init, MaxPrepare, MaxEval);
}

}  // namespace tflite
