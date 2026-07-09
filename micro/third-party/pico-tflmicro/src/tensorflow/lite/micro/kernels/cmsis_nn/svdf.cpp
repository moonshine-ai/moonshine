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

#include "tensorflow/lite/micro/kernels/svdf.h"

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/activation_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

struct CmsisNnOpDataSvdf {
  int32_t effective_scale_1_a;
  int32_t effective_scale_2_a;
  // b versions of each scale are kept at int since the numbers are just the
  // shift value - typically between [-32, 32].
  int effective_scale_1_b;
  int effective_scale_2_b;
  int scratch_tensor_index;
#if defined(KERNELS_OPTIMIZED_FOR_SIZE)
  int scratch_weight_tensor_index;
#endif
  int scratch_output_tensor_index;

  // Cached tensor zero point values for quantized operations.
  int input_zero_point;
  int output_zero_point;
  int activation_state_zero_point;
  int32_t* kernel_sums;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(CmsisNnOpDataSvdf));
}

TfLiteStatus CmsisNnPrepareSvdf(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);

  const auto* params = static_cast<const TfLiteSVDFParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  // Validate Tensor Inputs (dtype depends on quantization):
  // [0] = Input, {2, batch_size, input_size}
  // [1] = Weights Feature, {2, num_filters, input_size}
  // [2] = Weights Time, {2, num_filters, memory_size}
  // [3] = Bias (optional), {1, num_units}
  // [4] = Activation State (variable),
  //         {2, batch_size, memory_size * num_filters}
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kSvdfInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* weights_feature =
      micro_context->AllocateTempInputTensor(node, kSvdfWeightsFeatureTensor);
  TF_LITE_ENSURE(context, weights_feature != nullptr);
  TfLiteTensor* weights_time =
      micro_context->AllocateTempInputTensor(node, kSvdfWeightsTimeTensor);
  TF_LITE_ENSURE(context, weights_time != nullptr);
  TfLiteTensor* bias =
      micro_context->AllocateTempInputTensor(node, kSvdfBiasTensor);
  TfLiteTensor* activation_state = micro_context->AllocateTempInputTensor(
      node, kSvdfInputActivationStateTensor);
  TF_LITE_ENSURE(context, activation_state != nullptr);

  // Define input constants based on input tensor definition above:
  const int rank = params->rank;
  const int input_size = input->dims->data[1];
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Validate Input Tensor:
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);

  // Validate Tensor Output:
  // [0] = float/int8_t, {2, batch_size, num_units}
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kSvdfOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 2);
  TF_LITE_ENSURE_EQ(context, output->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, output->dims->data[1], num_units);

  // Validate Weights Feature Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_feature), 2);
  TF_LITE_ENSURE_EQ(context, weights_feature->dims->data[1], input_size);

  // Validate Weights Time Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights_time), 2);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[1], memory_size);

  // Validate Optional Bias Input Tensor:
  if (bias != nullptr) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
  }

  // Validate Activation State Input Tensor:
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[0], batch_size);
  TF_LITE_ENSURE_EQ(context, activation_state->dims->data[1],
                    memory_size * num_filters);
  // Since is_variable is not part of TFLiteEvalTensor, check is_variable here.
  TF_LITE_ENSURE_EQ(context, activation_state->is_variable, true);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);

  TFLITE_DCHECK(node->user_data != nullptr);
  CmsisNnOpDataSvdf* data = static_cast<CmsisNnOpDataSvdf*>(node->user_data);

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteInt8);
    TF_LITE_ENSURE(context, (weights_time->type == kTfLiteInt16) ||
                                (weights_time->type == kTfLiteInt8));
    TF_LITE_ENSURE(context, (activation_state->type == kTfLiteInt16) ||
                                (activation_state->type == kTfLiteInt8));
    if (bias != nullptr) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteInt32);
    }

    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt8);

    const double effective_scale_1 = static_cast<double>(
        input->params.scale * weights_feature->params.scale /
        activation_state->params.scale);
    const double effective_scale_2 =
        static_cast<double>(activation_state->params.scale *
                            weights_time->params.scale / output->params.scale);

    // TODO(b/162018098): Use TF_LITE_ENSURE_NEAR when it is ready.
    // TODO(#1751): account for optional bias tensor
    TF_LITE_ENSURE(
        context,
        std::abs(static_cast<double>(bias->params.scale) -
                 static_cast<double>(activation_state->params.scale *
                                     weights_time->params.scale)) < 1e-5);

    QuantizeMultiplier(effective_scale_1, &(data->effective_scale_1_a),
                       &(data->effective_scale_1_b));
    QuantizeMultiplier(effective_scale_2, &(data->effective_scale_2_a),
                       &(data->effective_scale_2_b));

    data->input_zero_point = input->params.zero_point;
    data->output_zero_point = output->params.zero_point;
    data->activation_state_zero_point = activation_state->params.zero_point;

    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);

    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, batch_size * num_filters * sizeof(int32_t),
        &(data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);

    const TfLiteStatus scratch_output_status =
        context->RequestScratchBufferInArena(
            context, batch_size * num_units * sizeof(int32_t),
            &(data->scratch_output_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_output_status);

    cmsis_nn_dims weights_feature_dims;
    weights_feature_dims.n = num_filters;
    weights_feature_dims.h = input_size;

    const int32_t buf_size = arm_svdf_s8_get_buffer_size(&weights_feature_dims);

    if (buf_size > 0) {
#if defined(KERNELS_OPTIMIZED_FOR_SPEED)
      data->kernel_sums = static_cast<int32_t*>(
          context->AllocatePersistentBuffer(context, buf_size));

      arm_vector_sum_s8(data->kernel_sums, input_size, num_filters,
                        GetTensorData<int8_t>(weights_feature),
                        -data->input_zero_point,
                        -data->activation_state_zero_point, nullptr);
#elif defined(KERNELS_OPTIMIZED_FOR_SIZE)
      const TfLiteStatus scratch_kernel_status =
          context->RequestScratchBufferInArena(
              context, buf_size, &(data->scratch_weight_tensor_index));
      TF_LITE_ENSURE_OK(context, scratch_kernel_status);
#else
      MicroPrintf(
          "Either KERNELS_OPTIMIZED_FOR_SIZE or KERNELS_OPTIMIZED_FOR_SPEED "
          "must be defined");
      return kTfLiteError;
#endif
    }

  } else {
    TF_LITE_ENSURE_EQ(context, weights_feature->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, weights_time->type, kTfLiteFloat32);
    TF_LITE_ENSURE_EQ(context, activation_state->type, kTfLiteFloat32);
    if (bias != nullptr) {
      TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteFloat32);
    }
    TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);

    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);
    const TfLiteStatus scratch_status = context->RequestScratchBufferInArena(
        context, batch_size * num_filters * sizeof(float),
        &(data->scratch_tensor_index));
    TF_LITE_ENSURE_OK(context, scratch_status);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(weights_feature);
  micro_context->DeallocateTempTfLiteTensor(weights_time);
  micro_context->DeallocateTempTfLiteTensor(activation_state);
  micro_context->DeallocateTempTfLiteTensor(output);
  // TODO(#1751): account for optional bias tensor
  micro_context->DeallocateTempTfLiteTensor(bias);
  return kTfLiteOk;
}

TfLiteStatus EvalIntegerSVDF(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteEvalTensor* input_tensor,
                             const TfLiteEvalTensor* weights_feature_tensor,
                             const TfLiteEvalTensor* weights_time_tensor,
                             const TfLiteEvalTensor* bias_tensor,
                             const TfLiteSVDFParams* params,
                             TfLiteEvalTensor* activation_state_tensor,
                             TfLiteEvalTensor* output_tensor,
                             const CmsisNnOpDataSvdf& data) {
  cmsis_nn_dims input_dims;
  input_dims.n = input_tensor->dims->data[0];
  input_dims.h = input_tensor->dims->data[1];

  cmsis_nn_dims weights_feature_dims;
  weights_feature_dims.n = weights_feature_tensor->dims->data[0];
  weights_feature_dims.h = weights_feature_tensor->dims->data[1];

  cmsis_nn_dims weights_time_dims;
  weights_time_dims.n = weights_time_tensor->dims->data[0];
  weights_time_dims.h = weights_time_tensor->dims->data[1];

  cmsis_nn_dims bias_dims;
  bias_dims.n = bias_tensor->dims->data[0];

  cmsis_nn_dims state_dims;
  state_dims.n = bias_tensor->dims->data[0];
  state_dims.h = bias_tensor->dims->data[1];

  cmsis_nn_dims output_dims;
  output_dims.n = output_tensor->dims->data[0];
  output_dims.h = output_tensor->dims->data[1];

  cmsis_nn_svdf_params svdf_params;
  svdf_params.rank = params->rank;
  svdf_params.input_offset = data.input_zero_point;
  svdf_params.output_offset = data.output_zero_point;

  svdf_params.input_activation.min = INT16_MIN;
  svdf_params.input_activation.max = INT16_MAX;

  svdf_params.output_activation.min = INT8_MIN;
  svdf_params.output_activation.max = INT8_MAX;

  cmsis_nn_per_tensor_quant_params in_quant_params;
  in_quant_params.multiplier = data.effective_scale_1_a;
  in_quant_params.shift = data.effective_scale_1_b;

  cmsis_nn_per_tensor_quant_params out_quant_params;
  out_quant_params.multiplier = data.effective_scale_2_a;
  out_quant_params.shift = data.effective_scale_2_b;

  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(context->GetScratchBuffer != nullptr);

  cmsis_nn_context scratch_ctx;
  scratch_ctx.buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_tensor_index));

  cmsis_nn_context scratch_output_ctx;
  scratch_output_ctx.buf = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.scratch_output_tensor_index));

  int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output_tensor);

  switch (weights_time_tensor->type) {
    case kTfLiteInt8: {
      cmsis_nn_context ctx;

#if defined(KERNELS_OPTIMIZED_FOR_SPEED)
      ctx.buf = data.kernel_sums;
#elif defined(KERNELS_OPTIMIZED_FOR_SIZE)
      ctx.buf = static_cast<int32_t*>(
          context->GetScratchBuffer(context, data.scratch_weight_tensor_index));

      const int input_size = input_tensor->dims->data[1];
      const int num_filters = weights_feature_tensor->dims->data[0];

      arm_vector_sum_s8(
          static_cast<int32_t*>(ctx.buf), input_size, num_filters,
          tflite::micro::GetTensorData<int8_t>(weights_feature_tensor),
          -data.input_zero_point, -data.activation_state_zero_point, nullptr);
#endif

      arm_svdf_s8(
          &ctx, &scratch_ctx, &scratch_output_ctx, &svdf_params,
          &in_quant_params, &out_quant_params, &input_dims,
          tflite::micro::GetTensorData<int8_t>(input_tensor), &state_dims,
          tflite::micro::GetTensorData<int8_t>(activation_state_tensor),
          &weights_feature_dims,
          tflite::micro::GetTensorData<int8_t>(weights_feature_tensor),
          &weights_time_dims,
          tflite::micro::GetTensorData<int8_t>(weights_time_tensor), &bias_dims,
          tflite::micro::GetTensorData<int32_t>(bias_tensor), &output_dims,
          output_data);
      return kTfLiteOk;
    }

    case kTfLiteInt16: {
      arm_svdf_state_s16_s8(
          &scratch_ctx, &scratch_output_ctx, &svdf_params, &in_quant_params,
          &out_quant_params, &input_dims,
          tflite::micro::GetTensorData<int8_t>(input_tensor), &state_dims,
          tflite::micro::GetTensorData<int16_t>(activation_state_tensor),
          &weights_feature_dims,
          tflite::micro::GetTensorData<int8_t>(weights_feature_tensor),
          &weights_time_dims,
          tflite::micro::GetTensorData<int16_t>(weights_time_tensor),
          &bias_dims, tflite::micro::GetTensorData<int32_t>(bias_tensor),
          &output_dims, output_data);
      return kTfLiteOk;
    }

    default:
      MicroPrintf("Could not find matching function for type %s.",
                  TfLiteTypeGetName(weights_time_tensor->type));
      return kTfLiteError;
  }
}

TfLiteStatus EvalSvdf(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const CmsisNnOpDataSvdf& data =
      *(static_cast<const CmsisNnOpDataSvdf*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  switch (weights_time->type) {
    case kTfLiteFloat32: {
      EvalFloatSvdfReference(
          context, node, input, weights_feature, weights_time, bias, params,
          data.scratch_tensor_index, activation_state, output);
      return kTfLiteOk;
    }

    case kTfLiteInt8:
    case kTfLiteInt16: {
      return EvalIntegerSVDF(context, node, input, weights_feature,
                             weights_time, bias, params, activation_state,
                             output, data);
    }

    default:
      MicroPrintf("Type %s not currently supported.",
                  TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus EvalSvdfInt8(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  TFLITE_DCHECK(node->user_data != nullptr);
  const CmsisNnOpDataSvdf& data =
      *(static_cast<const CmsisNnOpDataSvdf*>(node->user_data));

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kSvdfInputTensor);
  const TfLiteEvalTensor* weights_feature =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsFeatureTensor);
  const TfLiteEvalTensor* weights_time =
      tflite::micro::GetEvalInput(context, node, kSvdfWeightsTimeTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 5)
          ? tflite::micro::GetEvalInput(context, node, kSvdfBiasTensor)
          : nullptr;
  TfLiteEvalTensor* activation_state = tflite::micro::GetMutableEvalInput(
      context, node, kSvdfInputActivationStateTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kSvdfOutputTensor);

  TFLITE_DCHECK((weights_time->type == kTfLiteInt8) ||
                (weights_time->type == kTfLiteInt16));
  // Because of the TODO mentioned below, the int16 weight data type is not
  // split into a separate registration.
  // TODO(#523): remove 16-bit code when no longer needed.
  return EvalIntegerSVDF(context, node, input, weights_feature, weights_time,
                         bias, params, activation_state, output, data);
}

}  // namespace

TFLMRegistration Register_SVDF() {
  return tflite::micro::RegisterOp(Init, CmsisNnPrepareSvdf, EvalSvdf);
}

TFLMRegistration Register_SVDF_INT8() {
  return tflite::micro::RegisterOp(Init, CmsisNnPrepareSvdf, EvalSvdfInt8);
}

}  // namespace tflite
