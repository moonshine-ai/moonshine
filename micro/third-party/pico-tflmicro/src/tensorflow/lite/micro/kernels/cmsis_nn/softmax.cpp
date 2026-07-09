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

#include "tensorflow/lite/micro/kernels/softmax.h"

#include "third_party/cmsis_nn/Include/arm_nnfunctions.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct CMSISNNSoftmaxParams {
  SoftmaxParams softmax_params;
  int32_t num_rows;
  int32_t row_size;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(CMSISNNSoftmaxParams));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_MSG(
      context,
      input->type == output->type ||
          (input->type == kTfLiteInt8 && output->type == kTfLiteInt16),
      "Input and output data types are not supported together.");
  TF_LITE_ENSURE_MSG(context,
                     input->type == kTfLiteFloat32 ||
                         input->type == kTfLiteInt16 ||
                         input->type == kTfLiteInt8,
                     "Input data type not supported");

  TF_LITE_ENSURE(context, node->user_data != nullptr);
  CMSISNNSoftmaxParams* op_data =
      static_cast<CMSISNNSoftmaxParams*>(node->user_data);

  auto* params = static_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  auto ret_val = CalculateSoftmaxParams(context, input, output, params,
                                        &op_data->softmax_params);

  const auto input_shape = GetTensorShape(input);
  const auto output_shape = GetTensorShape(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  op_data->num_rows = outer_size;
  op_data->row_size = depth;

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return ret_val;
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  const CMSISNNSoftmaxParams op_data =
      *static_cast<const CMSISNNSoftmaxParams*>(node->user_data);

  switch (input->type) {
    case kTfLiteFloat32: {
      tflite::reference_ops::Softmax(
          op_data.softmax_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      if (output->type == kTfLiteInt8) {
        arm_softmax_s8(tflite::micro::GetTensorData<int8_t>(input),
                       op_data.num_rows, op_data.row_size,
                       op_data.softmax_params.input_multiplier,
                       op_data.softmax_params.input_left_shift,
                       op_data.softmax_params.diff_min,
                       tflite::micro::GetTensorData<int8_t>(output));
      } else {
        arm_softmax_s8_s16(tflite::micro::GetTensorData<int8_t>(input),
                           op_data.num_rows, op_data.row_size,
                           op_data.softmax_params.input_multiplier,
                           op_data.softmax_params.input_left_shift,
                           op_data.softmax_params.diff_min,
                           tflite::micro::GetTensorData<int16_t>(output));
      }
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      const cmsis_nn_softmax_lut_s16 softmax_params = {
          .exp_lut = op_data.softmax_params.exp_lut,
          .one_by_one_lut = op_data.softmax_params.one_over_one_plus_x_lut};

      TFLITE_DCHECK_EQ(
          arm_softmax_s16(
              tflite::micro::GetTensorData<int16_t>(input), op_data.num_rows,
              op_data.row_size, op_data.softmax_params.input_multiplier,
              op_data.softmax_params.input_left_shift, &softmax_params,
              tflite::micro::GetTensorData<int16_t>(output)),
          ARM_CMSIS_NN_SUCCESS);
      return kTfLiteOk;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
}

TfLiteStatus SoftmaxEvalInt8(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  const CMSISNNSoftmaxParams op_data =
      *static_cast<const CMSISNNSoftmaxParams*>(node->user_data);

  arm_softmax_s8(tflite::micro::GetTensorData<int8_t>(input), op_data.num_rows,
                 op_data.row_size, op_data.softmax_params.input_multiplier,
                 op_data.softmax_params.input_left_shift,
                 op_data.softmax_params.diff_min,
                 tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

TfLiteStatus SoftmaxEvalInt8_Int16(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  const CMSISNNSoftmaxParams op_data =
      *static_cast<const CMSISNNSoftmaxParams*>(node->user_data);

  arm_softmax_s8_s16(
      tflite::micro::GetTensorData<int8_t>(input), op_data.num_rows,
      op_data.row_size, op_data.softmax_params.input_multiplier,
      op_data.softmax_params.input_left_shift, op_data.softmax_params.diff_min,
      tflite::micro::GetTensorData<int16_t>(output));

  return kTfLiteOk;
}

TfLiteStatus SoftmaxEvalInt16(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  TFLITE_DCHECK(node->user_data != nullptr);
  const CMSISNNSoftmaxParams op_data =
      *static_cast<const CMSISNNSoftmaxParams*>(node->user_data);

  const cmsis_nn_softmax_lut_s16 softmax_params = {
      .exp_lut = op_data.softmax_params.exp_lut,
      .one_by_one_lut = op_data.softmax_params.one_over_one_plus_x_lut};

  TFLITE_DCHECK_EQ(
      arm_softmax_s16(tflite::micro::GetTensorData<int16_t>(input),
                      op_data.num_rows, op_data.row_size,
                      op_data.softmax_params.input_multiplier,
                      op_data.softmax_params.input_left_shift, &softmax_params,
                      tflite::micro::GetTensorData<int16_t>(output)),
      ARM_CMSIS_NN_SUCCESS);

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_SOFTMAX() {
  return tflite::micro::RegisterOp(Init, Prepare, SoftmaxEval);
}

TFLMRegistration Register_SOFTMAX_INT8() {
  return tflite::micro::RegisterOp(Init, Prepare, SoftmaxEvalInt8);
}

TFLMRegistration Register_SOFTMAX_INT8_INT16() {
  return tflite::micro::RegisterOp(Init, Prepare, SoftmaxEvalInt8_Int16);
}

TFLMRegistration Register_SOFTMAX_INT16() {
  return tflite::micro::RegisterOp(Init, Prepare, SoftmaxEvalInt16);
}

}  // namespace tflite
