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

// Ops that looks up items from matrix.
//
// Input:
//     Tensor[0]: Row numbers to lookup, dim.size == 1, int32
//     Tensor[1]: 2-dimensional matrix of multi-dimensional items
//                dim.size >= 2, all items are INT8 or FLOAT32.
//                first dimension is row, second dimension is column.
//
// Output:
//   Output.dim[0] == Tensor[0].dim[0], num of lookups
//   Output.dim[1] == Tensor[1].dim[1],  num of items per row
//   Each item in output is a raw bytes copy of the corresponding item in input,
//   or a dequantized value in the case of a INT8 input.
//   When indices are out of bound, the ops will not succeed.
//

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor_0 = 0;
constexpr int kInputTensor_1 = 1;
constexpr int kOutputTensor = 0;

struct OpData {
  float scale;         // quantization scale for tensor 1
  size_t num_columns;  // number of columns after flattening tensor 1 into 2D
};

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTensor* tensor_1,
                             const TfLiteTensor* output) {
  node->user_data = context->AllocatePersistentBuffer(context, sizeof(OpData));
  OpData* op_data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data != nullptr);

  if (tensor_1->type == kTfLiteInt8 && output->type == kTfLiteFloat32) {
    TF_LITE_ENSURE_EQ(context, tensor_1->params.zero_point, 0);
    op_data->scale = tensor_1->params.scale;
  }

  op_data->num_columns = NumElements(tensor_1) / tensor_1->dims->data[0];

  return kTfLiteOk;
}

TfLiteStatus EmbeddingLookUpPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* lookup =
      micro_context->AllocateTempInputTensor(node, kInputTensor_0);
  TF_LITE_ENSURE(context, lookup != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);

  TfLiteTensor* value =
      micro_context->AllocateTempInputTensor(node, kInputTensor_1);
  TF_LITE_ENSURE(context, value != nullptr);
  TF_LITE_ENSURE(context, NumDimensions(value) >= 2);
  TF_LITE_ENSURE(context,
                 value->type == kTfLiteFloat32 || value->type == kTfLiteInt8);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  if (value->type == kTfLiteFloat32) {
    TF_LITE_ENSURE(context, output->type == kTfLiteFloat32);
  } else {
    TF_LITE_ENSURE(
        context, output->type == kTfLiteFloat32 || output->type == kTfLiteInt8);
  }

  // make sure output dimensions size can hold the new dimension data
  TF_LITE_ENSURE(context, output->dims->size >= NumDimensions(value));
  // make the output tensor dimensions mutable
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));
  // set the new output dimensions
  output->dims->data[0] = SizeOfDimension(lookup, 0);
  output->dims->data[1] = SizeOfDimension(value, 1);
  for (int i = 2; i < NumDimensions(value); i++) {
    output->dims->data[i] = SizeOfDimension(value, i);
  }
  // check the new output dimensions do not exceed the output data buffer size
  size_t new_dims_size = NumElements(output) * TfLiteTypeGetSize(output->type);
  TF_LITE_ENSURE(context, new_dims_size <= output->bytes);

  TF_LITE_ENSURE_OK(context, CalculateOpData(context, node, value, output));

  micro_context->DeallocateTempTfLiteTensor(lookup);
  micro_context->DeallocateTempTfLiteTensor(value);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus EvalSimple(const OpData& op_data, const TfLiteEvalTensor* lookup,
                        const TfLiteEvalTensor* value,
                        TfLiteEvalTensor* output) {
  const int num_rows = value->dims->data[0];
  if (num_rows == 0) {
    // Propagate empty tensor if input is empty
    return kTfLiteOk;
  }
  const size_t row_bytes = op_data.num_columns * TfLiteTypeGetSize(value->type);

  int8_t* output_raw = tflite::micro::GetTensorData<int8_t>(output);
  const int8_t* value_raw = tflite::micro::GetTensorData<int8_t>(value);
  const int32_t* lookup_data = tflite::micro::GetTensorData<int32_t>(lookup);
  for (int i = 0; i < lookup->dims->data[0]; i++) {
    int32_t idx = lookup_data[i];
    if (idx >= num_rows || idx < 0) {
      MicroPrintf(
          "EMBEDDING_LOOKUP: index out of bounds. "
          "Got %d, and bounds are [0, %d]",
          idx, num_rows - 1);
      return kTfLiteError;
    } else {
      std::memcpy(output_raw + i * row_bytes, value_raw + idx * row_bytes,
                  row_bytes);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(const OpData& op_data, const TfLiteEvalTensor* lookup,
                        const TfLiteEvalTensor* value,
                        TfLiteEvalTensor* output) {
  const int num_rows = value->dims->data[0];
  const size_t num_colums = op_data.num_columns;

  float* output_ptr = tflite::micro::GetTensorData<float>(output);
  const int8_t* value_ptr = tflite::micro::GetTensorData<int8_t>(value);
  const int32_t* lookup_data = tflite::micro::GetTensorData<int32_t>(lookup);

  for (int i = 0; i < lookup->dims->data[0]; i++) {
    int32_t idx = lookup_data[i];
    if (idx >= num_rows || idx < 0) {
      MicroPrintf(
          "EMBEDDING_LOOKUP: index out of bounds. "
          "Got %d, and bounds are [0, %d]",
          idx, num_rows - 1);
      return kTfLiteError;
    } else {
      // Dequantize embedding values.
      Dequantize(&value_ptr[idx * num_colums], num_colums, op_data.scale, 0,
                 &output_ptr[i * num_colums]);
    }
  }

  return kTfLiteOk;
}

TfLiteStatus EmbeddingLookUpEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* lookup =
      tflite::micro::GetEvalInput(context, node, kInputTensor_0);
  const TfLiteEvalTensor* value =
      tflite::micro::GetEvalInput(context, node, kInputTensor_1);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  OpData& op_data = *static_cast<OpData*>(node->user_data);

  switch (value->type) {
    case kTfLiteFloat32:
      return EvalSimple(op_data, lookup, value, output);
    case kTfLiteInt8:
      if (output->type == kTfLiteFloat32) {
        return EvalHybrid(op_data, lookup, value, output);
      } else {
        return EvalSimple(op_data, lookup, value, output);
      }
    default:
      MicroPrintf("EMBEDDING_LOOKUP only supports FLOAT32 and INT8, got %s.",
                  TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
}

}  // namespace

TFLMRegistration Register_EMBEDDING_LOOKUP() {
  return tflite::micro::RegisterOp(nullptr, EmbeddingLookUpPrepare,
                                   EmbeddingLookUpEval);
}

}  // namespace tflite
