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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

extern constexpr int kBatchMatmulInputLhsTensor = 0;
extern constexpr int kBatchMatmulInputRhsTensor = 1;
extern constexpr int kBatchMatmulOutputTensor = 0;

struct QuantizationOpDataBatchMatmul {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;  // exponent

  // The range of the fused activation layer. For example for kNone and
  // int8_t these would be -128 and 127.
  int32_t output_activation_min;
  int32_t output_activation_max;

  int32_t lhs_zero_point;
  int32_t rhs_zero_point;
  int32_t output_zero_point;
};

struct OpDataBatchMatmul {
  QuantizationOpDataBatchMatmul* quantization;

  // Transpose tensors and state
  TfLiteEvalTensor* lhs_transposed_tensor;
  TfLiteEvalTensor* rhs_transposed_tensor;
  bool rhs_is_transposed;
  bool lhs_is_constant_tensor;
  bool rhs_is_constant_tensor;
};

TfLiteStatus ReshapeOutputTensor(TfLiteContext* context, TfLiteNode* node,
                                 const RuntimeShape& extended_lhs_shape,
                                 const RuntimeShape& extended_rhs_shape,
                                 bool adj_x, bool adj_y, int output_rank,
                                 TfLiteTensor* output) {
  int64_t orig_size = NumElements(output);

  // make sure the new output dims rank does not exceed the original rank
  TF_LITE_ENSURE(context, output_rank <= NumDimensions(output));

  // make sure output tensor dims are not in the FlatBuffer
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kBatchMatmulOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));

  // Fill in any broadcast dimensions.
  for (int i = 0; i < output_rank - 2; ++i) {
    const int lhs_dim = extended_lhs_shape.Dims(i);
    const int rhs_dim = extended_rhs_shape.Dims(i);
    int broadcast_dim = lhs_dim;
    if ((lhs_dim != rhs_dim) && (lhs_dim == 1)) {
      broadcast_dim = rhs_dim;
    }
    output->dims->data[i] = broadcast_dim;
  }
  // Fill in the matmul dimensions.
  int lhs_rows_index = adj_x ? output_rank - 1 : output_rank - 2;
  int rhs_cols_index = adj_y ? output_rank - 2 : output_rank - 1;

  output->dims->data[output_rank - 2] = extended_lhs_shape.Dims(lhs_rows_index);
  output->dims->data[output_rank - 1] = extended_rhs_shape.Dims(rhs_cols_index);
  output->dims->size = output_rank;

  // Check that output tensor has not been resized
  // since TFLM doesn't support tensor resizing.
  TF_LITE_ENSURE_EQ(context, orig_size, NumElements(output));

  return kTfLiteOk;
}

template <typename T>
void TransposeRowsColumnsImpl(const TfLiteEvalTensor& tensor_in,
                              TfLiteEvalTensor* tensor_out) {
  const T* input = tflite::micro::GetTensorData<T>(&tensor_in);
  T* output = tflite::micro::GetTensorData<T>(tensor_out);
  RuntimeShape transposed_shape(tflite::micro::GetTensorShape(&tensor_in));
  RuntimeShape shape(transposed_shape);
  TransposeParams params;
  const int rank = shape.DimensionsCount();
  params.perm_count = rank;
  for (int i = 0; i < rank - 2; ++i) {
    params.perm[i] = i;
  }
  // Transpose the last two dimensions.
  params.perm[rank - 2] = rank - 1;
  params.perm[rank - 1] = rank - 2;
  transposed_shape.SetDim(rank - 1, shape.Dims(rank - 2));
  transposed_shape.SetDim(rank - 2, shape.Dims(rank - 1));
  reference_ops::Transpose(params, shape, input, transposed_shape, output);
}

TfLiteStatus TransposeRowsColumns(const TfLiteEvalTensor& tensor_in,
                                  TfLiteEvalTensor* tensor_out) {
  if (tensor_in.type == kTfLiteFloat32) {
    TransposeRowsColumnsImpl<float>(tensor_in, tensor_out);
    return kTfLiteOk;
  } else if (tensor_in.type == kTfLiteInt8) {
    TransposeRowsColumnsImpl<int8_t>(tensor_in, tensor_out);
    return kTfLiteOk;
  } else if (tensor_in.type == kTfLiteInt16) {
    TransposeRowsColumnsImpl<int16_t>(tensor_in, tensor_out);
    return kTfLiteOk;
  } else {
    MicroPrintf(
        "BATCH_MATMUL can only transpose tensors with FLOAT32, INT8, INT16 "
        "type.");
  }
  return kTfLiteError;
}

RuntimeShape SwapRowColumnDims(const RuntimeShape& shape) {
  RuntimeShape swapped_shape(shape);
  const int32_t dims = shape.DimensionsCount();
  swapped_shape.SetDim(dims - 2, shape.Dims(dims - 1));
  swapped_shape.SetDim(dims - 1, shape.Dims(dims - 2));
  return swapped_shape;
}

TFLMRegistration Register_BATCH_MATMUL();

#if defined(ARDUINO)
// Returns a TFLMRegistration struct for kernel variant that only supports
// int8 matrix multiplication and uses the latency optimized
// implementations.
TFLMRegistration Register_BATCH_MATMUL_INT8();

// Returns a TFLMRegistration struct for kernel variant that only supports
// int16 matrix multiplication and uses the latency optimized
// implementations.
TFLMRegistration Register_BATCH_MATMUL_INT16();

#else
inline TFLMRegistration Register_BATCH_MATMUL_INT8() {
  return Register_BATCH_MATMUL();
}

inline TFLMRegistration Register_BATCH_MATMUL_INT16() {
  return Register_BATCH_MATMUL();
}
#endif  // defined(ARDUINO)

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_BATCH_MATMUL_H_
