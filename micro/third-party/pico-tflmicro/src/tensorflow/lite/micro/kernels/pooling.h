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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_POOLING_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_POOLING_H_

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

extern const int kPoolingInputTensor;
extern const int kPoolingOutputTensor;

struct OpDataPooling {
  TfLitePaddingValues padding;
  int32_t activation_min;
  int32_t activation_max;
  float activation_min_f32;
  float activation_max_f32;
};

TfLiteStatus CalculateOpDataPooling(const TfLiteContext* context,
                                    const TfLitePoolParams* params,
                                    const TfLiteTensor* input,
                                    const TfLiteTensor* output,
                                    OpDataPooling* data);

TfLiteStatus PoolingPrepare(TfLiteContext* context, TfLiteNode* node);

void AveragePoolingEvalFloat(const TfLiteContext* context,
                             const TfLiteNode* node,
                             const TfLitePoolParams* params,
                             const OpDataPooling* data,
                             const TfLiteEvalTensor* input,
                             TfLiteEvalTensor* output);

template <typename T>
void AveragePoolingEvalQuantized(TfLiteContext* context, const TfLiteNode* node,
                                 const TfLitePoolParams* params,
                                 const OpDataPooling* data,
                                 const TfLiteEvalTensor* input,
                                 TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteInt8 || input->type == kTfLiteInt16);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  reference_integer_ops::AveragePool(op_params,
                                     tflite::micro::GetTensorShape(input),
                                     tflite::micro::GetTensorData<T>(input),
                                     tflite::micro::GetTensorShape(output),
                                     tflite::micro::GetTensorData<T>(output));
}

void MaxPoolingEvalFloat(TfLiteContext* context, TfLiteNode* node,
                         TfLitePoolParams* params, const OpDataPooling* data,
                         const TfLiteEvalTensor* input,
                         TfLiteEvalTensor* output);

template <typename T>
void MaxPoolingEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                             TfLitePoolParams* params,
                             const OpDataPooling* data,
                             const TfLiteEvalTensor* input,
                             TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteInt8 || input->type == kTfLiteInt16);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  reference_integer_ops::MaxPool(op_params,
                                 tflite::micro::GetTensorShape(input),
                                 tflite::micro::GetTensorData<T>(input),
                                 tflite::micro::GetTensorShape(output),
                                 tflite::micro::GetTensorData<T>(output));
}

#if defined(ARDUINO) || defined(XTENSA)
TFLMRegistration Register_AVERAGE_POOL_2D_INT8();

TFLMRegistration Register_MAX_POOL_2D_INT8();

TFLMRegistration Register_AVERAGE_POOL_2D_INT16();

TFLMRegistration Register_MAX_POOL_2D_INT16();
#else
inline TFLMRegistration Register_AVERAGE_POOL_2D_INT8() {
  return tflite::Register_AVERAGE_POOL_2D();
}

inline TFLMRegistration Register_MAX_POOL_2D_INT8() {
  return tflite::Register_MAX_POOL_2D();
}

inline TFLMRegistration Register_AVERAGE_POOL_2D_INT16() {
  return tflite::Register_AVERAGE_POOL_2D();
}

inline TFLMRegistration Register_MAX_POOL_2D_INT16() {
  return tflite::Register_MAX_POOL_2D();
}
#endif
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_POOLING_H_
