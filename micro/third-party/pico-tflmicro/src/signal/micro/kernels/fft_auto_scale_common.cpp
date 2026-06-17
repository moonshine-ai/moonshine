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
#include "signal/micro/kernels/fft_auto_scale_kernel.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kScaleBitTensor = 1;

TfLiteStatus FftAutoScalePrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* scale_bit =
      micro_context->AllocateTempOutputTensor(node, kScaleBitTensor);
  TF_LITE_ENSURE(context, scale_bit != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(scale_bit), 0);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, scale_bit->type, kTfLiteInt32);

  micro_context->DeallocateTempTfLiteTensor(scale_bit);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

}  // namespace tflite
