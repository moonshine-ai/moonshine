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
#include "signal/micro/kernels/filter_bank_square_root.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {

constexpr int kInputTensor = 0;
constexpr int kScaleBitsTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus FilterBankSquareRootPrepare(TfLiteContext* context,
                                         TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* scale_bits =
      micro_context->AllocateTempInputTensor(node, kScaleBitsTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, scale_bits != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(scale_bits), 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt64);
  TF_LITE_ENSURE_TYPES_EQ(context, scale_bits->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt32);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(scale_bits);
  return kTfLiteOk;
}

}  // namespace tflite
