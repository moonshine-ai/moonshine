/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/src/filter_bank_log.h"

#include <stdint.h>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_context.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
constexpr int kInputCorrectionBitsIndex = 0;  // 'input_correction_bits'
constexpr int kOutputScaleIndex = 1;          // 'output_scale'

struct TFLMSignalLogParams {
  int input_correction_bits;
  int output_scale;
};

void* FilterBankLogInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);

  auto* params = static_cast<TFLMSignalLogParams*>(
      context->AllocatePersistentBuffer(context, sizeof(TFLMSignalLogParams)));

  if (params == nullptr) {
    return nullptr;
  }
  tflite::FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer),
                                length);

  params->input_correction_bits = fbw.ElementAsInt32(kInputCorrectionBitsIndex);
  params->output_scale = fbw.ElementAsInt32(kOutputScaleIndex);
  return params;
}

TfLiteStatus FilterBankLogPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus FilterBankLogEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TFLMSignalLogParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const uint32_t* input_data = tflite::micro::GetTensorData<uint32_t>(input);
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  int num_channels = input->dims->data[0];
  tflm_signal::FilterbankLog(input_data, num_channels, params->output_scale,
                             params->input_correction_bits, output_data);
  return kTfLiteOk;
}

}  // namespace

namespace tflm_signal {

TFLMRegistration* Register_FILTER_BANK_LOG() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      FilterBankLogInit, FilterBankLogPrepare, FilterBankLogEval);
  return &r;
}

}  // namespace tflm_signal

}  // namespace tflite
