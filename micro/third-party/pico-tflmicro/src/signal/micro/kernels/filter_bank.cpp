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

#include "signal/src/filter_bank.h"

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
constexpr int kWeightTensor = 1;
constexpr int kUnweightTensor = 2;
constexpr int kChFreqStartsTensor = 3;
constexpr int kChWeightStartsTensor = 4;
constexpr int kChannelWidthsTensor = 5;
constexpr int kOutputTensor = 0;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
constexpr int kNumChannelsIndex = 0;  // 'num_channels'

struct TFLMSignalFilterBankParams {
  tflm_signal::FilterbankConfig config;
  uint64_t* work_area;
};

void* FilterBankInit(TfLiteContext* context, const char* buffer,
                     size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);

  auto* params = static_cast<TFLMSignalFilterBankParams*>(
      context->AllocatePersistentBuffer(context,
                                        sizeof(TFLMSignalFilterBankParams)));
  if (params == nullptr) {
    return nullptr;
  }

  tflite::FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer),
                                length);
  params->config.num_channels = fbw.ElementAsInt32(kNumChannelsIndex);

  params->work_area = static_cast<uint64_t*>(context->AllocatePersistentBuffer(
      context, (params->config.num_channels + 1) * sizeof(uint64_t)));

  if (params->work_area == nullptr) {
    return nullptr;
  }

  return params;
}

TfLiteStatus FilterBankPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 6);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt32);
  micro_context->DeallocateTempTfLiteTensor(input);

  input = micro_context->AllocateTempInputTensor(node, kWeightTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  micro_context->DeallocateTempTfLiteTensor(input);

  input = micro_context->AllocateTempInputTensor(node, kUnweightTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  micro_context->DeallocateTempTfLiteTensor(input);

  input = micro_context->AllocateTempInputTensor(node, kChFreqStartsTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  micro_context->DeallocateTempTfLiteTensor(input);

  input = micro_context->AllocateTempInputTensor(node, kChWeightStartsTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  micro_context->DeallocateTempTfLiteTensor(input);

  input = micro_context->AllocateTempInputTensor(node, kChannelWidthsTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  micro_context->DeallocateTempTfLiteTensor(input);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt64);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus FilterBankEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TFLMSignalFilterBankParams*>(node->user_data);

  const TfLiteEvalTensor* input0 =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, kWeightTensor);
  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, kUnweightTensor);
  const TfLiteEvalTensor* input3 =
      tflite::micro::GetEvalInput(context, node, kChFreqStartsTensor);
  const TfLiteEvalTensor* input4 =
      tflite::micro::GetEvalInput(context, node, kChWeightStartsTensor);
  const TfLiteEvalTensor* input5 =
      tflite::micro::GetEvalInput(context, node, kChannelWidthsTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  params->config.weights = tflite::micro::GetTensorData<int16_t>(input1);
  params->config.unweights = tflite::micro::GetTensorData<int16_t>(input2);
  params->config.channel_frequency_starts =
      tflite::micro::GetTensorData<int16_t>(input3);
  params->config.channel_weight_starts =
      tflite::micro::GetTensorData<int16_t>(input4);
  params->config.channel_widths = tflite::micro::GetTensorData<int16_t>(input5);

  const uint32_t* input_data = tflite::micro::GetTensorData<uint32_t>(input0);
  uint64_t* output_data = tflite::micro::GetTensorData<uint64_t>(output);
  tflm_signal::FilterbankAccumulateChannels(&params->config, input_data,
                                            params->work_area);

  size_t output_size;
  TfLiteTypeSizeOf(output->type, &output_size);
  output_size *= ElementCount(*output->dims);
  // Discard channel 0, which is just scratch
  memcpy(output_data, params->work_area + 1, output_size);
  return kTfLiteOk;
}

}  // namespace

namespace tflm_signal {

TFLMRegistration* Register_FILTER_BANK() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      FilterBankInit, FilterBankPrepare, FilterBankEval);
  return &r;
}

}  // namespace tflm_signal

}  // namespace tflite
