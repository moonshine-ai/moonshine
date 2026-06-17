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

#include <stddef.h>
#include <stdint.h>

#include "signal/src/pcan_argc_fixed.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {
namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above

constexpr int kInputTensor = 0;
constexpr int kNoiseEstimateTensor = 1;
constexpr int kGainLutTensor = 2;
constexpr int kOutputTensor = 0;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
constexpr int kSnrShiftIndex = 0;  // 'snr_shift'

struct TfLitePcanParams {
  int snr_shift;
};

void* PcanInit(TfLiteContext* context, const char* buffer, size_t length) {
  auto* params = static_cast<TfLitePcanParams*>(
      context->AllocatePersistentBuffer(context, sizeof(TfLitePcanParams)));

  tflite::FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer),
                                length);
  params->snr_shift = fbw.ElementAsInt32(kSnrShiftIndex);
  return params;
}

TfLiteStatus PcanPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* noise_estimate =
      micro_context->AllocateTempInputTensor(node, kNoiseEstimateTensor);
  TF_LITE_ENSURE(context, noise_estimate != nullptr);
  TfLiteTensor* gain_lut =
      micro_context->AllocateTempInputTensor(node, kGainLutTensor);
  TF_LITE_ENSURE(context, gain_lut != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(noise_estimate), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(gain_lut), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, noise_estimate->type, kTfLiteUInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, gain_lut->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt32);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(noise_estimate);
  micro_context->DeallocateTempTfLiteTensor(gain_lut);
  return kTfLiteOk;
}

TfLiteStatus PcanEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePcanParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteEvalTensor* noise_estimate =
      tflite::micro::GetEvalInput(context, node, kNoiseEstimateTensor);
  TF_LITE_ENSURE(context, noise_estimate != nullptr);
  const TfLiteEvalTensor* gain_lut =
      tflite::micro::GetEvalInput(context, node, kGainLutTensor);
  TF_LITE_ENSURE(context, gain_lut != nullptr);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  const uint32_t* input_data = tflite::micro::GetTensorData<uint32_t>(input);
  const uint32_t* noise_estimate_data =
      tflite::micro::GetTensorData<uint32_t>(noise_estimate);
  const int16_t* gain_lut_data =
      tflite::micro::GetTensorData<int16_t>(gain_lut);
  uint32_t* output_data = tflite::micro::GetTensorData<uint32_t>(output);

  int num_channels = input->dims->data[0];

  size_t output_byte_size;
  TF_LITE_ENSURE_OK(
      context, tflite::TfLiteEvalTensorByteLength(output, &output_byte_size));

  memcpy(output_data, input_data, output_byte_size);

  tflite::tflm_signal::ApplyPcanAutoGainControlFixed(
      gain_lut_data, params->snr_shift, noise_estimate_data, output_data,
      num_channels);
  return kTfLiteOk;
}

TFLMRegistration* Register_PCAN() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(PcanInit, PcanPrepare, PcanEval);
  return &r;
}

}  // namespace tflm_signal
}  // namespace tflite
