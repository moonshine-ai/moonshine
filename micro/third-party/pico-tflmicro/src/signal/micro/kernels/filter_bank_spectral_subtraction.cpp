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

#include "signal/src/filter_bank_spectral_subtraction.h"

#include <stdint.h>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kNoiseEstimateTensor = 1;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
// 'alternate_one_minus_smoothing'
constexpr int kAlternateOneMinusSmoothingIndex = 0;
constexpr int kAlternateSmoothingIndex = 1;       // 'alternate_smoothing'
constexpr int kClampingIndex = 2;                 // 'clamping'
constexpr int kMinSignalRemainingIndex = 3;       // 'min_signal_remaining'
constexpr int kNumChannelsIndex = 4;              // 'num_channels'
constexpr int kOneMinusSmoothingIndex = 5;        // 'one_minus_smoothing'
constexpr int kSmoothingIndex = 6;                // 'smoothing'
constexpr int kSmoothingBitsIndex = 7;            // 'smoothing_bits'
constexpr int kSpectralSubtractionBitsIndex = 8;  // 'spectral_subtraction_bits'

struct TFLMSignalSpectralSubtractionParams {
  tflm_signal::SpectralSubtractionConfig config;
  uint32_t* noise_estimate;
  size_t noise_estimate_size;
};

void FilterBankSpectralSubtractionResetState(
    TFLMSignalSpectralSubtractionParams* params) {
  memset(params->noise_estimate, 0,
         sizeof(uint32_t) * params->config.num_channels);
}

void* FilterBankSpectralSubtractionInit(TfLiteContext* context,
                                        const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);

  auto* params = static_cast<TFLMSignalSpectralSubtractionParams*>(
      context->AllocatePersistentBuffer(
          context, sizeof(TFLMSignalSpectralSubtractionParams)));

  if (params == nullptr) {
    return nullptr;
  }

  tflite::FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer),
                                length);
  params->config.alternate_one_minus_smoothing =
      fbw.ElementAsInt32(kAlternateOneMinusSmoothingIndex);
  params->config.alternate_smoothing =
      fbw.ElementAsInt32(kAlternateSmoothingIndex);
  params->config.clamping = fbw.ElementAsBool(kClampingIndex);
  params->config.min_signal_remaining =
      fbw.ElementAsInt32(kMinSignalRemainingIndex);
  params->config.num_channels = fbw.ElementAsInt32(kNumChannelsIndex);
  params->config.one_minus_smoothing =
      fbw.ElementAsInt32(kOneMinusSmoothingIndex);
  params->config.one_minus_smoothing =
      fbw.ElementAsInt32(kOneMinusSmoothingIndex);
  params->config.smoothing = fbw.ElementAsInt32(kSmoothingIndex);
  params->config.smoothing_bits = fbw.ElementAsInt32(kSmoothingBitsIndex);
  params->config.spectral_subtraction_bits =
      fbw.ElementAsInt32(kSpectralSubtractionBitsIndex);
  params->noise_estimate =
      static_cast<uint32_t*>(context->AllocatePersistentBuffer(
          context, params->config.num_channels * sizeof(uint32_t)));

  if (params->noise_estimate == nullptr) {
    return nullptr;
  }

  return params;
}

TfLiteStatus FilterBankSpectralSubtractionPrepare(TfLiteContext* context,
                                                  TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TfLiteTensor* noise_estimate =
      micro_context->AllocateTempOutputTensor(node, kNoiseEstimateTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE(context, noise_estimate != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(noise_estimate), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteUInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, noise_estimate->type, kTfLiteUInt32);

  auto* params =
      reinterpret_cast<TFLMSignalSpectralSubtractionParams*>(node->user_data);
  TfLiteTypeSizeOf(output->type, &params->noise_estimate_size);
  params->noise_estimate_size *= ElementCount(*noise_estimate->dims);

  FilterBankSpectralSubtractionResetState(params);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(noise_estimate);
  return kTfLiteOk;
}

TfLiteStatus FilterBankSpectralSubtractionEval(TfLiteContext* context,
                                               TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TFLMSignalSpectralSubtractionParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TfLiteEvalTensor* noise_estimate =
      tflite::micro::GetEvalOutput(context, node, kNoiseEstimateTensor);

  const uint32_t* input_data = tflite::micro::GetTensorData<uint32_t>(input);
  uint32_t* output_data = tflite::micro::GetTensorData<uint32_t>(output);
  uint32_t* noise_estimate_data =
      tflite::micro::GetTensorData<uint32_t>(noise_estimate);

  FilterbankSpectralSubtraction(&params->config, input_data, output_data,
                                params->noise_estimate);

  memcpy(noise_estimate_data, params->noise_estimate,
         params->noise_estimate_size);

  return kTfLiteOk;
}

void FilterBankSpectralSubtractionReset(TfLiteContext* context, void* buffer) {
  FilterBankSpectralSubtractionResetState(
      static_cast<TFLMSignalSpectralSubtractionParams*>(buffer));
}

}  // namespace

namespace tflm_signal {

TFLMRegistration* Register_FILTER_BANK_SPECTRAL_SUBTRACTION() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      FilterBankSpectralSubtractionInit, FilterBankSpectralSubtractionPrepare,
      FilterBankSpectralSubtractionEval,
      /*Free*/ nullptr, FilterBankSpectralSubtractionReset);
  return &r;
}

}  // namespace tflm_signal

}  // namespace tflite
