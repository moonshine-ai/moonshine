/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/src/energy.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
constexpr int kEndIndexIndex = 0;    // 'end_index'
constexpr int kStartIndexIndex = 1;  // 'start_index'

struct TFLMSignalEnergyParams {
  int32_t end_index;
  int32_t start_index;
};

void* EnergyInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);

  auto* data =
      static_cast<TFLMSignalEnergyParams*>(context->AllocatePersistentBuffer(
          context, sizeof(TFLMSignalEnergyParams)));

  if (data == nullptr) {
    return nullptr;
  }

  tflite::FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer),
                                length);
  data->end_index = fbw.ElementAsInt32(kEndIndexIndex);
  data->start_index = fbw.ElementAsInt32(kStartIndexIndex);
  return data;
}

TfLiteStatus EnergyPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteUInt32);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus EnergyEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TFLMSignalEnergyParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const Complex<int16_t>* input_data =
      tflite::micro::GetTensorData<Complex<int16_t>>(input);
  uint32_t* output_data = tflite::micro::GetTensorData<uint32_t>(output);

  tflm_signal::SpectrumToEnergy(input_data, params->start_index,
                                params->end_index, output_data);
  return kTfLiteOk;
}

}  // namespace

namespace tflm_signal {
TFLMRegistration* Register_ENERGY() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(EnergyInit, EnergyPrepare, EnergyEval);
  return &r;
}
}  // namespace tflm_signal

}  // namespace tflite
