/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "signal/src/window.h"

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
constexpr int kWeightsTensor = 1;
constexpr int kOutputTensor = 0;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
constexpr int kShiftIndex = 0;  // 'shift'

struct TFLMSignalWindowParams {
  int32_t shift;
  int32_t input_size;
};

void* WindowInit(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  auto* params =
      static_cast<TFLMSignalWindowParams*>(context->AllocatePersistentBuffer(
          context, sizeof(TFLMSignalWindowParams)));

  tflite::FlexbufferWrapper fbw(buffer_t, length);
  params->shift = fbw.ElementAsInt32(kShiftIndex);
  return params;
}

TfLiteStatus WindowPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* weights =
      micro_context->AllocateTempInputTensor(node, kWeightsTensor);
  TF_LITE_ENSURE(context, weights != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), NumDimensions(output));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, weights->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);

  auto* params = reinterpret_cast<TFLMSignalWindowParams*>(node->user_data);
  RuntimeShape input_shape = GetTensorShape(input);
  params->input_size = input_shape.FlatSize();

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(weights);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus WindowEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TFLMSignalWindowParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* weights =
      tflite::micro::GetEvalInput(context, node, kWeightsTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const int16_t* input_data = tflite::micro::GetTensorData<int16_t>(input);
  const int16_t* weight_data = tflite::micro::GetTensorData<int16_t>(weights);
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  int weight_size = weights->dims->data[0];

  for (int i = 0; i < params->input_size; i += weight_size) {
    ::tflm_signal::ApplyWindow(&input_data[i], weight_data, weight_size,
                               params->shift, &output_data[i]);
  }
  return kTfLiteOk;
}
}  // namespace

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

TFLMRegistration* Register_WINDOW() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(WindowInit, WindowPrepare, WindowEval);
  return &r;
}

}  // namespace tflm_signal
}  // namespace tflite
