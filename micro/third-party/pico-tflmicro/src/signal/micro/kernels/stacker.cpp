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

#include <stdint.h>

#include "signal/src/circular_buffer.h"
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
constexpr int kOutputValidTensor = 1;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
constexpr int kNumChannelsIndex = 0;          // 'num_channels'
constexpr int kStackerLeftContextIndex = 1;   // 'stacker_left_context'
constexpr int kStackerRightContextIndex = 2;  // 'stacker_right_context'
constexpr int kStackerStepIndex = 3;          // 'stacker_step'

struct TFLMSignalStackerParams {
  int32_t num_channels;
  int32_t stacker_left_context;
  int32_t stacker_right_context;
  int32_t stacker_step;

  size_t buffer_size;
  size_t step_size;
  bool stacker_has_first_frame;

  int8_t* state;
  tflm_signal::CircularBuffer* circular_buffer;
};

void* StackerInit(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  auto* params =
      static_cast<TFLMSignalStackerParams*>(context->AllocatePersistentBuffer(
          context, sizeof(TFLMSignalStackerParams)));
  if (params == nullptr) {
    return nullptr;
  }

  tflite::FlexbufferWrapper fbw(buffer_t, length);
  params->num_channels = fbw.ElementAsInt32(kNumChannelsIndex);
  params->stacker_left_context = fbw.ElementAsInt32(kStackerLeftContextIndex);
  params->stacker_right_context = fbw.ElementAsInt32(kStackerRightContextIndex);
  params->stacker_step = fbw.ElementAsInt32(kStackerStepIndex);

  params->buffer_size =
      params->num_channels *
      (params->stacker_left_context + params->stacker_right_context + 1);
  params->step_size = params->num_channels * params->stacker_step;
  params->stacker_has_first_frame = false;

  size_t state_size =
      tflm_signal::CircularBufferGetNeededMemory(params->buffer_size);
  params->state = static_cast<int8_t*>(
      context->AllocatePersistentBuffer(context, sizeof(int8_t) * state_size));

  if (params->state == nullptr) {
    return nullptr;
  }

  params->circular_buffer = tflm_signal::CircularBufferInit(
      params->buffer_size, params->state, state_size);
  return params;
}

TfLiteStatus StackerPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* output_valid =
      micro_context->AllocateTempOutputTensor(node, kOutputValidTensor);
  TF_LITE_ENSURE(context, output_valid != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_valid), 0);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output_valid->type, kTfLiteBool);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(output_valid);
  return kTfLiteOk;
}

TfLiteStatus StackerEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TFLMSignalStackerParams*>(node->user_data);
  TF_LITE_ENSURE(context, params != nullptr);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TfLiteEvalTensor* output_valid =
      tflite::micro::GetEvalOutput(context, node, kOutputValidTensor);

  const int16_t* input_data = tflite::micro::GetTensorData<int16_t>(input);

  tflm_signal::CircularBufferWrite(params->circular_buffer, input_data,
                                   params->num_channels);

  // The first frame is replicated an extra left_context times to pad.
  if (params->stacker_has_first_frame == false) {
    tflm_signal::CircularBufferExtend(params->circular_buffer,
                                      params->num_channels,
                                      params->stacker_left_context);
    params->stacker_has_first_frame = true;
  }

  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  bool* output_valid_data = tflite::micro::GetTensorData<bool>(output_valid);
  if (tflm_signal::CircularBufferAvailable(params->circular_buffer) >=
      params->buffer_size) {
    tflm_signal::CircularBufferGet(params->circular_buffer, params->buffer_size,
                                   output_data);
    tflm_signal::CircularBufferDiscard(params->circular_buffer,
                                       params->step_size);
    *output_valid_data = true;
  } else {
    *output_valid_data = false;
  }
  return kTfLiteOk;
}

void StackerReset(TfLiteContext* context, void* buffer) {
  auto* params = static_cast<TFLMSignalStackerParams*>(buffer);
  tflm_signal::CircularBufferReset(params->circular_buffer);
  params->stacker_has_first_frame = false;
}

}  // namespace

namespace tflm_signal {
TFLMRegistration* Register_STACKER() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      StackerInit, StackerPrepare, StackerEval, /*Free*/ nullptr, StackerReset);
  return &r;
}
}  // namespace tflm_signal

}  // namespace tflite
