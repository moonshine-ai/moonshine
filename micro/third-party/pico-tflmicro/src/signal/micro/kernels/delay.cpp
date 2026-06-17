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

#include <stdint.h>

#include "signal/src/circular_buffer.h"
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
constexpr int kDelayLengthIndex = 0;  // 'delay_length'

struct TFLMSignalFrontendDelayParams {
  int32_t frame_size;
  int32_t delay_length;
  int32_t outer_dims;

  int8_t** state_buffers;
  tflm_signal::CircularBuffer** circular_buffers;
};

void* DelayInit(TfLiteContext* context, const char* buffer, size_t length) {
  auto* params = static_cast<TFLMSignalFrontendDelayParams*>(
      context->AllocatePersistentBuffer(context,
                                        sizeof(TFLMSignalFrontendDelayParams)));

  if (params == nullptr) {
    return nullptr;
  }

  FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer), length);
  params->delay_length = fbw.ElementAsInt32(kDelayLengthIndex);
  return params;
}

TfLiteStatus DelayPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);

  auto* params =
      reinterpret_cast<TFLMSignalFrontendDelayParams*>(node->user_data);

  TF_LITE_ENSURE(context, params != nullptr);

  RuntimeShape input_shape = GetTensorShape(input);
  int innermost_dim = input_shape.Dims(input_shape.DimensionsCount() - 1);
  params->outer_dims = input_shape.FlatSize() / innermost_dim;
  params->frame_size = innermost_dim;

  params->state_buffers =
      static_cast<int8_t**>(context->AllocatePersistentBuffer(
          context, params->outer_dims * sizeof(int8_t*)));
  params->circular_buffers = static_cast<tflm_signal::CircularBuffer**>(
      context->AllocatePersistentBuffer(
          context, params->outer_dims * sizeof(tflm_signal::CircularBuffer*)));

  for (int i = 0; i < params->outer_dims; i++) {
    size_t capacity = params->frame_size + params->delay_length;

    size_t state_size = tflm_signal::CircularBufferGetNeededMemory(capacity);
    params->state_buffers[i] =
        static_cast<int8_t*>(context->AllocatePersistentBuffer(
            context, state_size * sizeof(int8_t)));
    params->circular_buffers[i] = tflm_signal::CircularBufferInit(
        capacity, params->state_buffers[i], state_size);
    tflm_signal::CircularBufferWriteZeros(params->circular_buffers[i],
                                          params->delay_length);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus DelayEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TFLMSignalFrontendDelayParams*>(node->user_data);
  const TfLiteEvalTensor* input =
      micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output = micro::GetEvalOutput(context, node, kOutputTensor);

  const int16_t* input_data = micro::GetTensorData<int16_t>(input);
  int16_t* output_data = micro::GetTensorData<int16_t>(output);

  for (int dim_index = 0, sample_index = 0; dim_index < params->outer_dims;
       dim_index++, sample_index += params->frame_size) {
    tflm_signal::CircularBufferWrite(params->circular_buffers[dim_index],
                                     &input_data[sample_index],
                                     params->frame_size);
    tflm_signal::CircularBufferGet(params->circular_buffers[dim_index],
                                   params->frame_size,
                                   &output_data[sample_index]);
    tflm_signal::CircularBufferDiscard(params->circular_buffers[dim_index],
                                       params->frame_size);
  }
  return kTfLiteOk;
}

void DelayReset(TfLiteContext* context, void* buffer) {
  auto* params = static_cast<TFLMSignalFrontendDelayParams*>(buffer);
  for (int i = 0; i < params->outer_dims; ++i) {
    tflm_signal::CircularBufferReset(params->circular_buffers[i]);
    tflm_signal::CircularBufferWriteZeros(params->circular_buffers[i],
                                          params->delay_length);
  }
}

}  // namespace

namespace tflm_signal {
TFLMRegistration* Register_DELAY() {
  static TFLMRegistration r = micro::RegisterOp(DelayInit, DelayPrepare,
                                                DelayEval, nullptr, DelayReset);
  return &r;
}
}  // namespace tflm_signal

}  // namespace tflite
