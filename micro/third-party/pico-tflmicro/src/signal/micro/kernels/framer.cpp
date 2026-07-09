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
constexpr int kFrameSizeIndex = 0;  // 'frame_size'
constexpr int kFrameStepIndex = 1;  // 'frame_step'
constexpr int kPrefillIndex = 2;    // 'prefill'

struct TFLMSignalFramerParams {
  int32_t frame_size;
  int32_t frame_step;
  int32_t outer_dims;
  int32_t n_frames;
  bool prefill;

  int8_t** state_buffers;
  tflite::tflm_signal::CircularBuffer** circular_buffers;
};

void FramerResetState(TFLMSignalFramerParams* params) {
  for (int i = 0; i < params->outer_dims; ++i) {
    tflite::tflm_signal::CircularBufferReset(params->circular_buffers[i]);
    if (params->prefill) {
      tflite::tflm_signal::CircularBufferWriteZeros(
          params->circular_buffers[i], params->frame_size - params->frame_step);
    }
  }
}

void* FramerInit(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  auto* params =
      static_cast<TFLMSignalFramerParams*>(context->AllocatePersistentBuffer(
          context, sizeof(TFLMSignalFramerParams)));

  if (params == nullptr) {
    return nullptr;
  }

  tflite::FlexbufferWrapper fbw(buffer_t, length);
  params->frame_size = fbw.ElementAsInt32(kFrameSizeIndex);
  params->frame_step = fbw.ElementAsInt32(kFrameStepIndex);
  params->prefill = fbw.ElementAsBool(kPrefillIndex);
  return params;
}

TfLiteStatus FramerPrepare(TfLiteContext* context, TfLiteNode* node) {
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

  TF_LITE_ENSURE_EQ(context, NumDimensions(input) + 1, NumDimensions(output));
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_valid), 0);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output_valid->type, kTfLiteBool);

  auto* params = reinterpret_cast<TFLMSignalFramerParams*>(node->user_data);

  RuntimeShape input_shape = GetTensorShape(input);
  int innermost_dim = input_shape.Dims(input_shape.DimensionsCount() - 1);
  TF_LITE_ENSURE(context, innermost_dim >= params->frame_step);
  TF_LITE_ENSURE_EQ(context, innermost_dim % params->frame_step, 0);
  params->outer_dims = input_shape.FlatSize() / innermost_dim;
  params->n_frames = innermost_dim / params->frame_step;

  params->state_buffers =
      static_cast<int8_t**>(context->AllocatePersistentBuffer(
          context, params->outer_dims * sizeof(int8_t*)));
  params->circular_buffers = static_cast<tflite::tflm_signal::CircularBuffer**>(
      context->AllocatePersistentBuffer(
          context,
          params->outer_dims * sizeof(tflite::tflm_signal::CircularBuffer*)));
  for (int i = 0; i < params->outer_dims; i++) {
    // Calculate the capacity of the circular buffer. Round up the frame size to
    // a multiple of frame step. Saves memory relative to the simpler frame_size
    // + frame_step. For example: step_size = 160, frame_size = 400 capacity =
    // 480 vs. step_size + frame_size = 560
    size_t capacity = (params->frame_size + params->frame_step - 1) /
                      params->frame_step * params->frame_step;

    size_t state_size =
        tflite::tflm_signal::CircularBufferGetNeededMemory(capacity);
    params->state_buffers[i] =
        static_cast<int8_t*>(context->AllocatePersistentBuffer(
            context, state_size * sizeof(int8_t)));
    params->circular_buffers[i] = tflite::tflm_signal::CircularBufferInit(
        capacity, params->state_buffers[i], state_size);
  }

  FramerResetState(params);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(output_valid);

  return kTfLiteOk;
}

TfLiteStatus FramerEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TFLMSignalFramerParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TfLiteEvalTensor* output_valid =
      tflite::micro::GetEvalOutput(context, node, kOutputValidTensor);

  const int16_t* input_data = tflite::micro::GetTensorData<int16_t>(input);
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  bool* output_valid_data = tflite::micro::GetTensorData<bool>(output_valid);
  *output_valid_data = true;

  for (int i = 0; i < params->outer_dims; i++) {
    for (int frame = 0; frame < params->n_frames; frame++) {
      int input_idx = (i * params->n_frames + frame) * params->frame_step;
      int output_idx = (i * params->n_frames + frame) * params->frame_size;
      tflite::tflm_signal::CircularBufferWrite(params->circular_buffers[i],
                                               &input_data[input_idx],
                                               params->frame_step);

      if (tflite::tflm_signal::CircularBufferAvailable(
              params->circular_buffers[i]) >=
          static_cast<size_t>(params->frame_size)) {
        tflite::tflm_signal::CircularBufferGet(params->circular_buffers[i],
                                               params->frame_size,
                                               &output_data[output_idx]);
        tflite::tflm_signal::CircularBufferDiscard(params->circular_buffers[i],
                                                   params->frame_step);
      } else {
        *output_valid_data = false;
      }
    }
  }

  return kTfLiteOk;
}

void FramerReset(TfLiteContext* context, void* buffer) {
  FramerResetState(static_cast<TFLMSignalFramerParams*>(buffer));
}

}  // namespace

namespace tflm_signal {
// TODO(b/286250473): remove namespace once de-duped libraries above
TFLMRegistration* Register_FRAMER() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      FramerInit, FramerPrepare, FramerEval, nullptr, FramerReset);
  return &r;
}
}  // namespace tflm_signal

}  // namespace tflite
