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

#include "signal/src/overlap_add.h"

#include <stdint.h>

#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

// Indices into the init flexbuffer's vector.
// The parameter's name is in the comment that follows.
// Elements in the vectors are ordered alphabetically by parameter name.
// 'T' is added implicitly by the TensorFlow framework when the type is resolved
// during graph construction.
// constexpr int kTypeIndex = 0;  // 'T' (unused)
constexpr int kFrameStepIndex = 1;  // 'frame_step'

template <typename T>
struct TFLMSignalOverlapAddParams {
  int32_t frame_size;
  int32_t frame_step;
  int32_t outer_dims;
  int32_t n_frames;
  TfLiteType type;
  T** state_buffers;
};

template <typename T>
void OverlapAddResetState(TFLMSignalOverlapAddParams<T>* params) {
  for (int i = 0; i < params->outer_dims; i++) {
    memset(params->state_buffers[i], 0, sizeof(T) * params->frame_size);
  }
}

template <typename T>
void* OverlapAddInit(TfLiteContext* context, const char* buffer,
                     size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);

  auto* params = static_cast<TFLMSignalOverlapAddParams<T>*>(
      context->AllocatePersistentBuffer(context,
                                        sizeof(TFLMSignalOverlapAddParams<T>)));

  if (params == nullptr) {
    return nullptr;
  }

  tflite::FlexbufferWrapper fbw(buffer_t, length);
  params->type = typeToTfLiteType<T>();
  params->frame_step = fbw.ElementAsInt32(kFrameStepIndex);
  return params;
}

template <typename T, TfLiteType TfLiteTypeEnum>
TfLiteStatus OverlapAddPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), NumDimensions(output) + 1);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, TfLiteTypeEnum);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, TfLiteTypeEnum);

  auto* params =
      reinterpret_cast<TFLMSignalOverlapAddParams<T>*>(node->user_data);
  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);
  TF_LITE_ENSURE(context, input_shape.DimensionsCount() >= 2);
  TF_LITE_ENSURE_EQ(context, input_shape.DimensionsCount(),
                    output_shape.DimensionsCount() + 1);

  params->frame_size = input_shape.Dims(input_shape.DimensionsCount() - 1);
  params->n_frames = input_shape.Dims(input_shape.DimensionsCount() - 2);
  params->outer_dims =
      input_shape.FlatSize() / (params->frame_size * params->n_frames);
  params->state_buffers = static_cast<T**>(context->AllocatePersistentBuffer(
      context, params->outer_dims * sizeof(T*)));
  TF_LITE_ENSURE(context, params != nullptr);

  for (int i = 0; i < params->outer_dims; i++) {
    params->state_buffers[i] =
        static_cast<T*>(context->AllocatePersistentBuffer(
            context, params->frame_size * sizeof(T)));
  }
  OverlapAddResetState(params);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus OverlapAddEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TFLMSignalOverlapAddParams<T>*>(node->user_data);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const T* input_data = tflite::micro::GetTensorData<T>(input);
  T* output_data = tflite::micro::GetTensorData<T>(output);
  for (int i = 0; i < params->outer_dims; i++) {
    T* buffer = params->state_buffers[i];
    for (int frame = 0; frame < params->n_frames; frame++) {
      int input_index = (i * params->n_frames + frame) * params->frame_size;
      int output_index = (i * params->n_frames + frame) * params->frame_step;
      tflm_signal::OverlapAdd(&input_data[input_index], buffer,
                              params->frame_size, &output_data[output_index],
                              params->frame_step);
    }
  }
  return kTfLiteOk;
}

template <typename T>
void OverlapAddReset(TfLiteContext* context, void* buffer) {
  OverlapAddResetState(static_cast<TFLMSignalOverlapAddParams<T>*>(buffer));
}

void* OverlapAddInitAll(TfLiteContext* context, const char* buffer,
                        size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  auto tensor_type = static_cast<tflite::TensorType>(m["T"].AsInt32());

  switch (tensor_type) {
    case TensorType_INT16: {
      return OverlapAddInit<int16_t>(context, buffer, length);
    }
    case TensorType_FLOAT32: {
      return OverlapAddInit<float>(context, buffer, length);
    }
    default:
      return nullptr;
  }
}

TfLiteStatus OverlapAddPrepareAll(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TFLMSignalOverlapAddParams<void>*>(node->user_data);

  switch (params->type) {
    case kTfLiteInt16: {
      return OverlapAddPrepare<int16_t, kTfLiteInt16>(context, node);
    }
    case kTfLiteFloat32: {
      return OverlapAddPrepare<float, kTfLiteFloat32>(context, node);
    }
    default:
      return kTfLiteError;
  }
}

TfLiteStatus OverlapAddEvalAll(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TFLMSignalOverlapAddParams<void>*>(node->user_data);

  switch (params->type) {
    case kTfLiteInt16: {
      return OverlapAddEval<int16_t>(context, node);
    }
    case kTfLiteFloat32: {
      return OverlapAddEval<float>(context, node);
    }
    default:
      return kTfLiteError;
  }
}

void OverlapAddResetAll(TfLiteContext* context, void* buffer) {
  auto* params = reinterpret_cast<TFLMSignalOverlapAddParams<void>*>(buffer);

  switch (params->type) {
    case kTfLiteInt16: {
      OverlapAddReset<int16_t>(context, buffer);
      break;
    }
    case kTfLiteFloat32: {
      OverlapAddReset<float>(context, buffer);
      break;
    }
    default:
      break;
  }
}

}  // namespace

namespace tflm_signal {
TFLMRegistration* Register_OVERLAP_ADD() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(OverlapAddInitAll, OverlapAddPrepareAll,
                                OverlapAddEvalAll, nullptr, OverlapAddResetAll);
  return &r;
}

TFLMRegistration* Register_OVERLAP_ADD_FLOAT() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      OverlapAddInit<float>, OverlapAddPrepare<float, kTfLiteFloat32>,
      OverlapAddEval<float>, nullptr, OverlapAddReset<float>);
  return &r;
}

TFLMRegistration* Register_OVERLAP_ADD_INT16() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      OverlapAddInit<int16_t>, OverlapAddPrepare<int16_t, kTfLiteInt16>,
      OverlapAddEval<int16_t>, nullptr, OverlapAddReset<int16_t>);
  return &r;
}
}  // namespace tflm_signal

}  // namespace tflite
