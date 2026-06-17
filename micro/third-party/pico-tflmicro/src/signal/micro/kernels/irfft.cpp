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

#include "signal/src/irfft.h"

#include <math.h>
#include <stddef.h>
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
constexpr int kFftLengthIndex = 1;  // 'fft_length'

struct TfLiteAudioFrontendIrfftParams {
  int32_t fft_length;
  int32_t input_size;
  int32_t input_length;
  int32_t output_length;
  TfLiteType fft_type;
  int8_t* state;
};

template <typename T, size_t (*get_needed_memory_func)(int32_t),
          void* (*init_func)(int32_t, void*, size_t)>
void* IrfftInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);

  auto* params = static_cast<TfLiteAudioFrontendIrfftParams*>(
      context->AllocatePersistentBuffer(
          context, sizeof(TfLiteAudioFrontendIrfftParams)));

  if (params == nullptr) {
    return nullptr;
  }

  tflite::FlexbufferWrapper fbw(reinterpret_cast<const uint8_t*>(buffer),
                                length);
  params->fft_length = fbw.ElementAsInt32(kFftLengthIndex);
  params->fft_type = typeToTfLiteType<T>();

  size_t state_size = (*get_needed_memory_func)(params->fft_length);
  params->state = reinterpret_cast<int8_t*>(
      context->AllocatePersistentBuffer(context, state_size * sizeof(int8_t)));

  if (params->state == nullptr) {
    return nullptr;
  }

  (*init_func)(params->fft_length, params->state, state_size);
  return params;
}

template <TfLiteType TfLiteTypeEnum>
TfLiteStatus IrfftPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_EQ(context, NumDimensions(input), NumDimensions(output));

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, TfLiteTypeEnum);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, TfLiteTypeEnum);

  auto* params =
      reinterpret_cast<TfLiteAudioFrontendIrfftParams*>(node->user_data);
  RuntimeShape input_shape = GetTensorShape(input);
  RuntimeShape output_shape = GetTensorShape(output);
  // Divide by 2 because input is complex.
  params->input_length =
      input_shape.Dims(input_shape.DimensionsCount() - 1) / 2;
  params->input_size = input_shape.FlatSize() / 2;
  params->output_length = output_shape.Dims(output_shape.DimensionsCount() - 1);

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

template <typename T, void (*apply_func)(void*, const Complex<T>* input, T*)>
TfLiteStatus IrfftEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioFrontendIrfftParams*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const Complex<T>* input_data =
      tflite::micro::GetTensorData<Complex<T>>(input);
  T* output_data = tflite::micro::GetTensorData<T>(output);
  for (int input_idx = 0, output_idx = 0; input_idx < params->input_size;
       input_idx += params->input_length, output_idx += params->output_length) {
    (*apply_func)(params->state, &input_data[input_idx],
                  &output_data[output_idx]);
  }
  return kTfLiteOk;
}

void* IrfftInitAll(TfLiteContext* context, const char* buffer, size_t length) {
  const uint8_t* buffer_t = reinterpret_cast<const uint8_t*>(buffer);
  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_t, length).AsMap();
  auto tensor_type = static_cast<tflite::TensorType>(m["T"].AsInt32());

  switch (tensor_type) {
    case TensorType_INT16: {
      return IrfftInit<int16_t, tflm_signal::IrfftInt16GetNeededMemory,
                       tflm_signal::IrfftInt16Init>(context, buffer, length);
    }
    case TensorType_INT32: {
      return IrfftInit<int32_t, tflm_signal::IrfftInt32GetNeededMemory,
                       tflm_signal::IrfftInt32Init>(context, buffer, length);
    }
    case TensorType_FLOAT32: {
      return IrfftInit<float, tflm_signal::IrfftFloatGetNeededMemory,
                       tflm_signal::IrfftFloatInit>(context, buffer, length);
    }
    default:
      return nullptr;
  }
}

TfLiteStatus IrfftPrepareAll(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioFrontendIrfftParams*>(node->user_data);

  switch (params->fft_type) {
    case kTfLiteInt16: {
      return IrfftPrepare<kTfLiteInt16>(context, node);
    }
    case kTfLiteInt32: {
      return IrfftPrepare<kTfLiteInt32>(context, node);
    }
    case kTfLiteFloat32: {
      return IrfftPrepare<kTfLiteFloat32>(context, node);
    }
    default:
      return kTfLiteError;
  }
}

TfLiteStatus IrfftEvalAll(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteAudioFrontendIrfftParams*>(node->user_data);

  switch (params->fft_type) {
    case kTfLiteInt16: {
      return IrfftEval<int16_t, tflm_signal::IrfftInt16Apply>(context, node);
    }
    case kTfLiteInt32: {
      return IrfftEval<int32_t, tflm_signal::IrfftInt32Apply>(context, node);
    }
    case kTfLiteFloat32: {
      return IrfftEval<float, tflm_signal::IrfftFloatApply>(context, node);
    }
    default:
      return kTfLiteError;
  }
}

}  // namespace

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

TFLMRegistration* Register_IRFFT() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(IrfftInitAll, IrfftPrepareAll, IrfftEvalAll);
  return &r;
}

TFLMRegistration* Register_IRFFT_FLOAT() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      IrfftInit<float, IrfftFloatGetNeededMemory, IrfftFloatInit>,
      IrfftPrepare<kTfLiteFloat32>, IrfftEval<float, IrfftFloatApply>);
  return &r;
}

TFLMRegistration* Register_IRFFT_INT16() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      IrfftInit<int16_t, IrfftInt16GetNeededMemory, IrfftInt16Init>,
      IrfftPrepare<kTfLiteInt16>, IrfftEval<int16_t, IrfftInt16Apply>);
  return &r;
}

TFLMRegistration* Register_IRFFT_INT32() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      IrfftInit<int32_t, IrfftInt32GetNeededMemory, IrfftInt32Init>,
      IrfftPrepare<kTfLiteInt32>, IrfftEval<int32_t, IrfftInt32Apply>);
  return &r;
}

}  // namespace tflm_signal
}  // namespace tflite