/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

// GELU micro kernel (local addition, not yet in upstream tflite-micro; see
// PATCHES.md). Quantized paths use the shared LUT helpers from
// kernels/internal/common.h -- a 256-entry table for int8 and a 513-entry
// interpolated table for int16 -- mirroring how full TFLite lowers GELU.

#include <cmath>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct GeluOpData {
  bool approximate;
  union {
    int8_t lut_int8[LUTSize<int8_t>()];
    int16_t lut_int16[LUTSize<int16_t>()];
  };
};

inline float GeluTransform(float in) {
  // 0.5 * x * ( 1 + erf( x / sqrt( 2 ) ) )
  return 0.5f * in * (1.f + std::erf(in * static_cast<float>(M_SQRT1_2)));
}

inline float GeluTransformApproximate(float in) {
  // 0.5 * x * ( 1 + tanh( sqrt( 2 / pi ) * ( x + 0.044715 * x^3 ) ) )
  return 0.5f * in *
         (1.f + std::tanh(static_cast<float>(M_2_SQRTPI * M_SQRT1_2) *
                          (in + 0.044715f * in * in * in)));
}

void* GeluInit(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(GeluOpData));
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
  GeluOpData* data = static_cast<GeluOpData*>(node->user_data);
  const auto* params = static_cast<TfLiteGeluParams*>(node->builtin_data);
  data->approximate = params->approximate;

  MicroContext* micro_context = GetMicroContext(context);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  float (*transform)(float) =
      data->approximate ? GeluTransformApproximate : GeluTransform;
  if (input->type == kTfLiteInt8) {
    LUTPopulate<int8_t>(input->params.scale, input->params.zero_point,
                        output->params.scale, output->params.zero_point,
                        transform, data->lut_int8);
  } else if (input->type == kTfLiteInt16) {
    LUTPopulate<int16_t>(input->params.scale, input->params.zero_point,
                         output->params.scale, output->params.zero_point,
                         transform, data->lut_int16);
  }

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(output);
  return kTfLiteOk;
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TFLITE_DCHECK(node->user_data != nullptr);
  const GeluOpData& data = *(static_cast<const GeluOpData*>(node->user_data));

  const int flat_size =
      tflite::micro::GetTensorShape(input).FlatSize();

  switch (input->type) {
    case kTfLiteFloat32: {
      const float* in = tflite::micro::GetTensorData<float>(input);
      float* out = tflite::micro::GetTensorData<float>(output);
      float (*transform)(float) =
          data.approximate ? GeluTransformApproximate : GeluTransform;
      for (int i = 0; i < flat_size; ++i) {
        out[i] = transform(in[i]);
      }
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      const int8_t* in = tflite::micro::GetTensorData<int8_t>(input);
      int8_t* out = tflite::micro::GetTensorData<int8_t>(output);
      for (int i = 0; i < flat_size; ++i) {
        out[i] = LUTLookup(in[i], data.lut_int8);
      }
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      const int16_t* in = tflite::micro::GetTensorData<int16_t>(input);
      int16_t* out = tflite::micro::GetTensorData<int16_t>(output);
      for (int i = 0; i < flat_size; ++i) {
        out[i] = LUTLookup(in[i], data.lut_int16);
      }
      return kTfLiteOk;
    }
    default:
      MicroPrintf("GELU only supports float32, int8, int16, got %s.",
                  TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace

TFLMRegistration Register_GELU() {
  return tflite::micro::RegisterOp(GeluInit, GeluPrepare, GeluEval);
}

}  // namespace tflite
