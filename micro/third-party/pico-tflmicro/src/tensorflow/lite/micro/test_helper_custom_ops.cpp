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

#include "tensorflow/lite/micro/test_helper_custom_ops.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>

#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"

// TODO(b/170464050): Use TFLM test only version of schema_utils.

namespace tflite {
namespace testing {

namespace {

template <typename T>
void BroadcastAdd(const T input_scalar, const T* weights, T* output,
                  const size_t count) {
  for (size_t i = 0; i < count; i++) {
    output[i] = input_scalar + weights[i];
  }
}

}  // namespace

const TFLMRegistration* PackerOp::getRegistration() {
  return GetMutableRegistration();
}

TFLMRegistration* PackerOp::GetMutableRegistration() {
  static TFLMRegistration r;
  r.init = Init;
  r.prepare = Prepare;
  r.invoke = Invoke;
  r.free = Free;
  return &r;
}

void* PackerOp::Init(TfLiteContext* context, const char* buffer,
                     size_t length) {
  freed_ = false;
  // Do nothing.
  return nullptr;
}

void PackerOp::Free(TfLiteContext* context, void* buffer) { freed_ = true; }

TfLiteStatus PackerOp::Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

TfLiteStatus PackerOp::Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input1 =
      tflite::micro::GetEvalInput(context, node, 0);
  TF_LITE_ENSURE(context, input1 != nullptr);
  const int32_t* input1_data = input1->data.i32;
  TF_LITE_ENSURE_EQ(context, input1->dims->size, 1);
  const int32_t input1_len = input1->dims->data[0];

  const TfLiteEvalTensor* input2 =
      tflite::micro::GetEvalInput(context, node, 1);
  TF_LITE_ENSURE(context, input2 != nullptr);
  const int32_t* input2_data = input2->data.i32;
  TF_LITE_ENSURE_EQ(context, input2->dims->size, 1);
  const int32_t input2_len = input2->dims->data[0];

  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);
  int32_t* output_data = output->data.i32;
  int32_t output_len = output->dims->data[0];

  // Fill output with input: first with the first tensor, then with the second
  // tensor up to the size of the output tensor.
  int cnt = 0;
  int i;
  for (i = 0; i < input1_len && cnt < output_len; i++, cnt++) {
    output_data[cnt] = input1_data[i];
  }
  if (cnt >= output_len) {
    return kTfLiteOk;
  }

  for (i = 0; i < input2_len && cnt < output_len; i++, cnt++) {
    output_data[cnt] = input2_data[i];
  }
  if (cnt >= output_len) {
    return kTfLiteOk;
  }

  for (; cnt < output_len; cnt++) {
    output_data[cnt] = 0;
  }
  return kTfLiteOk;
}

bool PackerOp::freed_ = false;

const TFLMRegistration* BroadcastAddOp::getRegistration() {
  return GetMutableRegistration();
}

TFLMRegistration* BroadcastAddOp::GetMutableRegistration() {
  static TFLMRegistration r;
  r.init = Init;
  r.prepare = Prepare;
  r.invoke = Invoke;
  return &r;
}

void* BroadcastAddOp::Init(TfLiteContext* context, const char* buffer,
                           size_t length) {
#ifdef USE_TFLM_COMPRESSION

  weight_scratch_index_ = -1;

#endif  // USE_TFLM_COMPRESSION

  // Do nothing.
  return nullptr;
}

TfLiteStatus BroadcastAddOp::Prepare(TfLiteContext* context, TfLiteNode* node) {
  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* weights = micro_context->AllocateTempInputTensor(node, 1);
  TF_LITE_ENSURE(context, weights != nullptr);
  TfLiteTensor* output = micro_context->AllocateTempOutputTensor(node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, weights->type);
  TF_LITE_ENSURE(
      context, input->type == kTfLiteFloat32 || input->type == kTfLiteInt8 ||
                   input->type == kTfLiteInt16 || input->type == kTfLiteInt32 ||
                   input->type == kTfLiteInt64);
  TF_LITE_ENSURE(context, input->quantization.type == kTfLiteNoQuantization);
  TF_LITE_ENSURE(context, weights->quantization.type == kTfLiteNoQuantization);
  TF_LITE_ENSURE(context, output->quantization.type == kTfLiteNoQuantization);
  TF_LITE_ENSURE(context,
                 ElementCount(*weights->dims) == ElementCount(*output->dims));
  TF_LITE_ENSURE(context, ElementCount(*input->dims) == 1);
  TF_LITE_ENSURE(context, input->dims->size == 1);
  TF_LITE_ENSURE(context, weights->dims->size == 1);

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  weight_scratch_index_ =
      micro_context->AllocateDecompressionScratchBuffer(node, 1);
  if (!micro_context->IsTensorCompressed(node, 1)) {
    TF_LITE_ENSURE(context, weight_scratch_index_ == -1);
  }

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(weights);
  micro_context->DeallocateTempTfLiteTensor(output);

  return kTfLiteOk;
}

TfLiteStatus BroadcastAddOp::Invoke(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteEvalTensor* weights =
      tflite::micro::GetEvalInput(context, node, 1);
  TF_LITE_ENSURE(context, weights != nullptr);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TF_LITE_ENSURE(context, output != nullptr);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* weights_comp_td =
      micro_context->GetTensorCompressionData(node, 1);
  if (micro_context->IsTensorCompressed(node, 1)) {
    TF_LITE_ENSURE(context, weights_comp_td != nullptr);
  } else {
    TF_LITE_ENSURE(context, weights_comp_td == nullptr);
  }

#endif  // USE_TFLM_COMPRESSION

  switch (input->type) {
    case kTfLiteFloat32: {
      BroadcastAdd(
          tflite::micro::GetTensorData<float>(input)[0],
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(
              micro_context, weights, weights_comp_td, weight_scratch_index_),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(weights),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(output),
          ElementCount(*output->dims));
    } break;

    case kTfLiteInt8: {
      BroadcastAdd(
          tflite::micro::GetTensorData<int8_t>(input)[0],
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int8_t>(
              micro_context, weights, weights_comp_td, weight_scratch_index_),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int8_t>(weights),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int8_t>(output),
          ElementCount(*output->dims));
    } break;

    case kTfLiteInt16: {
      BroadcastAdd(
          tflite::micro::GetTensorData<int16_t>(input)[0],
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int16_t>(
              micro_context, weights, weights_comp_td, weight_scratch_index_),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int16_t>(weights),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int16_t>(output),
          ElementCount(*output->dims));
    } break;

    case kTfLiteInt32: {
      BroadcastAdd(
          tflite::micro::GetTensorData<int32_t>(input)[0],
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int32_t>(
              micro_context, weights, weights_comp_td, weight_scratch_index_),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int32_t>(weights),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int32_t>(output),
          ElementCount(*output->dims));
    } break;

    case kTfLiteInt64: {
      BroadcastAdd(
          tflite::micro::GetTensorData<int64_t>(input)[0],
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int64_t>(
              micro_context, weights, weights_comp_td, weight_scratch_index_),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int64_t>(weights),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int64_t>(output),
          ElementCount(*output->dims));
    } break;

    default: {
      MicroPrintf("Input type %s (%d) not supported.",
                  TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

#ifdef USE_TFLM_COMPRESSION

int BroadcastAddOp::weight_scratch_index_ = -1;

#endif  // USE_TFLM_COMPRESSION

}  // namespace testing
}  // namespace tflite
