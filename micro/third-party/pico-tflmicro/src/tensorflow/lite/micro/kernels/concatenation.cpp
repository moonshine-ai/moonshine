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
#include "tensorflow/lite/kernels/internal/reference/concatenation.h"

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

namespace {

constexpr int kMaxInputNum = 10;  // Maximum number of input tensors
constexpr int kOutputTensor = 0;

struct OpData {
  ConcatenationParams params;

#ifdef USE_TFLM_COMPRESSION

  // scratch buffers for compressed tensors
  int scratch_indices[kMaxInputNum];

#endif  // USE_TFLM_COMPRESSION
};

// Handles negative axis index, coerces to positive index value.
inline int CalculatePositiveAxis(int axis, const TfLiteTensor* output_tensor) {
  if (axis >= 0) {
    return axis;
  } else {
    return NumDimensions(output_tensor) + axis;
  }
}

// The following functions are helpers to get tensor data in the format that the
// reference op implementation expects. They provide the same functionality as
// class VectorOfTensors and class VectorOfQuantizedTensors in TFLite.

// Gets shapes from a list of tensors.
inline void GetAllInputTensorShapes(const TfLiteContext* context,
                                    const TfLiteNode* node,
                                    RuntimeShape all_shapes[kMaxInputNum]) {
  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
    RuntimeShape shape = tflite::micro::GetTensorShape(t);
    all_shapes[i].ReplaceWith(shape.DimensionsCount(), shape.DimsData());
  }
}

// Get shape pointers from a list of shapes.
inline void GetShapesPointers(const RuntimeShape* shapes, size_t num,
                              const RuntimeShape* pointers[]) {
  for (size_t i = 0; i < num; ++i) {
    pointers[i] = &shapes[i];
  }
}

// Gets data pointers from a list of tensors.
template <typename T>
inline void GetAllInputTensorData(const TfLiteContext* context,
                                  const TfLiteNode* node,
                                  const T* all_data[kMaxInputNum]) {
#ifdef USE_TFLM_COMPRESSION
  const OpData* data = static_cast<const OpData*>(node->user_data);
  MicroContext* micro_context = GetMicroContext(context);
#endif  // USE_TFLM_COMPRESSION

  for (int i = 0; i < node->inputs->size; ++i) {
    const TfLiteEvalTensor* t = tflite::micro::GetEvalInput(context, node, i);
#ifdef USE_TFLM_COMPRESSION
    const CompressionTensorData* comp_td =
        micro_context->GetTensorCompressionData(node, i);
    all_data[i] = tflite::micro::GetTensorData<T>(micro_context, t, comp_td,
                                                  data->scratch_indices[i]);
#else   // USE_TFLM_COMPRESSION
    all_data[i] = tflite::micro::GetTensorData<T>(t);
#endif  // USE_TFLM_COMPRESSION
  }
}

template <typename data_type>
void EvalUnquantized(TfLiteContext* context, TfLiteNode* node) {
  // Collect the shapes and data pointer of input tensors
  RuntimeShape inputs_shape[kMaxInputNum];
  const RuntimeShape* inputs_shape_ptr[kMaxInputNum];
  const data_type* inputs_data[kMaxInputNum];
  TFLITE_DCHECK(context != nullptr);
  TFLITE_DCHECK(node != nullptr);
  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);
  GetAllInputTensorShapes(context, node, inputs_shape);
  GetShapesPointers(inputs_shape, node->inputs->size, inputs_shape_ptr);
  GetAllInputTensorData(context, node, inputs_data);

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  reference_ops::Concatenation(data->params, inputs_shape_ptr, inputs_data,
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<data_type>(output));
}

void* ConcatenationInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus ConcatenationPrepare(TfLiteContext* context, TfLiteNode* node) {
  // This function only checks the types. Additional shape validations are
  // performed in the reference implementation called during Eval().
  const TfLiteConcatenationParams* params =
      reinterpret_cast<TfLiteConcatenationParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* input_tensor = micro_context->AllocateTempInputTensor(node, 0);
  TF_LITE_ENSURE(context, input_tensor != nullptr);
  TfLiteType input_type = input_tensor->type;
  TfLiteTensor* output_tensor =
      micro_context->AllocateTempOutputTensor(node, kOutputTensor);
  TF_LITE_ENSURE(context, output_tensor != nullptr);
  TfLiteType output_type = output_tensor->type;

  micro_context->DeallocateTempTfLiteTensor(input_tensor);

  // Check activation and input type
  TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActNone);
  TF_LITE_ENSURE(context,
                 input_type == kTfLiteFloat32 || input_type == kTfLiteInt8 ||
                     input_type == kTfLiteInt16 || input_type == kTfLiteInt32 ||
                     input_type == kTfLiteInt64 || input_type == kTfLiteBool);

  // Output type must match input type
  TF_LITE_ENSURE_TYPES_EQ(context, output_type, input_type);

  // This implementation does not support large number of input tensors
  const int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs <= kMaxInputNum);

  // Calculate OpData.
  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  // Shapes with dimensions > kMaxSmallSize are not yet supported with static
  // allocation.
  for (int i = 0; i < num_inputs; ++i) {
    TfLiteTensor* input = micro_context->AllocateTempInputTensor(node, i);
    TF_LITE_ENSURE(context, input != nullptr);
    TF_LITE_ENSURE_TYPES_EQ(context, input->type, input_type);
    int num_dimensions = NumDimensions(input);

    if (num_dimensions > RuntimeShape::kMaxSmallSize) {
      MicroPrintf(
          "Op Concatenation does not currently support num dimensions > %d "
          "Tensor has %d dimensions.",
          RuntimeShape::kMaxSmallSize, num_dimensions);
      return kTfLiteError;
    }

    if (input_type == kTfLiteInt8) {
      // Make sure there is no re-scaling needed for Int8 quantized kernel. This
      // is a restriction we introduced to Int8 kernels.
      TF_LITE_ENSURE_EQ(context, static_cast<double>(input->params.scale),
                        static_cast<double>(output_tensor->params.scale));
      TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                        output_tensor->params.zero_point);
    } else if (input_type == kTfLiteInt16) {
      // Make sure that all Int16 inputs have a null zero-point.
      TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    }

#ifdef USE_TFLM_COMPRESSION

    // Compression scratch buffers.
    // These will only be allocated if the tensor is compressed.
    data->scratch_indices[i] =
        micro_context->AllocateDecompressionScratchBuffer(node, i);

#endif  // USE_TFLM_COMPRESSION

    micro_context->DeallocateTempTfLiteTensor(input);
  }

  if (input_type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, output_tensor->params.zero_point, 0);
  }

  switch (output_type) {  // Already know in/outtypes are same.
    case kTfLiteBool:
    case kTfLiteFloat32:
    case kTfLiteInt8:
    case kTfLiteInt16:
    case kTfLiteInt32:
    case kTfLiteInt64: {
      data->params.axis = CalculatePositiveAxis(params->axis, output_tensor);
      data->params.inputs_count = node->inputs->size;
      break;
    }
    default:
      MicroPrintf("Op Concatenation does not currently support type '%s'.",
                  TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  micro_context->DeallocateTempTfLiteTensor(output_tensor);

  return kTfLiteOk;
}

TfLiteStatus ConcatenationEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* output_tensor =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output_tensor != nullptr);
  TfLiteType output_type = output_tensor->type;

  switch (output_type) {  // Already know in/outtypes are same.
    case kTfLiteFloat32:
      EvalUnquantized<float>(context, node);
      break;
    case kTfLiteInt32:
      EvalUnquantized<int32_t>(context, node);
      break;
    case kTfLiteInt8:
      EvalUnquantized<int8_t>(context, node);
      break;
    case kTfLiteInt64:
      EvalUnquantized<int64_t>(context, node);
      break;
    case kTfLiteInt16:
      EvalUnquantized<int16_t>(context, node);
      break;
    case kTfLiteBool:
      EvalUnquantized<bool>(context, node);
      break;

    default:
      MicroPrintf("Op Concatenation does not currently support Type '%s'.",
                  TfLiteTypeGetName(output_type));
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_CONCATENATION() {
  return tflite::micro::RegisterOp(ConcatenationInit, ConcatenationPrepare,
                                   ConcatenationEval);
}

}  // namespace tflite
