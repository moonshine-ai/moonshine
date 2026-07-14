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

#include "tensorflow/lite/kernels/internal/reference/transpose_conv.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/transpose_conv.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {
namespace {

struct OpData {
  ConvParams params;

  // A scratch buffer is required for quantized implementations.
  int scratch_buffer_index;

#ifdef USE_TFLM_COMPRESSION

  // scratch buffers for compressed tensors
  int filter_scratch_index;
  int bias_scratch_index;

#endif  // USE_TFLM_COMPRESSION

  // Index to the converted 64-bit bias buffer from 16-bit bias. This is
  // required to handle 16x8 transpose convolutions where a 16-bit bias is
  // provided, whereas the kernel expects 64-bit biases.
  int bias_converted_buffer_index;

  // Multiplier and shift arrays are required for the int8 implementation.
  int32_t* per_channel_output_multiplier;
  int32_t* per_channel_output_shift;
};

inline PaddingType RuntimePaddingType(TfLitePadding padding) {
  switch (padding) {
    case TfLitePadding::kTfLitePaddingSame:
      return PaddingType::kSame;
    case TfLitePadding::kTfLitePaddingValid:
      return PaddingType::kValid;
    case TfLitePadding::kTfLitePaddingUnknown:
    default:
      return PaddingType::kNone;
  }
}

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node,
                             const TfLiteTransposeConvParams* params, int width,
                             int height, int filter_width, int filter_height,
                             const TfLiteType data_type, OpData* data) {
  bool has_bias = node->inputs->size == 4;
  // Check number of inputs/outputs
  TF_LITE_ENSURE(context, has_bias || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  int unused_output_width;
  int unused_output_height;
  TfLitePaddingValues padding_values = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1,
      1,  // Dilation height and width are always 1 for transpose_conv.
      height, width, filter_height, filter_width, padding,
      &unused_output_height, &unused_output_width);

  data->params.padding_type = RuntimePaddingType(padding);
  data->params.padding_values.width = padding_values.width;
  data->params.padding_values.height = padding_values.height;

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (data_type != kTfLiteFloat32) {
    MicroContext* micro_context = GetMicroContext(context);

    TfLiteTensor* input =
        micro_context->AllocateTempInputTensor(node, kTransposeConvInputTensor);
    TF_LITE_ENSURE(context, input != nullptr);
    TfLiteTensor* filter = micro_context->AllocateTempInputTensor(
        node, kTransposeConvFilterTensor);
    TF_LITE_ENSURE(context, filter != nullptr);
    TfLiteTensor* bias =
        micro_context->AllocateTempInputTensor(node, kTransposeConvBiasTensor);
    TfLiteTensor* output = micro_context->AllocateTempOutputTensor(
        node, kTransposeConvOutputTensor);
    TF_LITE_ENSURE(context, output != nullptr);
    int output_channels = filter->dims->data[kTransposeConvQuantizedDimension];

    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation,
        &data->params.output_multiplier, &data->params.output_shift,
        &data->params.quantized_activation_min,
        &data->params.quantized_activation_max,
        data->per_channel_output_multiplier, data->per_channel_output_shift,
        output_channels));

    // TODO(b/192090531): Remove this once all 8x16 transpose conv models use
    // 64-bit biases.
    if (input->type == kTfLiteInt16) {
      TFLITE_DCHECK(filter->type == kTfLiteInt8);
      TFLITE_DCHECK(output->type == kTfLiteInt16);
      // Handle the case where the bias is 16 bits for 16x8 transpose
      // convolution where the kernel actually expects 64-bit biases.
      if (bias != nullptr && bias->type == kTfLiteInt16) {
        TFLITE_DCHECK(
            context->RequestScratchBufferInArena(
                context, GetTensorShape(bias).FlatSize() * sizeof(std::int64_t),
                &(data->bias_converted_buffer_index)) == kTfLiteOk);
      }
    }

    micro_context->DeallocateTempTfLiteTensor(input);
    micro_context->DeallocateTempTfLiteTensor(filter);
    micro_context->DeallocateTempTfLiteTensor(output);
    if (bias != nullptr) {
      micro_context->DeallocateTempTfLiteTensor(bias);
    }
  }
  return kTfLiteOk;
}

void* TransposeConvInit(TfLiteContext* context, const char* buffer,
                        size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus TransposeConvPrepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteTransposeConvParams*>(node->builtin_data);

  MicroContext* micro_context = GetMicroContext(context);

  TfLiteTensor* output =
      micro_context->AllocateTempOutputTensor(node, kTransposeConvOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  TfLiteTensor* input =
      micro_context->AllocateTempInputTensor(node, kTransposeConvInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* filter =
      micro_context->AllocateTempInputTensor(node, kTransposeConvFilterTensor);
  TF_LITE_ENSURE(context, filter != nullptr);

  TF_LITE_ENSURE_MSG(
      context,
      input->type == filter->type ||
          (input->type == kTfLiteInt16 && filter->type == kTfLiteInt8),
      "Hybrid models are not supported on TFLite Micro.");

  // Get height and width of the output.
  const int width = SizeOfDimension(output, 2);
  const int height = SizeOfDimension(output, 1);
  const int filter_width = SizeOfDimension(filter, 2);
  const int filter_height = SizeOfDimension(filter, 1);

  // Dynamically allocate per-channel quantization parameters.
  const int num_channels = filter->dims->data[kTransposeConvQuantizedDimension];
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  // Quantized kernels use an int32 scratch buffer.
  if (input->type == kTfLiteInt8) {
    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);
    TFLITE_DCHECK(context->RequestScratchBufferInArena(
                      context,
                      GetTensorShape(output).FlatSize() * sizeof(int32_t),
                      &(data->scratch_buffer_index)) == kTfLiteOk);
  }

  // Quantized 16x8 kernels use an int64 scratch buffer.
  if (input->type == kTfLiteInt16) {
    TFLITE_DCHECK(context->RequestScratchBufferInArena != nullptr);
    TFLITE_DCHECK(context->RequestScratchBufferInArena(
                      context,
                      GetTensorShape(output).FlatSize() * sizeof(std::int64_t),
                      &(data->scratch_buffer_index)) == kTfLiteOk);
  }

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        static_cast<TfLiteAffineQuantization*>(filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);

    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kTransposeConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height,
                                        input->type, data));

  // Offsets (zero points)
  data->params.input_offset = -input->params.zero_point;
  data->params.weights_offset = -filter->params.zero_point;
  data->params.output_offset = output->params.zero_point;

  // Stride
  data->params.stride_width = params->stride_width;
  data->params.stride_height = params->stride_height;

#ifdef USE_TFLM_COMPRESSION

  // Compression scratch buffers.
  // These will only be allocated if the tensor is compressed.
  data->filter_scratch_index =
      micro_context->AllocateDecompressionScratchBuffer(
          node, kTransposeConvFilterTensor);
  data->bias_scratch_index = micro_context->AllocateDecompressionScratchBuffer(
      node, kTransposeConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  micro_context->DeallocateTempTfLiteTensor(output);
  micro_context->DeallocateTempTfLiteTensor(input);
  micro_context->DeallocateTempTfLiteTensor(filter);
  return kTfLiteOk;
}

TfLiteStatus TransposeConvEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kTransposeConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kTransposeConvFilterTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 4)
          ? tflite::micro::GetEvalInput(context, node, kTransposeConvBiasTensor)
          : nullptr;
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kTransposeConvOutputTensor);

#ifdef USE_TFLM_COMPRESSION

  MicroContext* micro_context = GetMicroContext(context);

  const CompressionTensorData* filter_comp_td =
      micro_context->GetTensorCompressionData(node, kTransposeConvFilterTensor);
  const CompressionTensorData* bias_comp_td =
      micro_context->GetTensorCompressionData(node, kTransposeConvBiasTensor);

#endif  // USE_TFLM_COMPRESSION

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
      const auto& params =
          *(reinterpret_cast<TfLiteConvParams*>(node->builtin_data));
      ConvParams op_params = data.params;
      CalculateActivationRange(params.activation,
                               &op_params.float_activation_min,
                               &op_params.float_activation_max);

      reference_ops::TransposeConv(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<float>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(
              micro_context, filter, filter_comp_td, data.filter_scratch_index),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(
              micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<float>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<float>(bias),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr);
      break;
    }
    case kTfLiteInt8: {
      int32_t* scratch_buffer = static_cast<int32_t*>(
          context->GetScratchBuffer(context, data.scratch_buffer_index));
      reference_integer_ops::TransposeConv(
          data.params, data.per_channel_output_multiplier,
          data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int8_t>(
              micro_context, filter, filter_comp_td, data.filter_scratch_index),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(
              micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorData<int8_t>(filter),
          tflite::micro::GetTensorShape(bias),
          tflite::micro::GetOptionalTensorData<int32_t>(bias),
#endif  // USE_TFLM_COMPRESSION
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output),
          tflite::micro::GetTensorShape(nullptr), nullptr, scratch_buffer);
      break;
    }
    case kTfLiteInt16: {
      auto* scratch_buffer = static_cast<int64_t*>(
          context->GetScratchBuffer(context, data.scratch_buffer_index));
      if (bias != nullptr && bias->type == kTfLiteInt16) {
        auto* bias_converted_buffer =
            static_cast<int64_t*>(context->GetScratchBuffer(
                context, data.bias_converted_buffer_index));
        const int16_t* const bias_int16_data =
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int16_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index);
#else   // USE_TFLM_COMPRESSION
            static_cast<int16_t*>(bias->data.data);
#endif  // USE_TFLM_COMPRESSION
        for (int i = 0; i < tflite::micro::GetTensorShape(bias).FlatSize();
             i++) {
          bias_converted_buffer[i] = bias_int16_data[i];
        }
        reference_integer_ops::TransposeConv(
            data.params, data.per_channel_output_multiplier,
            data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 filter_comp_td,
                                                 data.filter_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(bias), bias_converted_buffer,
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output),
            tflite::micro::GetTensorShape(nullptr), nullptr, scratch_buffer);
      } else {
        reference_integer_ops::TransposeConv(
            data.params, data.per_channel_output_multiplier,
            data.per_channel_output_shift, tflite::micro::GetTensorShape(input),
            tflite::micro::GetTensorData<int16_t>(input),
            tflite::micro::GetTensorShape(filter),
#ifdef USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(micro_context, filter,
                                                 filter_comp_td,
                                                 data.filter_scratch_index),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<int64_t>(
                micro_context, bias, bias_comp_td, data.bias_scratch_index),
#else   // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorData<int8_t>(filter),
            tflite::micro::GetTensorShape(bias),
            tflite::micro::GetOptionalTensorData<int64_t>(bias),
#endif  // USE_TFLM_COMPRESSION
            tflite::micro::GetTensorShape(output),
            tflite::micro::GetTensorData<int16_t>(output),
            tflite::micro::GetTensorShape(nullptr), nullptr, scratch_buffer);
      }
      break;
    }
    default:
      MicroPrintf("Type %s (%d) not supported.", TfLiteTypeGetName(input->type),
                  input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TFLMRegistration Register_TRANSPOSE_CONV() {
  return tflite::micro::RegisterOp(TransposeConvInit, TransposeConvPrepare,
                                   TransposeConvEval);
}

}  // namespace tflite
