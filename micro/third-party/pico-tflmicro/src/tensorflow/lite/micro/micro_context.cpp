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

#include "tensorflow/lite/micro/micro_context.h"

#include <cstdarg>
#include <cstddef>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/kernels/decompress.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

int GetTensorIndex(int index, int max_size, const int* tensor_indices) {
  if (index >= 0 && index < max_size) {
    const int tensor_index = tensor_indices[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

}  // namespace

TfLiteTensor* MicroContext::AllocateTempInputTensor(const TfLiteNode* node,
                                                    int index) {
  const int tensor_index =
      GetTensorIndex(index, node->inputs->size, node->inputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempOutputTensor(const TfLiteNode* node,
                                                     int index) {
  const int tensor_index =
      GetTensorIndex(index, node->outputs->size, node->outputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

TfLiteTensor* MicroContext::AllocateTempIntermediateTensor(
    const TfLiteNode* node, int index) {
  const int tensor_index = GetTensorIndex(index, node->intermediates->size,
                                          node->intermediates->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return AllocateTempTfLiteTensor(tensor_index);
}

void MicroContextReportOpError(struct TfLiteContext* context,
                               const char* format, ...) {
  va_list args;
  va_start(args, format);
  VMicroPrintf(format, args);
  va_end(args);
}

#ifdef USE_TFLM_COMPRESSION

void* MicroContext::DecompressTensorToBuffer(
    const TfLiteEvalTensor& tensor,
    const CompressionTensorData& compression_data, void* buffer) {
  TFLITE_DCHECK(compression_data.scheme == CompressionScheme::kBinQuant);
  TFLITE_DCHECK(buffer != nullptr);
  size_t count = ElementCount(*tensor.dims);
  size_t num_channels = 1;

  if (compression_data.data.lut_data->is_per_channel_quantized) {
    const size_t channel_axis =
        compression_data.data.lut_data->use_alternate_axis
            ? tensor.dims->size - 1
            : 0;
    num_channels = tensor.dims->data[channel_axis];
  }

  DecompressionState ds(static_cast<uint8_t*>(tensor.data.data), count,
                        compression_data, num_channels, GetAlternateProfiler());

  switch (tensor.type) {
    case kTfLiteBool: {
      return ds.DecompressToBuffer<bool>(buffer);
    } break;
    case kTfLiteInt8: {
      return ds.DecompressToBuffer<int8_t>(buffer);
    } break;
    case kTfLiteInt16: {
      return ds.DecompressToBuffer<int16_t>(buffer);
    } break;
    case kTfLiteInt32: {
      return ds.DecompressToBuffer<int32_t>(buffer);
    } break;
    case kTfLiteInt64: {
      return ds.DecompressToBuffer<int64_t>(buffer);
    } break;
    case kTfLiteFloat32: {
      return ds.DecompressToBuffer<float>(buffer);
    } break;
    default: {
      MicroPrintf("Unsupported decompression tensor type %d", tensor.type);
    } break;
  }

  return nullptr;
}

TfLiteStatus MicroContext::SetDecompressionMemory(
    const std::initializer_list<AlternateMemoryRegion>& regions) {
  return kTfLiteError;
}

void* MicroContext::AllocateDecompressionMemory(size_t bytes,
                                                size_t alignment) {
  return nullptr;
}

void MicroContext::ResetDecompressionMemoryAllocations() {}

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite
