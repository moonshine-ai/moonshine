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

#ifdef USE_TFLM_COMPRESSION

#include "tensorflow/lite/micro/kernels/decompress.h"

#include <cstddef>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {

template <typename T>
T* DecompressionState::DecompressToBuffer(void* buffer) {
  TFLITE_DCHECK(compressed_bit_width_ <= LookupTableData::kMaxBitWidth);
  TFLITE_DCHECK(compressed_bit_width_ > 0);

  if (std::is_same<T, int8_t>::value &&
      comp_data_.data.lut_data->compressed_bit_width == 4 &&
      !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth4_16(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value &&
             comp_data_.data.lut_data->compressed_bit_width == 3 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth3_32(static_cast<int8_t*>(buffer));
  } else if (std::is_same<T, int8_t>::value &&
             comp_data_.data.lut_data->compressed_bit_width == 2 &&
             !comp_data_.data.lut_data->use_alternate_axis) {
    DecompressToBufferWidth2_16(static_cast<int8_t*>(buffer));
  } else {
    DecompressToBufferWidthAny<T>(static_cast<T*>(buffer));
  }

  return static_cast<T*>(buffer);
}

template bool* DecompressionState::DecompressToBuffer<bool>(void*);
template float* DecompressionState::DecompressToBuffer<float>(void*);
template int8_t* DecompressionState::DecompressToBuffer<int8_t>(void*);
template int16_t* DecompressionState::DecompressToBuffer<int16_t>(void*);
template int32_t* DecompressionState::DecompressToBuffer<int32_t>(void*);
template int64_t* DecompressionState::DecompressToBuffer<int64_t>(void*);

}  // namespace tflite

#endif  // USE_TFLM_COMPRESSION
