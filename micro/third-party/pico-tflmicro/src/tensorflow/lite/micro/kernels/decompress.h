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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECOMPRESS_H_
#define TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECOMPRESS_H_

#include <cstdint>

#include "tensorflow/lite/micro/compression.h"
#include "tensorflow/lite/micro/micro_profiler.h"

namespace tflite {

#ifdef USE_TFLM_COMPRESSION

struct DecompressionState {
  DecompressionState() = delete;

  DecompressionState(const uint8_t* compressed_indices,
                     const size_t count_indices,
                     const CompressionTensorData& comp_data,
                     const size_t num_channels,
                     MicroProfilerInterface* profiler = nullptr)
      : compressed_indices_(compressed_indices),
        count_indices_(count_indices),
        comp_data_(comp_data),
        num_channels_(num_channels),
        micro_profiler_(profiler) {}

  DecompressionState(const DecompressionState& other)
      : compressed_indices_(other.compressed_indices_),
        count_indices_(other.count_indices_),
        comp_data_(other.comp_data_),
        num_channels_(other.num_channels_),
        micro_profiler_(other.micro_profiler_) {}

  template <typename T>
  T* DecompressToBuffer(void* buffer);

 protected:
  // optimized C++ for INT8, use_alt_axis == false
  void DecompressToBufferWidth4_16(int8_t* buffer);
  void DecompressToBufferWidth3_32(int8_t* buffer);
  void DecompressToBufferWidth2_16(int8_t* buffer);

  // generic C++ for any bit width and value table type
  template <typename T>
  void DecompressToBufferWidthAny(T* buffer);

  // Optimized C++ table index fetch
  inline size_t GetNextTableIndexWidth7(const size_t current_offset);
  inline size_t GetNextTableIndexWidth6(const size_t current_offset);
  inline size_t GetNextTableIndexWidth5(const size_t current_offset);
  inline size_t GetNextTableIndexWidth4(const size_t current_offset);
  inline size_t GetNextTableIndexWidth3(const size_t current_offset);
  inline size_t GetNextTableIndexWidth2(const size_t current_offset);
  inline size_t GetNextTableIndexWidth1(const size_t current_offset);

 protected:
  const uint8_t* compressed_indices_;
  const size_t count_indices_;
  const CompressionTensorData& comp_data_;
  const size_t num_channels_;
  const size_t compressed_bit_width_ =
      comp_data_.data.lut_data->compressed_bit_width;
  const size_t elements_per_channel_ =
      comp_data_.data.lut_data->use_alternate_axis
          ? 1
          : count_indices_ / num_channels_;
  MicroProfilerInterface* micro_profiler_;
};

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_KERNELS_DECOMPRESS_H_
