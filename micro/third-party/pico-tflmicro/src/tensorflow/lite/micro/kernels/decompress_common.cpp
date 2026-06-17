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

#include <cstddef>
#include <type_traits>

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/kernels/decompress.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"

namespace tflite {

void DecompressionState::DecompressToBufferWidth4_16(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;
  size_t current_offset = 0;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    size_t count = max_count;

    // process elements at start of channel up to next uint64_t alignment of
    // compressed_indices_
    while (count > 0 && (current_offset & 0x0F)) {
      const size_t index = GetNextTableIndexWidth4(current_offset++);
      *buffer++ = value_table[index];
      count -= 1;
    }

    // process elements in current channel in groups of 16
    if (count >= 16) {
      const uint64_t* indices = reinterpret_cast<const uint64_t*>(
          &compressed_indices_[current_offset >> 1]);

      while (count >= 16) {
        count -= 16;
        uint64_t index = *indices++;
        uint64_t value, value2;

        value = static_cast<uint64_t>(value_table[(index >> 4) & 0x0F]);
        value |= static_cast<uint64_t>(value_table[index & 0x0F]) << 8;
        value |= static_cast<uint64_t>(value_table[(index >> 12) & 0x0F]) << 16;
        value |= static_cast<uint64_t>(value_table[(index >> 8) & 0x0F]) << 24;
        value |= static_cast<uint64_t>(value_table[(index >> 20) & 0x0F]) << 32;
        value |= static_cast<uint64_t>(value_table[(index >> 16) & 0x0F]) << 40;
        value |= static_cast<uint64_t>(value_table[(index >> 28) & 0x0F]) << 48;
        value |= static_cast<uint64_t>(value_table[(index >> 24) & 0x0F]) << 56;

        *reinterpret_cast<uint64_t*>(buffer) = value;

        value2 = static_cast<uint64_t>(value_table[(index >> 36) & 0x0F]);
        value2 |= static_cast<uint64_t>(value_table[(index >> 32) & 0x0F]) << 8;
        value2 |= static_cast<uint64_t>(value_table[(index >> 44) & 0x0F])
                  << 16;
        value2 |= static_cast<uint64_t>(value_table[(index >> 40) & 0x0F])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index >> 52) & 0x0F])
                  << 32;
        value2 |= static_cast<uint64_t>(value_table[(index >> 48) & 0x0F])
                  << 40;
        value2 |= static_cast<uint64_t>(value_table[(index >> 60) & 0x0F])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index >> 56) & 0x0F])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 8) = value2;

        buffer += 16;
      }

      current_offset =
          (reinterpret_cast<const uint8_t*>(indices) - compressed_indices_)
          << 1;
    }

    // process remaining elements in current channel
    while (count > 0) {
      count -= 1;
      const size_t index = GetNextTableIndexWidth4(current_offset++);
      *buffer++ = value_table[index];
    }

    value_table += stride;
  }
}

void DecompressionState::DecompressToBufferWidth2_16(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;
  size_t current_offset = 0;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    size_t count = max_count;

    // process elements at start of channel up to next uint32_t alignment of
    // compressed_indices_
    while (count > 0 && (current_offset & 0x0F)) {
      const size_t index = GetNextTableIndexWidth2(current_offset++);
      *buffer++ = value_table[index];
      count -= 1;
    }

    // process elements in current channel in groups of 16
    if (count >= 16) {
      const uint32_t* indices = reinterpret_cast<const uint32_t*>(
          &compressed_indices_[current_offset >> 2]);

      while (count >= 16) {
        count -= 16;
        uint32_t index = *indices++;
        uint64_t value, value2;

        value = static_cast<uint64_t>(value_table[(index >> 6) & 0x03]);
        value |= static_cast<uint64_t>(value_table[(index >> 4) & 0x03]) << 8;
        value |= static_cast<uint64_t>(value_table[(index >> 2) & 0x03]) << 16;
        value |= static_cast<uint64_t>(value_table[index & 0x03]) << 24;
        value |= static_cast<uint64_t>(value_table[(index >> 14) & 0x03]) << 32;
        value |= static_cast<uint64_t>(value_table[(index >> 12) & 0x03]) << 40;
        value |= static_cast<uint64_t>(value_table[(index >> 10) & 0x03]) << 48;
        value |= static_cast<uint64_t>(value_table[(index >> 8) & 0x03]) << 56;

        *reinterpret_cast<uint64_t*>(buffer) = value;

        value2 = static_cast<uint64_t>(value_table[(index >> 22) & 0x03]);
        value2 |= static_cast<uint64_t>(value_table[(index >> 20) & 0x03]) << 8;
        value2 |= static_cast<uint64_t>(value_table[(index >> 18) & 0x03])
                  << 16;
        value2 |= static_cast<uint64_t>(value_table[(index >> 16) & 0x03])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index >> 30) & 0x03])
                  << 32;
        value2 |= static_cast<uint64_t>(value_table[(index >> 28) & 0x03])
                  << 40;
        value2 |= static_cast<uint64_t>(value_table[(index >> 26) & 0x03])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index >> 24) & 0x03])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 8) = value2;

        buffer += 16;
      }

      current_offset =
          (reinterpret_cast<const uint8_t*>(indices) - compressed_indices_)
          << 2;
    }

    // process remaining elements in current channel
    while (count > 0) {
      count -= 1;
      const size_t index = GetNextTableIndexWidth2(current_offset++);
      *buffer++ = value_table[index];
    }

    value_table += stride;
  }
}

void DecompressionState::DecompressToBufferWidth3_32(int8_t* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
  const uint8_t* value_table =
      static_cast<const uint8_t*>(comp_data_.data.lut_data->value_table);
  const size_t max_count = elements_per_channel_;
  size_t current_offset = 0;

  for (size_t channel = 0; channel < num_channels_; channel++) {
    size_t count = max_count;

    // process elements at start of channel up to next uint32_t alignment of
    // compressed_indices_
    while (count > 0 && (current_offset & 0x1F)) {
      const size_t index = GetNextTableIndexWidth3(current_offset++);
      *buffer++ = value_table[index];
      count -= 1;
    }

    // process elements in current channel in groups of 32
    if (count >= 32) {
      const uint32_t* indices = reinterpret_cast<const uint32_t*>(
          &compressed_indices_[(current_offset >> 5) * 12]);

      while (count >= 32) {
        count -= 32;
        uint32_t index0 = *indices++;
        uint32_t index1 = *indices++;
        uint32_t index2 = *indices++;
        uint64_t value, value2;

        value = static_cast<uint64_t>(value_table[(index0 >> 5) & 0x07]);
        value |= static_cast<uint64_t>(value_table[(index0 >> 2) & 0x07]) << 8;
        value |=
            static_cast<uint64_t>(
                value_table[((index0 << 1) & 0b110) | ((index0 >> 15) & 0b1)])
            << 16;
        value |= static_cast<uint64_t>(value_table[(index0 >> 12) & 0x07])
                 << 24;
        value |= static_cast<uint64_t>(value_table[(index0 >> 9) & 0x07]) << 32;
        value |=
            static_cast<uint64_t>(
                value_table[((index0 >> 6) & 0b100) | ((index0 >> 22) & 0b11)])
            << 40;
        value |= static_cast<uint64_t>(value_table[(index0 >> 19) & 0x07])
                 << 48;
        value |= static_cast<uint64_t>(value_table[(index0 >> 16) & 0x07])
                 << 56;

        *reinterpret_cast<uint64_t*>(buffer) = value;

        value2 = static_cast<uint64_t>(value_table[(index0 >> 29) & 0x07]);
        value2 |= static_cast<uint64_t>(value_table[(index0 >> 26) & 0x07])
                  << 8;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index0 >> 23) & 0b110) | ((index1 >> 7) & 0b1)])
            << 16;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 4) & 0x07])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 1) & 0x07])
                  << 32;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index1 << 2) & 0b100) | ((index1 >> 14) & 0b11)])
            << 40;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 11) & 0x07])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index1 >> 8) & 0x07])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 8) = value2;

        value = static_cast<uint64_t>(value_table[(index1 >> 21) & 0x07]);
        value |= static_cast<uint64_t>(value_table[(index1 >> 18) & 0x07]) << 8;
        value |=
            static_cast<uint64_t>(
                value_table[((index1 >> 15) & 0b110) | ((index1 >> 31) & 0b1)])
            << 16;
        value |= static_cast<uint64_t>(value_table[(index1 >> 28) & 0x07])
                 << 24;
        value |= static_cast<uint64_t>(value_table[(index1 >> 25) & 0x07])
                 << 32;
        value |=
            static_cast<uint64_t>(
                value_table[((index1 >> 22) & 0b100) | ((index2 >> 6) & 0b11)])
            << 40;
        value |= static_cast<uint64_t>(value_table[(index2 >> 3) & 0x07]) << 48;
        value |= static_cast<uint64_t>(value_table[(index2 >> 0) & 0x07]) << 56;

        *reinterpret_cast<uint64_t*>(buffer + 16) = value;

        value2 = static_cast<uint64_t>(value_table[(index2 >> 13) & 0x07]);
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 10) & 0x07])
                  << 8;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index2 >> 7) & 0b110) | ((index2 >> 23) & 0b1)])
            << 16;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 20) & 0x07])
                  << 24;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 17) & 0x07])
                  << 32;
        value2 |=
            static_cast<uint64_t>(
                value_table[((index2 >> 14) & 0b100) | ((index2 >> 30) & 0b11)])
            << 40;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 27) & 0x07])
                  << 48;
        value2 |= static_cast<uint64_t>(value_table[(index2 >> 24) & 0x07])
                  << 56;

        *reinterpret_cast<uint64_t*>(buffer + 24) = value2;

        buffer += 32;
        current_offset += 32;
      }
    }

    // process remaining elements in current channel
    while (count > 0) {
      count -= 1;
      const size_t index = GetNextTableIndexWidth3(current_offset++);
      *buffer++ = value_table[index];
    }

    value_table += stride;
  }
}

// TODO(ddavis-2015): templating GetNextTableIndexWidth<N> makes this method
// more than 2x faster, but with a large code size increase
template <typename T>
void DecompressionState::DecompressToBufferWidthAny(T* buffer) {
  ScopedMicroProfiler scoped_profiler(__func__, micro_profiler_);

  if (comp_data_.data.lut_data->use_alternate_axis) {
    const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
    size_t current_offset = 0;
    size_t count = count_indices_;

    while (count > 0) {
      const T* value_table =
          static_cast<const T*>(comp_data_.data.lut_data->value_table);
      for (size_t channel = 0; channel < num_channels_; channel++) {
        size_t index;
        switch (compressed_bit_width_) {
          case 1:
            index = GetNextTableIndexWidth1(current_offset);
            break;
          case 2:
            index = GetNextTableIndexWidth2(current_offset);
            break;
          case 3:
            index = GetNextTableIndexWidth3(current_offset);
            break;
          case 4:
            index = GetNextTableIndexWidth4(current_offset);
            break;
          case 5:
            index = GetNextTableIndexWidth5(current_offset);
            break;
          case 6:
            index = GetNextTableIndexWidth6(current_offset);
            break;
          case 7:
            index = GetNextTableIndexWidth7(current_offset);
            break;
        }
        current_offset++;
        *buffer++ = value_table[index];
        value_table += stride;
      }
      count -= num_channels_;
    }
  } else {
    const size_t stride = comp_data_.data.lut_data->value_table_channel_stride;
    const T* value_table =
        static_cast<const T*>(comp_data_.data.lut_data->value_table);
    const size_t max_count = elements_per_channel_;
    size_t current_offset = 0;

    for (size_t channel = 0; channel < num_channels_; channel++) {
      size_t count = max_count;

      while (count-- > 0) {
        size_t index;
        switch (compressed_bit_width_) {
          case 1:
            index = GetNextTableIndexWidth1(current_offset);
            break;
          case 2:
            index = GetNextTableIndexWidth2(current_offset);
            break;
          case 3:
            index = GetNextTableIndexWidth3(current_offset);
            break;
          case 4:
            index = GetNextTableIndexWidth4(current_offset);
            break;
          case 5:
            index = GetNextTableIndexWidth5(current_offset);
            break;
          case 6:
            index = GetNextTableIndexWidth6(current_offset);
            break;
          case 7:
            index = GetNextTableIndexWidth7(current_offset);
            break;
        }
        current_offset++;
        *buffer++ = value_table[index];
      }
      value_table += stride;
    }
  }
}

template void DecompressionState::DecompressToBufferWidthAny(bool*);
template void DecompressionState::DecompressToBufferWidthAny(float*);
template void DecompressionState::DecompressToBufferWidthAny(int8_t*);
template void DecompressionState::DecompressToBufferWidthAny(int16_t*);
template void DecompressionState::DecompressToBufferWidthAny(int32_t*);
template void DecompressionState::DecompressToBufferWidthAny(int64_t*);

inline size_t DecompressionState::GetNextTableIndexWidth7(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 3) * 7;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b111) {
    case 0:
      return indices[0] >> 1;
    case 1:
      return ((indices[0] & 0b1) << 6) | (indices[1] >> 2);
    case 2:
      return ((indices[1] & 0b11) << 5) | (indices[2] >> 3);
    case 3:
      return ((indices[2] & 0b111) << 4) | (indices[3] >> 4);
    case 4:
      return ((indices[3] & 0x0F) << 3) | (indices[4] >> 5);
    case 5:
      return ((indices[4] & 0x1F) << 2) | (indices[5] >> 6);
    case 6:
      return ((indices[5] & 0x3F) << 1) | (indices[6] >> 7);
    case 7:
      return indices[6] & 0x7F;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecompressionState::GetNextTableIndexWidth6(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 2) * 3;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b11) {
    case 0:
      return indices[0] >> 2;
    case 1:
      return ((indices[0] & 0b11) << 4) | (indices[1] >> 4);
    case 2:
      return ((indices[1] & 0x0F) << 2) | (indices[2] >> 6);
    case 3:
      return indices[2] & 0x3F;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecompressionState::GetNextTableIndexWidth5(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 3) * 5;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b111) {
    case 0:
      return indices[0] >> 3;
    case 1:
      return ((indices[0] & 0b111) << 2) | (indices[1] >> 6);
    case 2:
      return (indices[1] >> 1) & 0x1F;
    case 3:
      return ((indices[1] & 0b1) << 4) | (indices[2] >> 4);
    case 4:
      return ((indices[2] & 0x0F) << 1) | (indices[3] >> 7);
    case 5:
      return (indices[3] >> 2) & 0x1F;
    case 6:
      return ((indices[3] & 0b11) << 3) | (indices[4] >> 5);
    case 7:
      return indices[4] & 0x1F;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecompressionState::GetNextTableIndexWidth4(
    const size_t current_offset) {
  if (current_offset & 1) {
    return compressed_indices_[current_offset >> 1] & 0x0F;
  } else {
    return compressed_indices_[current_offset >> 1] >> 4;
  }
}

inline size_t DecompressionState::GetNextTableIndexWidth3(
    const size_t current_offset) {
  const size_t current_byte_index = (current_offset >> 3) * 3;
  const uint8_t* indices = &compressed_indices_[current_byte_index];
  switch (current_offset & 0b111) {
    case 0:
      return indices[0] >> 5;
    case 1:
      return (indices[0] >> 2) & 0b111;
    case 2:
      return ((indices[0] & 0b11) << 1) | (indices[1] >> 7);
    case 3:
      return (indices[1] >> 4) & 0b111;
    case 4:
      return (indices[1] >> 1) & 0b111;
    case 5:
      return ((indices[1] & 0b1) << 2) | (indices[2] >> 6);
    case 6:
      return (indices[2] >> 3) & 0b111;
    case 7:
      return indices[2] & 0b111;
  }
  // NOTREACHED
  return 0;
}

inline size_t DecompressionState::GetNextTableIndexWidth2(
    const size_t current_offset) {
  if (current_offset & 0b10) {
    if (current_offset & 1) {
      return compressed_indices_[current_offset >> 2] & 0x03;
    } else {
      return (compressed_indices_[current_offset >> 2] >> 2) & 0x03;
    }
  } else {
    if (current_offset & 1) {
      return (compressed_indices_[current_offset >> 2] >> 4) & 0x03;
    } else {
      return (compressed_indices_[current_offset >> 2] >> 6) & 0x03;
    }
  }
}

inline size_t DecompressionState::GetNextTableIndexWidth1(
    const size_t current_offset) {
  const size_t shift = ~current_offset & 0b111;
  return (compressed_indices_[current_offset >> 3] >> shift) & 0b1;
}

}  // namespace tflite

#endif  // USE_TFLM_COMPRESSION
