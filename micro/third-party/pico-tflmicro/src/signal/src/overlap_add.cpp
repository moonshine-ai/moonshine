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

#include <stdio.h>
#include <string.h>

namespace tflm_signal {

void OverlapAdd(const int16_t* input, int16_t* buffer, int input_size,
                int16_t* output, int output_size) {
  for (int i = 0; i < input_size; ++i) {
    int32_t overlap_added_sample = input[i] + buffer[i];
    if (overlap_added_sample < INT16_MIN) {
      buffer[i] = INT16_MIN;
    } else if (overlap_added_sample > INT16_MAX) {
      buffer[i] = INT16_MAX;
    } else {
      buffer[i] = (int16_t)overlap_added_sample;
    }
  }
  memcpy(output, buffer, output_size * sizeof(output[0]));
  memmove(buffer, &buffer[output_size],
          (input_size - output_size) * sizeof(buffer[0]));
  memset(&buffer[input_size - output_size], 0, output_size * sizeof(buffer[0]));
}

void OverlapAdd(const float* input, float* buffer, int input_size,
                float* output, int output_size) {
  for (int i = 0; i < input_size; ++i) {
    buffer[i] += input[i];
  }
  memcpy(output, buffer, output_size * sizeof(output[0]));
  memmove(buffer, &buffer[output_size],
          (input_size - output_size) * sizeof(buffer[0]));
  memset(&buffer[input_size - output_size], 0, output_size * sizeof(buffer[0]));
}

}  // namespace tflm_signal
