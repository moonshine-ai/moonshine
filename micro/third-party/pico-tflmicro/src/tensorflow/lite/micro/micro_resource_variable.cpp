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

#include "tensorflow/lite/micro/micro_resource_variable.h"

#include <cstring>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

namespace {}  // namespace

MicroResourceVariables* MicroResourceVariables::Create(
    MicroAllocator* allocator, int max_num_variables) {
  TFLITE_DCHECK(allocator != nullptr);

  uint8_t* allocator_buffer = static_cast<uint8_t*>(
      allocator->AllocatePersistentBuffer(sizeof(MicroResourceVariables)));
  MicroResourceVariable* variable_array =
      static_cast<MicroResourceVariable*>(allocator->AllocatePersistentBuffer(
          sizeof(MicroResourceVariable) * max_num_variables));
  MicroResourceVariables* variables = new (allocator_buffer)
      MicroResourceVariables(variable_array, max_num_variables);
  return variables;
}

int MicroResourceVariables::CreateIdIfNoneFound(const char* container,
                                                const char* shared_name) {
  int resource_id = FindId(container, shared_name);
  if (resource_id >= 0) {
    return resource_id;
  }

  // no existing variable found for the given container and shared name pair.
  if (num_resource_variables_ >= max_variable_count_) {
    MicroPrintf(
        "Failed to allocate resource variable. Maximum resource variable count "
        "(%d) "
        "reached.",
        max_variable_count_);
    return -1;
  }

  resource_id = num_resource_variables_++;
  resource_variables_[resource_id].container = container;
  resource_variables_[resource_id].shared_name = shared_name;
  resource_variables_[resource_id].resource_buffer = nullptr;
  resource_variables_[resource_id].bytes = 0;
  resource_variables_[resource_id].default_value = 0;
  return resource_id;
}

TfLiteStatus MicroResourceVariables::Read(int id,
                                          const TfLiteEvalTensor* tensor) {
  if (id < 0 || id >= num_resource_variables_) {
    MicroPrintf("Attempting to read non-existent resource variable %d", id);
    return kTfLiteError;
  }
  MicroResourceVariable variable = resource_variables_[id];
  TFLITE_DCHECK(EvalTensorBytes(tensor) == variable.bytes);
  TFLITE_DCHECK(variable.resource_buffer != nullptr);
  memcpy(tensor->data.raw, variable.resource_buffer, variable.bytes);
  return kTfLiteOk;
}

TfLiteStatus MicroResourceVariables::Allocate(int id, TfLiteContext* context,
                                              const TfLiteTensor* tensor) {
  if (id < 0 || id >= num_resource_variables_) {
    MicroPrintf("Attempting to read non-existent resource variable %d", id);
    return kTfLiteError;
  }

  MicroResourceVariable& variable = resource_variables_[id];

  if (variable.resource_buffer == nullptr) {
    variable.bytes = tensor->bytes;
    variable.resource_buffer =
        context->AllocatePersistentBuffer(context, tensor->bytes);
    if (variable.resource_buffer == nullptr) {
      MicroPrintf("Failed to allocate resource buffer.");
      return kTfLiteError;
    }
    // Set resource buffers to the zero_point by default. Buffers can be
    // initialized to nonzero values using ASSIGN_VARIABLE.
    // See comment#2 in b/269648474 for more details why we use zero_point.
    if (tensor->quantization.params != nullptr) {
      auto* quantization_data = reinterpret_cast<TfLiteAffineQuantization*>(
          tensor->quantization.params);
      int8_t zero_point = quantization_data->zero_point[0].data[0];
      variable.default_value = zero_point;
    }
    // TODO(b/269669735): Explains why casting zero_point to int8 and memset.
    memset(variable.resource_buffer, variable.default_value, variable.bytes);
  }

  return kTfLiteOk;
}

TfLiteStatus MicroResourceVariables::Assign(int id, size_t count_bytes,
                                            const void* input_buffer) {
  if (id < 0 || id >= num_resource_variables_) {
    MicroPrintf("Attempting to read non-existent resource variable %d", id);
    return kTfLiteError;
  }
  MicroResourceVariable variable = resource_variables_[id];

  if (variable.resource_buffer == nullptr) {
    MicroPrintf(
        "Attempting to assign from a TfLiteEvalTensor before the resource "
        "buffer has been allocated. Make sure to call AssignResourceVariable "
        "with a TfLiteTensor first.");
    return kTfLiteError;
  }
  TFLITE_DCHECK(count_bytes == variable.bytes);
  TFLITE_DCHECK(input_buffer != nullptr);
  memcpy(variable.resource_buffer, input_buffer, variable.bytes);
  return kTfLiteOk;
}

TfLiteStatus MicroResourceVariables::ResetAll() {
  for (int i = 0; i < num_resource_variables_; i++) {
    MicroResourceVariable variable = resource_variables_[i];
    // TODO(b/269669735): Explains why casting zero_point to int8 and memset.
    memset(variable.resource_buffer, variable.default_value, variable.bytes);
  }
  return kTfLiteOk;
}

int MicroResourceVariables::FindId(const char* container,
                                   const char* shared_name) {
  for (int i = 0; i < num_resource_variables_; i++) {
    // Some TFLite flatbuffers contain null container names to save space.
    if ((container == nullptr ||
         !strcmp(container, resource_variables_[i].container)) &&
        !strcmp(shared_name, resource_variables_[i].shared_name)) {
      return i;
    }
  }
  return -1;
}

}  // namespace tflite
