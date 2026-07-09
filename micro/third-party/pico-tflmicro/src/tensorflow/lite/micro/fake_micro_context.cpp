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

#include "tensorflow/lite/micro/fake_micro_context.h"

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

FakeMicroContext::FakeMicroContext(
    TfLiteTensor* tensors, SingleArenaBufferAllocator* allocator,
    MicroGraph* micro_graph
#ifdef USE_TFLM_COMPRESSION
    ,
    const CompressedTensorList* compressed_tensors
#endif  // USE_TFLM_COMPRESSION
    )
    : graph_(*micro_graph),
      tensors_(tensors),
      allocator_(allocator)
#ifdef USE_TFLM_COMPRESSION
      ,
      compressed_tensors_(compressed_tensors)
#endif  // USE_TFLM_COMPRESSION
{
}

TfLiteTensor* FakeMicroContext::AllocateTempTfLiteTensor(int tensor_index) {
  allocated_temp_count_++;
  return &tensors_[tensor_index];
}

void FakeMicroContext::DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
  allocated_temp_count_--;
}

bool FakeMicroContext::IsAllTempTfLiteTensorDeallocated() {
  return !allocated_temp_count_;
}

uint8_t* FakeMicroContext::AllocateTempBuffer(size_t size, size_t alignment) {
  allocated_temp_count_++;
  return allocator_->AllocateTemp(size, alignment);
}

void FakeMicroContext::DeallocateTempBuffer(uint8_t* buffer) {
  allocated_temp_count_--;
  allocator_->DeallocateTemp(buffer);
}

TfLiteEvalTensor* FakeMicroContext::GetEvalTensor(int tensor_index) {
  TfLiteEvalTensor* eval_tensor =
      reinterpret_cast<TfLiteEvalTensor*>(allocator_->AllocateTemp(
          sizeof(TfLiteEvalTensor), alignof(TfLiteEvalTensor)));
  TFLITE_DCHECK(eval_tensor != nullptr);

  // In unit tests, the TfLiteTensor pointer contains the source of truth for
  // buffers and values:
  eval_tensor->data = tensors_[tensor_index].data;
  eval_tensor->dims = tensors_[tensor_index].dims;
  eval_tensor->type = tensors_[tensor_index].type;
  return eval_tensor;
}

void* FakeMicroContext::AllocatePersistentBuffer(size_t bytes) {
  // FakeMicroContext use SingleArenaBufferAllocator, which does not
  // automatically apply the buffer alignment like MicroAllocator. The buffer
  // alignment is potentially wasteful but allows the fake_micro_context to work
  // correctly with optimized kernels.
  return allocator_->AllocatePersistentBuffer(bytes,
                                              MicroArenaBufferAlignment());
}

TfLiteStatus FakeMicroContext::RequestScratchBufferInArena(size_t bytes,
                                                           int* buffer_index) {
  TFLITE_DCHECK(buffer_index != nullptr);

  if (scratch_buffer_count_ == kNumScratchBuffers_) {
    MicroPrintf("Exceeded the maximum number of scratch tensors allowed (%d).",
                kNumScratchBuffers_);
    return kTfLiteError;
  }

  // For tests, we allocate scratch buffers from the tail and keep them around
  // for the lifetime of model. This means that the arena size in the tests will
  // be more than what we would have if the scratch buffers could share memory.
  scratch_buffers_[scratch_buffer_count_] =
      allocator_->AllocatePersistentBuffer(bytes, MicroArenaBufferAlignment());
  TFLITE_DCHECK(scratch_buffers_[scratch_buffer_count_] != nullptr);

  *buffer_index = scratch_buffer_count_++;
  return kTfLiteOk;
}

void* FakeMicroContext::GetScratchBuffer(int buffer_index) {
  TFLITE_DCHECK(scratch_buffer_count_ <= kNumScratchBuffers_);
  if (buffer_index >= scratch_buffer_count_) {
    return nullptr;
  }
  return scratch_buffers_[buffer_index];
}

TfLiteStatus FakeMicroContext::set_external_context(
    void* external_context_payload) {
  return kTfLiteError;
}

void* FakeMicroContext::external_context() { return nullptr; }

MicroGraph& FakeMicroContext::graph() { return graph_; }

#ifdef USE_TFLM_COMPRESSION

// Available during Prepare & Eval. Returns false if tensor is not
// compressed.
bool FakeMicroContext::IsTensorCompressed(const TfLiteNode* node,
                                          int tensor_idx) {
  if (compressed_tensors_ != nullptr && tensor_idx < node->inputs->size) {
    int index = node->inputs->data[tensor_idx];
    if (index >= 0 && compressed_tensors_->tensors[index] != nullptr) {
      return true;
    }
  }

  return false;
}

// Only available during Prepare. The kernel is responsible for storing the
// scratch buffer handle.
int FakeMicroContext::AllocateDecompressionScratchBuffer(const TfLiteNode* node,
                                                         int tensor_idx) {
  if (compressed_tensors_ == nullptr || tensor_idx >= node->inputs->size) {
    return -1;
  }
  int index = node->inputs->data[tensor_idx];
  if (index < 0 || compressed_tensors_->tensors[index] == nullptr) {
    return -1;
  }
  TfLiteTensor* tensor = &tensors_[index];
  int scratch_index = -1;
  TfLiteStatus result =
      RequestScratchBufferInArena(tensor->bytes, &scratch_index);
  if (result != kTfLiteOk) {
    return -1;
  }

  return scratch_index;
}

// Available during Prepare & Eval. Returns nullptr if tensor is not
// compressed.
const CompressionTensorData* FakeMicroContext::GetTensorCompressionData(
    const TfLiteNode* node, int tensor_idx) {
  if (compressed_tensors_ == nullptr || tensor_idx >= node->inputs->size) {
    return nullptr;
  }

  int index = node->inputs->data[tensor_idx];
  if (index < 0) {
    return nullptr;
  }

  return compressed_tensors_->tensors[index];
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite
