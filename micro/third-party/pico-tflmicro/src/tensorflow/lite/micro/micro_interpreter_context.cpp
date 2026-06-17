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

#include "tensorflow/lite/micro/micro_interpreter_context.h"

#include <cstdint>

#ifdef USE_TFLM_COMPRESSION

#include <algorithm>

#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_arena_constants.h"

#endif  // USE_TFLM_COMPRESSION

#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {

namespace {

#ifdef USE_TFLM_COMPRESSION

int GetInputTensorIndex(const TfLiteNode* node, const int index) {
  if (index >= 0 && index < node->inputs->size) {
    const int tensor_index = node->inputs->data[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace

MicroInterpreterContext::MicroInterpreterContext(MicroAllocator* allocator,
                                                 const Model* model,
                                                 MicroInterpreterGraph* graph)
    : allocator_(*allocator),
      graph_(*graph),
      model_(model),
      state_(InterpreterState::kInit) {}

MicroInterpreterContext::~MicroInterpreterContext() {}

void* MicroInterpreterContext::AllocatePersistentBuffer(size_t bytes) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInit);
  return allocator_.AllocatePersistentBuffer(bytes);
}

TfLiteStatus MicroInterpreterContext::RequestScratchBufferInArena(
    size_t bytes, int* buffer_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  return allocator_.RequestScratchBufferInArena(
      bytes, graph_.GetCurrentSubgraphIndex(), buffer_idx);
}

void* MicroInterpreterContext::GetScratchBuffer(int buffer_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kInvoke);
  ScratchBufferHandle* handle = scratch_buffer_handles_ + buffer_idx;
  return handle->data;
}

TfLiteTensor* MicroInterpreterContext::AllocateTempTfLiteTensor(
    int tensor_idx) {
  return allocator_.AllocateTempTfLiteTensor(model_, graph_.GetAllocations(),
                                             tensor_idx,
                                             graph_.GetCurrentSubgraphIndex());
}

void MicroInterpreterContext::DeallocateTempTfLiteTensor(TfLiteTensor* tensor) {
  return allocator_.DeallocateTempTfLiteTensor(tensor);
}

uint8_t* MicroInterpreterContext::AllocateTempBuffer(size_t size,
                                                     size_t alignment) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  return allocator_.AllocateTempBuffer(size, alignment);
}

void MicroInterpreterContext::DeallocateTempBuffer(uint8_t* buffer) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);
  allocator_.DeallocateTempBuffer(buffer);
}

TfLiteEvalTensor* MicroInterpreterContext::GetEvalTensor(int tensor_idx) {
  return &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()]
              .tensors[tensor_idx];
}

void MicroInterpreterContext::SetScratchBufferHandles(
    ScratchBufferHandle* scratch_buffer_handles) {
  scratch_buffer_handles_ = scratch_buffer_handles;
}

TfLiteStatus MicroInterpreterContext::set_external_context(
    void* external_context_payload) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);
  if (external_context_payload == nullptr ||
      external_context_payload_ != nullptr) {
    MicroPrintf(
        "Attempting to set external context to %x but it was %x already",
        external_context_payload, external_context_payload_);
    return kTfLiteError;
  }

  external_context_payload_ = external_context_payload;
  return kTfLiteOk;
}

void MicroInterpreterContext::SetInterpreterState(InterpreterState state) {
  state_ = state;
}

MicroInterpreterContext::InterpreterState
MicroInterpreterContext::GetInterpreterState() const {
  return state_;
}

#ifdef USE_TFLM_COMPRESSION

// Available during Prepare & Eval. Returns false if tensor is not
// compressed.
bool MicroInterpreterContext::IsTensorCompressed(const TfLiteNode* node,
                                                 int tensor_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);

  const SubgraphAllocations* allocations =
      &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()];
  if (allocations->compressed.tensors == nullptr) {
    return false;
  }
  int index = GetInputTensorIndex(node, tensor_idx);
  if (index == -1) {
    return false;
  }
  return allocations->compressed.tensors[index] != nullptr;
}

// Only available during Prepare. The kernel is responsible for storing the
// scratch buffer handle.
int MicroInterpreterContext::AllocateDecompressionScratchBuffer(
    const TfLiteNode* node, int tensor_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare);

  const SubgraphAllocations* allocations =
      &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()];
  if (allocations->compressed.tensors == nullptr) {
    return -1;
  }
  int index = GetInputTensorIndex(node, tensor_idx);
  if (index == -1 || allocations->compressed.tensors[index] == nullptr) {
    return -1;
  }
  const TfLiteEvalTensor* tensor = &allocations->tensors[index];
  const size_t byte_count = EvalTensorBytes(tensor);

  if (AllocateDecompressionMemory(byte_count, MicroArenaBufferAlignment()) !=
      nullptr) {
    // Tensor fits in alternate decompression memory, no need to allocate
    // scratch buffer.
    return -1;
  }

  int scratch_index = -1;
  TfLiteStatus result = RequestScratchBufferInArena(byte_count, &scratch_index);
  TFLITE_DCHECK(scratch_index != -1 && result == kTfLiteOk);

  return scratch_index;
}

// Available during Prepare & Eval. Returns nullptr if tensor is not
// compressed.
const CompressionTensorData* MicroInterpreterContext::GetTensorCompressionData(
    const TfLiteNode* node, int tensor_idx) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);

  const SubgraphAllocations* allocations =
      &graph_.GetAllocations()[graph_.GetCurrentSubgraphIndex()];
  if (allocations->compressed.tensors == nullptr) {
    return nullptr;
  }
  int index = GetInputTensorIndex(node, tensor_idx);
  if (index == -1) {
    return nullptr;
  }
  return allocations->compressed.tensors[index];
}

// Only available during Prepare & Eval. Returns nullptr on failure, otherwise
// returns a pointer to the buffer.
void* MicroInterpreterContext::DecompressTensorToBuffer(
    const TfLiteEvalTensor& tensor,
    const CompressionTensorData& compression_data, void* buffer) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);

  return MicroContext::DecompressTensorToBuffer(tensor, compression_data,
                                                buffer);
}

TfLiteStatus MicroInterpreterContext::SetDecompressionMemory(
    const std::initializer_list<MicroContext::AlternateMemoryRegion>& regions) {
  if (state_ != InterpreterState::kInit) {
    return kTfLiteError;
  }

  decompress_regions_ = &regions;
  decompress_regions_allocations_ = static_cast<size_t*>(
      AllocatePersistentBuffer(sizeof(size_t) * regions.size()));
  if (decompress_regions_allocations_ == nullptr) {
    return kTfLiteError;
  }
  ResetDecompressionMemoryAllocations();

  return kTfLiteOk;
}

void* MicroInterpreterContext::AllocateDecompressionMemory(size_t bytes,
                                                           size_t alignment) {
  TFLITE_DCHECK(state_ == InterpreterState::kPrepare ||
                state_ == InterpreterState::kInvoke);
  if (decompress_regions_ != nullptr) {
    for (size_t i = 0; i < decompress_regions_->size(); i++) {
      const AlternateMemoryRegion* region = &decompress_regions_->begin()[i];
      uint8_t* start = static_cast<uint8_t*>(region->address) +
                       decompress_regions_allocations_[i];
      uint8_t* aligned_start = AlignPointerUp(start, alignment);
      size_t total = bytes + (aligned_start - start);
      if (total + decompress_regions_allocations_[i] <= region->bytes) {
        decompress_regions_allocations_[i] += total;
        return aligned_start;
      }
    }
  }

  return nullptr;
}

void MicroInterpreterContext::ResetDecompressionMemoryAllocations() {
  if (decompress_regions_ == nullptr) {
    return;
  }
  TFLITE_DCHECK(decompress_regions_allocations_ != nullptr);
  std::fill_n(decompress_regions_allocations_, decompress_regions_->size(), 0);
}

#endif  // USE_TFLM_COMPRESSION

TfLiteStatus MicroInterpreterContext::SetAlternateProfiler(
    tflite::MicroProfilerInterface* alt_profiler) {
  alt_profiler_ = alt_profiler;
  return kTfLiteOk;
}

MicroProfilerInterface* MicroInterpreterContext::GetAlternateProfiler() const {
  return alt_profiler_;
}

}  // namespace tflite
