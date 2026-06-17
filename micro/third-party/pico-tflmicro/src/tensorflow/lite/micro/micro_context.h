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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_
#define TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_graph.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"

#ifdef USE_TFLM_COMPRESSION

#include <initializer_list>

#include "tensorflow/lite/micro/compression.h"

#endif  // USE_TFLM_COMPRESSION

namespace tflite {
// TODO(b/149795762): kTfLiteAbort cannot be part of the tflite TfLiteStatus.
const TfLiteStatus kTfLiteAbort = static_cast<TfLiteStatus>(15);

// MicroContext is eventually going to become the API between TFLM and the
// kernels, replacing all the functions in TfLiteContext. The end state is code
// kernels to have code like:
//
// MicroContext* micro_context = GetMicroContext(context);
// micro_context-><TFLM kernel API>
class MicroContext {
 public:
  virtual ~MicroContext() = default;

  // Allocate persistent buffer which has the same life time as the interpreter.
  // Returns nullptr on failure.
  // The memory is allocated from the tail.
  // This method is only available in Init or Prepare stage.
  virtual void* AllocatePersistentBuffer(size_t bytes) = 0;

  // Request a scratch buffer in the arena through static memory planning.
  // This method is only available in Prepare stage and the buffer is allocated
  // by the interpreter between Prepare and Eval stage. In Eval stage,
  // GetScratchBuffer API can be used to fetch the address.
  virtual TfLiteStatus RequestScratchBufferInArena(size_t bytes,
                                                   int* buffer_idx) = 0;

  // Get the scratch buffer pointer.
  // This method is only available in Eval stage.
  virtual void* GetScratchBuffer(int buffer_idx) = 0;

  // Returns a temporary TfLiteTensor struct for a given index.
  virtual TfLiteTensor* AllocateTempTfLiteTensor(int tensor_idx) = 0;

  // Returns a temporary TfLiteTensor struct for the specified input tensor of a
  // given mode. This is the recommended API over the deprecated
  // GetInput/GetInputSafe to get a temp input tensor. The returned tensor shall
  // be freed via calling DeallocateTempTfLiteTensor.
  TfLiteTensor* AllocateTempInputTensor(const TfLiteNode* node, int index);

  // Returns a temporary TfLiteTensor struct for the specified output tensor of
  // a given mode. This is the recommended API over the deprecated
  // GetOutput/GetOutputSafe to get a temp output tensor. The returned tensor
  // shall be freed via calling DeallocateTempTfLiteTensor.
  TfLiteTensor* AllocateTempOutputTensor(const TfLiteNode* node, int index);

  // Returns a temporary TfLiteTensor struct for the specified intermediate
  // tensor of a given mode. This is the recommended API over the deprecated
  // GetIntermediates/GetIntermediatesSafe to get a temp intermediate tensor.
  // The returned tensor shall be freed via calling DeallocateTempTfLiteTensor.
  TfLiteTensor* AllocateTempIntermediateTensor(const TfLiteNode* node,
                                               int index);

  // Deallocates a temp TfLiteTensor.
  virtual void DeallocateTempTfLiteTensor(TfLiteTensor* tensor) = 0;

  // Returns a pointer to a temporary buffer (from the arena).
  // This API is only valid from the kernel's Prepare function and
  // the buffer's lifetime is also that of the Prepare function.
  virtual uint8_t* AllocateTempBuffer(size_t size, size_t alignment) = 0;

  // Signals that the temporary buffer is no longer needed.
  virtual void DeallocateTempBuffer(uint8_t* buffer) = 0;

  // Returns a TfLiteEvalTensor struct for a given index.
  virtual TfLiteEvalTensor* GetEvalTensor(int tensor_idx) = 0;

  // Does not take ownership of the pointer and the pointer must refer to valid
  // an object that outlive this class instance.
  // This can only be called once to set one external context.
  virtual TfLiteStatus set_external_context(void* external_context_payload) = 0;

  virtual void* external_context() = 0;

  virtual MicroGraph& graph() = 0;

#ifdef USE_TFLM_COMPRESSION

  // Available during Prepare & Eval. Returns false if tensor is not
  // compressed.
  virtual bool IsTensorCompressed(const TfLiteNode* node, int tensor_idx) = 0;

  // Only available during Prepare. The kernel is responsible for storing the
  // scratch buffer handle.
  virtual int AllocateDecompressionScratchBuffer(const TfLiteNode* node,
                                                 int tensor_idx) = 0;

  // Available during Prepare & Eval. Returns nullptr if tensor is not
  // compressed.
  virtual const CompressionTensorData* GetTensorCompressionData(
      const TfLiteNode* node, int tensor_idx) = 0;

  // Only available during Prepare & Eval. Returns nullptr on failure, otherwise
  // returns a pointer to the buffer.
  virtual void* DecompressTensorToBuffer(
      const TfLiteEvalTensor& tensor,
      const CompressionTensorData& compression_data, void* buffer);

  // Used for configuring alternate decompression memory
  struct AlternateMemoryRegion {
    void* address;
    size_t bytes;
  };

  // Set the alternate decompression memory regions.
  // Can only be called during the MicroInterpreter kInit state.
  virtual TfLiteStatus SetDecompressionMemory(
      const std::initializer_list<AlternateMemoryRegion>& regions);

  // Return a pointer to memory that can be used for decompression.
  // The pointer will be aligned to the <alignment> value.
  // Return nullptr if the requested size is not available.
  // Can be called during kPrepare and kInvoke states.
  virtual void* AllocateDecompressionMemory(size_t bytes, size_t alignment);

  // reset all allocation tracking
  virtual void ResetDecompressionMemoryAllocations();

#endif  // USE_TFLM_COMPRESSION

  // Set the alternate MicroProfilerInterface.
  // This can be used to profile subsystems simultaneously with the profiling
  // of kernels during the Eval phase.  See (b/379584353).
  // The alternate MicroProfilerInterface is currently used by the tensor
  // decompression subsystem.
  virtual TfLiteStatus SetAlternateProfiler(
      MicroProfilerInterface* alt_profiler) {
    return kTfLiteError;
  }

  // Get the alternate MicroProfilerInterface.
  // This can be used to profile subsystems simultaneously with the profiling
  // of kernels during the Eval phase.  See (b/379584353).
  // The alternate MicroProfilerInterface is currently used by the tensor
  // decompression subsystem.
  virtual MicroProfilerInterface* GetAlternateProfiler() const {
    return nullptr;
  }

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

inline MicroContext* GetMicroContext(const struct TfLiteContext* context) {
  return reinterpret_cast<MicroContext*>(context->impl_);
}

// Deprecated API. Prefer to using the MicroContext API directly from the
// kernels.
// TODO(b/213010668): migrate all existing kernels to use MicroContext, delete
// these functions, and remove corresponding members from the TfLiteContext
// struct for TFLM.
inline void* MicroContextAllocatePersistentBuffer(TfLiteContext* ctx,
                                                  size_t bytes) {
  return GetMicroContext(ctx)->AllocatePersistentBuffer(bytes);
}
inline TfLiteStatus MicroContextRequestScratchBufferInArena(TfLiteContext* ctx,
                                                            size_t bytes,
                                                            int* buffer_idx) {
  return GetMicroContext(ctx)->RequestScratchBufferInArena(bytes, buffer_idx);
}
inline void* MicroContextGetScratchBuffer(TfLiteContext* ctx, int buffer_idx) {
  return GetMicroContext(ctx)->GetScratchBuffer(buffer_idx);
}
inline TfLiteTensor* MicroContextGetTensor(const struct TfLiteContext* context,
                                           int tensor_idx) {
  return GetMicroContext(context)->AllocateTempTfLiteTensor(tensor_idx);
}
inline TfLiteEvalTensor* MicroContextGetEvalTensor(
    const struct TfLiteContext* context, int tensor_idx) {
  return GetMicroContext(context)->GetEvalTensor(tensor_idx);
}
inline TfLiteExternalContext* MicroContextGetExternalContext(
    TfLiteContext* context, TfLiteExternalContextType unused) {
  return reinterpret_cast<TfLiteExternalContext*>(
      GetMicroContext(context)->external_context());
}

// Requests that an error be reported with format string msg.
void MicroContextReportOpError(struct TfLiteContext* context,
                               const char* format, ...);

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_CONTEXT_H_
