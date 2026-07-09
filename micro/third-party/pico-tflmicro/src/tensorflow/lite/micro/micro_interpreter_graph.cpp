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

#include "tensorflow/lite/micro/micro_interpreter_graph.h"

#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_profiler.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifdef USE_TFLM_COMPRESSION

#include "tensorflow/lite/micro/micro_context.h"

#endif  // USE_TFLM_COMPRESSION

namespace tflite {
namespace {

const char* OpNameFromRegistration(const TFLMRegistration* registration) {
  if (registration->builtin_code == BuiltinOperator_CUSTOM) {
    return registration->custom_name;
  } else {
    return EnumNameBuiltinOperator(BuiltinOperator(registration->builtin_code));
  }
}

}  // namespace

MicroInterpreterGraph::MicroInterpreterGraph(
    TfLiteContext* context, const Model* model, MicroAllocator* allocator,
    MicroResourceVariables* resource_variables)
    : context_(context),
      model_(model),
      allocator_(allocator),
      current_subgraph_index_(0),
      current_operator_index_(0),
      resource_variables_(resource_variables) {
  if (model != nullptr) {
    subgraphs_ = model->subgraphs();
  }
}

MicroInterpreterGraph::~MicroInterpreterGraph() {}

TfLiteStatus MicroInterpreterGraph::InitSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      size_t init_data_size;
      const char* init_data;
      if (registration->builtin_code == BuiltinOperator_CUSTOM) {
        init_data = reinterpret_cast<const char*>(node->custom_initial_data);
        init_data_size = node->custom_initial_data_size;
      } else {
        init_data = reinterpret_cast<const char*>(node->builtin_data);
        init_data_size = 0;
      }
      if (registration->init) {
        node->user_data =
            registration->init(context_, init_data, init_data_size);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::PrepareSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      if (registration->prepare != nullptr) {
        TfLiteStatus prepare_status = registration->prepare(context_, node);
        if (prepare_status != kTfLiteOk) {
          MicroPrintf("Node %s (number %df) failed to prepare with status %d",
                      OpNameFromRegistration(registration),
                      current_operator_index_, prepare_status);
          return kTfLiteError;
        }
#ifdef USE_TFLM_COMPRESSION
        GetMicroContext(context_)->ResetDecompressionMemoryAllocations();
#endif  // USE_TFLM_COMPRESSION
      }
      allocator_->FinishPrepareNodeAllocations(
          /*node_id=*/current_operator_index_);
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::ResetSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->reset != nullptr) {
        registration->reset(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::FreeSubgraphs() {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;

  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    current_subgraph_index_ = subgraph_idx;
    uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
    for (current_operator_index_ = 0; current_operator_index_ < operators_size;
         ++current_operator_index_) {
      TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                               .node_and_registrations[current_operator_index_]
                               .node);
      const TFLMRegistration* registration =
          subgraph_allocations_[subgraph_idx]
              .node_and_registrations[current_operator_index_]
              .registration;
      // registration is allocated outside the interpreter, so double check to
      // make sure it's not nullptr;
      if (registration != nullptr && registration->free != nullptr) {
        registration->free(context_, node->user_data);
      }
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;

  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::InvokeSubgraph(int subgraph_idx) {
  int previous_subgraph_idx = current_subgraph_index_;
  uint32_t previous_operator_idx = current_operator_index_;
  current_subgraph_index_ = subgraph_idx;

  if (static_cast<size_t>(subgraph_idx) >= subgraphs_->size()) {
    MicroPrintf("Accessing subgraph %d but only %d subgraphs found",
                subgraph_idx, subgraphs_->size());
    return kTfLiteError;
  }
  uint32_t operators_size = NumSubgraphOperators(model_, subgraph_idx);
  for (current_operator_index_ = 0; current_operator_index_ < operators_size;
       ++current_operator_index_) {
    TfLiteNode* node = &(subgraph_allocations_[subgraph_idx]
                             .node_and_registrations[current_operator_index_]
                             .node);
    const TFLMRegistration* registration =
        subgraph_allocations_[subgraph_idx]
            .node_and_registrations[current_operator_index_]
            .registration;

// This ifdef is needed (even though ScopedMicroProfiler itself is a no-op with
// -DTF_LITE_STRIP_ERROR_STRINGS) because the function OpNameFromRegistration is
// only defined for builds with the error strings.
#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
    ScopedMicroProfiler scoped_profiler(
        OpNameFromRegistration(registration),
        reinterpret_cast<MicroProfilerInterface*>(context_->profiler));
#endif

    TFLITE_DCHECK(registration->invoke);
    TfLiteStatus invoke_status = registration->invoke(context_, node);
#ifdef USE_TFLM_COMPRESSION
    GetMicroContext(context_)->ResetDecompressionMemoryAllocations();
#endif  // USE_TFLM_COMPRESSION

    // All TfLiteTensor structs used in the kernel are allocated from temp
    // memory in the allocator. This creates a chain of allocations in the
    // temp section. The call below resets the chain of allocations to
    // prepare for the next call.
    allocator_->ResetTempAllocations();

    if (invoke_status == kTfLiteError) {
      MicroPrintf("Node %s (number %d) failed to invoke with status %d",
                  OpNameFromRegistration(registration), current_operator_index_,
                  invoke_status);
      return kTfLiteError;
    } else if (invoke_status != kTfLiteOk) {
      return invoke_status;
    }
  }
  current_subgraph_index_ = previous_subgraph_idx;
  current_operator_index_ = previous_operator_idx;
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreterGraph::ResetVariableTensors() {
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs_->size();
       subgraph_idx++) {
    const SubGraph* subgraph = (*subgraphs_)[subgraph_idx];
    for (size_t i = 0; i < subgraph->tensors()->size(); ++i) {
      auto* tensor = subgraph->tensors()->Get(i);
      if (tensor->is_variable()) {
        size_t buffer_size;
        TF_LITE_ENSURE_STATUS(TfLiteEvalTensorByteLength(
            &subgraph_allocations_[subgraph_idx].tensors[i], &buffer_size));

        int value = 0;
        if (tensor->type() == tflite::TensorType_INT8) {
          value = tensor->quantization()->zero_point()->Get(0);
        }
        memset(subgraph_allocations_[subgraph_idx].tensors[i].data.raw, value,
               buffer_size);
      }
    }
  }
  if (resource_variables_ != nullptr) {
    resource_variables_->ResetAll();
  }

  return kTfLiteOk;
}

int MicroInterpreterGraph::NumSubgraphs() {
  return model_->subgraphs()->size();
}

void MicroInterpreterGraph::SetSubgraphAllocations(
    SubgraphAllocations* subgraph_allocations) {
  subgraph_allocations_ = subgraph_allocations;
}

size_t MicroInterpreterGraph::NumSubgraphInputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->inputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphInput(int subgraph_idx,
                                                          int input_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->inputs()->Get(input_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

size_t MicroInterpreterGraph::NumSubgraphOutputs(int subgraph_idx) {
  return model_->subgraphs()->Get(subgraph_idx)->outputs() == nullptr
             ? 0
             : model_->subgraphs()->Get(subgraph_idx)->outputs()->size();
}

TfLiteEvalTensor* MicroInterpreterGraph::GetSubgraphOutput(int subgraph_idx,
                                                           int output_idx) {
  int tensor_idx =
      model_->subgraphs()->Get(subgraph_idx)->outputs()->Get(output_idx);
  return &subgraph_allocations_[subgraph_idx].tensors[tensor_idx];
}

}  // namespace tflite
