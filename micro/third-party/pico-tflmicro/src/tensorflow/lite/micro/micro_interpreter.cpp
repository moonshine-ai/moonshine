/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/micro_interpreter.h"

#include <cstdarg>
#include <cstddef>
#include <cstdint>

#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/flatbuffer_utils.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_allocator.h"
#include "tensorflow/lite/micro/micro_interpreter_context.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {
namespace {
MemoryPlannerType FlagToMemoryPlannerType(bool preserve_all_tensors) {
  if (preserve_all_tensors) {
    return MemoryPlannerType::kLinear;
  } else {
    return MemoryPlannerType::kGreedy;
  }
}
}  // namespace

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   uint8_t* tensor_arena,
                                   size_t tensor_arena_size,
                                   MicroResourceVariables* resource_variables,
                                   MicroProfilerInterface* profiler,
                                   bool preserve_all_tensors)
    : model_(model),
      op_resolver_(op_resolver),
      allocator_(*MicroAllocator::Create(
          tensor_arena, tensor_arena_size,
          FlagToMemoryPlannerType(preserve_all_tensors))),
      graph_(&context_, model, &allocator_, resource_variables),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      input_tensors_(nullptr),
      output_tensors_(nullptr),
      micro_context_(&allocator_, model_, &graph_) {
  Init(profiler);
}

MicroInterpreter::MicroInterpreter(const Model* model,
                                   const MicroOpResolver& op_resolver,
                                   MicroAllocator* allocator,
                                   MicroResourceVariables* resource_variables,
                                   MicroProfilerInterface* profiler)
    : model_(model),
      op_resolver_(op_resolver),
      allocator_(*allocator),
      graph_(&context_, model, allocator, resource_variables),
      tensors_allocated_(false),
      initialization_status_(kTfLiteError),
      input_tensors_(nullptr),
      output_tensors_(nullptr),
      micro_context_(&allocator_, model_, &graph_) {
  Init(profiler);
}

MicroInterpreter::~MicroInterpreter() {
  if (graph_.GetAllocations() != nullptr) {
    graph_.FreeSubgraphs();
  }
}

void MicroInterpreter::Init(MicroProfilerInterface* profiler) {
  micro_context_.SetInterpreterState(
      MicroInterpreterContext::InterpreterState::kInit);
  context_.impl_ = static_cast<void*>(&micro_context_);
  context_.ReportError = MicroContextReportOpError;
  context_.GetTensor = MicroContextGetTensor;
  context_.GetEvalTensor = MicroContextGetEvalTensor;
  context_.profiler = profiler;
  context_.RequestScratchBufferInArena =
      MicroContextRequestScratchBufferInArena;
  context_.GetExternalContext = MicroContextGetExternalContext;
  context_.AllocatePersistentBuffer = MicroContextAllocatePersistentBuffer;
  context_.GetScratchBuffer = MicroContextGetScratchBuffer;

  initialization_status_ = kTfLiteOk;
}

TfLiteStatus MicroInterpreter::PrepareNodeAndRegistrationDataFromFlatbuffer() {
  for (int subgraph_idx = 0; subgraph_idx < graph_.NumSubgraphs();
       subgraph_idx++) {
    const SubGraph* subgraph = model_->subgraphs()->Get(subgraph_idx);
    TFLITE_DCHECK(subgraph != nullptr);

    auto* opcodes = model_->operator_codes();
    TfLiteBridgeBuiltinDataAllocator* builtin_data_allocator =
        allocator_.GetBuiltinDataAllocator();
    uint32_t operators_size = NumSubgraphOperators(subgraph);
    for (size_t i = 0; i < operators_size; ++i) {
      const auto* op = subgraph->operators()->Get(i);
      const size_t index = op->opcode_index();
      if (index >= opcodes->size()) {
        MicroPrintf("Missing registration for opcode_index %d\n", index);
        return kTfLiteError;
      }
      const auto* opcode = opcodes->Get(index);
      TfLiteStatus status =
          GetRegistrationFromOpCode(opcode, op_resolver_,
                                    &(graph_.GetAllocations()[subgraph_idx]
                                          .node_and_registrations[i]
                                          .registration));
      if (status != kTfLiteOk) {
        MicroPrintf("Failed to get registration from op code %s\n ",
                    EnumNameBuiltinOperator(GetBuiltinCode(opcode)));
        return status;
      }
      const auto* registration = graph_.GetAllocations()[subgraph_idx]
                                     .node_and_registrations[i]
                                     .registration;
      if (registration == nullptr) {
        MicroPrintf("Skipping op for opcode_index %d\n", index);
        return kTfLiteError;
      }
      BuiltinOperator op_type =
          static_cast<BuiltinOperator>(registration->builtin_code);

      const char* custom_data = nullptr;
      size_t custom_data_size = 0;
      unsigned char* builtin_data = nullptr;

      if (op_type == BuiltinOperator_CUSTOM) {
        // Custom Ops may or may not have a non-null custom_options field.
        if (op->custom_options() != nullptr) {
          custom_data =
              reinterpret_cast<const char*>(op->custom_options()->data());
          custom_data_size = op->custom_options()->size();
        }
      } else {
        if (op->custom_options() != nullptr) {
          MicroPrintf(
              "Unsupported behavior: found builtin operator %s with custom "
              "options.\n",
              EnumNameBuiltinOperator(op_type));
          return kTfLiteError;
        }

        TfLiteBridgeBuiltinParseFunction parser =
            op_resolver_.GetOpDataParser(op_type);
        if (parser == nullptr) {
          MicroPrintf("Did not find a parser for %s",
                      EnumNameBuiltinOperator(op_type));

          return kTfLiteError;
        }
        TF_LITE_ENSURE_STATUS(CallBuiltinParseFunction(
            parser, op, builtin_data_allocator, (void**)(&builtin_data)));
      }

      TfLiteIntArray* inputs_array =
          FlatBufferVectorToTfLiteTypeArray(op->inputs());
      TfLiteIntArray* outputs_array =
          FlatBufferVectorToTfLiteTypeArray(op->outputs());

      TfLiteNode* node = &(
          graph_.GetAllocations()[subgraph_idx].node_and_registrations[i].node);
      *node = {};
      node->inputs = inputs_array;
      node->outputs = outputs_array;
      node->builtin_data = reinterpret_cast<void*>(builtin_data);
      node->custom_initial_data = custom_data;
      node->custom_initial_data_size = custom_data_size;

      if (op->intermediates() && (op->intermediates()->size() > 0)) {
        node->intermediates =
            FlatBufferVectorToTfLiteTypeArray(op->intermediates());
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreter::AllocateTensors() {
  SubgraphAllocations* allocations = allocator_.StartModelAllocation(model_);

  if (allocations == nullptr) {
    MicroPrintf("Failed starting model allocation.\n");
    initialization_status_ = kTfLiteError;
    return kTfLiteError;
  }

  graph_.SetSubgraphAllocations(allocations);

  TF_LITE_ENSURE_STATUS(PrepareNodeAndRegistrationDataFromFlatbuffer());

  micro_context_.SetInterpreterState(
      MicroInterpreterContext::InterpreterState::kInit);
  TF_LITE_ENSURE_STATUS(graph_.InitSubgraphs());

  micro_context_.SetInterpreterState(
      MicroInterpreterContext::InterpreterState::kPrepare);

  TF_LITE_ENSURE_STATUS(graph_.PrepareSubgraphs());

  micro_context_.SetInterpreterState(
      MicroInterpreterContext::InterpreterState::kMemoryPlanning);

  TF_LITE_ENSURE_OK(&context_, allocator_.FinishModelAllocation(
                                   model_, graph_.GetAllocations(),
                                   &scratch_buffer_handles_));

  micro_context_.SetScratchBufferHandles(scratch_buffer_handles_);

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  input_tensors_ =
      reinterpret_cast<TfLiteTensor**>(allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor*) * inputs_size()));
  if (input_tensors_ == nullptr) {
    MicroPrintf(
        "Failed to allocate memory for context->input_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor*) * inputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < inputs_size(); ++i) {
    input_tensors_[i] = allocator_.AllocatePersistentTfLiteTensor(
        model_, graph_.GetAllocations(), inputs().Get(i), 0);
    if (input_tensors_[i] == nullptr) {
      MicroPrintf("Failed to initialize input tensor %d", i);
      return kTfLiteError;
    }
  }

  // TODO(b/162311891): Drop these allocations when the interpreter supports
  // handling buffers from TfLiteEvalTensor.
  output_tensors_ =
      reinterpret_cast<TfLiteTensor**>(allocator_.AllocatePersistentBuffer(
          sizeof(TfLiteTensor*) * outputs_size()));
  if (output_tensors_ == nullptr) {
    MicroPrintf(
        "Failed to allocate memory for context->output_tensors_, "
        "%d bytes required",
        sizeof(TfLiteTensor*) * outputs_size());
    return kTfLiteError;
  }

  for (size_t i = 0; i < outputs_size(); ++i) {
    output_tensors_[i] = allocator_.AllocatePersistentTfLiteTensor(
        model_, graph_.GetAllocations(), outputs().Get(i), 0);
    if (output_tensors_[i] == nullptr) {
      MicroPrintf("Failed to initialize output tensor %d", i);
      return kTfLiteError;
    }
  }

  TF_LITE_ENSURE_STATUS(Reset());

  tensors_allocated_ = true;
  micro_context_.SetInterpreterState(
      MicroInterpreterContext::InterpreterState::kInvoke);
  return kTfLiteOk;
}

TfLiteStatus MicroInterpreter::Invoke() {
  if (initialization_status_ != kTfLiteOk) {
    MicroPrintf("Invoke() called after initialization failed\n");
    return kTfLiteError;
  }

  // Ensure tensors are allocated before the interpreter is invoked to avoid
  // difficult to debug segfaults.
  if (!tensors_allocated_) {
    TF_LITE_ENSURE_OK(&context_, AllocateTensors());
  }
  return graph_.InvokeSubgraph(0);
}

TfLiteTensor* MicroInterpreter::input(size_t index) {
  const size_t length = inputs_size();
  if (index >= length) {
    MicroPrintf("Input index %d out of range (length is %d)", index, length);
    return nullptr;
  }
  return input_tensors_[index];
}

TfLiteTensor* MicroInterpreter::output(size_t index) {
  const size_t length = outputs_size();
  if (index >= length) {
    MicroPrintf("Output index %d out of range (length is %d)", index, length);
    return nullptr;
  }
  return output_tensors_[index];
}

TfLiteStatus MicroInterpreter::Reset() {
  TfLiteStatus status = graph_.ResetSubgraphs();
  if (status != kTfLiteOk) {
    return status;
  }
  return graph_.ResetVariableTensors();
}

TfLiteEvalTensor* MicroInterpreter::GetTensor(int tensor_index,
                                              int subgraph_index) {
  if (!allocator_.preserves_all_tensor()) {
    MicroPrintf("GetTensor requires all tensors to be preserved");
    return nullptr;
  }
  return &graph_.GetAllocations()[subgraph_index].tensors[tensor_index];
}

TfLiteStatus MicroInterpreter::SetMicroExternalContext(
    void* external_context_payload) {
  return micro_context_.set_external_context(external_context_payload);
}

TfLiteStatus MicroInterpreter::SetAlternateProfiler(
    MicroProfilerInterface* alt_profiler) {
  return micro_context_.SetAlternateProfiler(alt_profiler);
}

#ifdef USE_TFLM_COMPRESSION

TfLiteStatus MicroInterpreter::SetDecompressionMemory(
    const std::initializer_list<MicroInterpreterContext::AlternateMemoryRegion>&
        regions) {
  return micro_context_.SetDecompressionMemory(regions);
}

#endif  // USE_TFLM_COMPRESSION

}  // namespace tflite
