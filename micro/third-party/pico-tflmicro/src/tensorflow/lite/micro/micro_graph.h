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

#ifndef TENSORFLOW_LITE_MICRO_MICRO_GRAPH_H_
#define TENSORFLOW_LITE_MICRO_MICRO_GRAPH_H_

#include "tensorflow/lite/micro/compatibility.h"
#include "tensorflow/lite/micro/micro_common.h"
#include "tensorflow/lite/micro/micro_resource_variable.h"

namespace tflite {

// Abstracts the details of interacting with the graph from the kernels
//
// Provides methods to invoke any subgraph in the tflite::Graph.
class MicroGraph {
 public:
  virtual ~MicroGraph() = default;

  // Calls TFLMRegistration->Invoke for every operator in a single subgraph
  // in the model.
  virtual TfLiteStatus InvokeSubgraph(int subgraph_idx) = 0;

  // Number of tensor inputs to a specified subgraph in the model.
  virtual size_t NumSubgraphInputs(int subgraph_idx) = 0;

  // Get the specified input tensor of a specified subgraph in the model.
  virtual TfLiteEvalTensor* GetSubgraphInput(int subgraph_idx,
                                             int input_idx) = 0;

  // Number of tensor outputs from a specified subgraph in the model.
  virtual size_t NumSubgraphOutputs(int subgraph_idx) = 0;

  // Get the specified output tensor of a specified subgraph in the model.
  virtual TfLiteEvalTensor* GetSubgraphOutput(int subgraph_idx,
                                              int output_idx) = 0;

  // Number of subgraphs in the model.
  virtual int NumSubgraphs() = 0;

  // Get the resource variables for this TFLM graph.
  virtual MicroResourceVariables* GetResourceVariables() = 0;

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_GRAPH_H_
