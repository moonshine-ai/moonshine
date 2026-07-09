/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_LSTM_SHARED_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_LSTM_SHARED_H_

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Input Tensors of size {n_batch, n_input}
constexpr int kLstmInputTensor = 0;

// Input weight tensors of size: {n_cell, n_input}
constexpr int kLstmInputToInputWeightsTensor = 1;  // Optional
constexpr int kLstmInputToForgetWeightsTensor = 2;
constexpr int kLstmInputToCellWeightsTensor = 3;
constexpr int kLstmInputToOutputWeightsTensor = 4;

// Recurrent weight tensors of size {n_cell, n_output}
constexpr int kLstmRecurrentToInputWeightsTensor = 5;  // Optional
constexpr int kLstmRecurrentToForgetWeightsTensor = 6;
constexpr int kLstmRecurrentToCellWeightsTensor = 7;
constexpr int kLstmRecurrentToOutputWeightsTensor = 8;

// Peephole weights tensors of size {n_cell}, representing a diagonal matrix.
constexpr int kLstmCellToInputWeightsTensor = 9;    // Optional
constexpr int kLstmCellToForgetWeightsTensor = 10;  // Optional
constexpr int kLstmCellToOutputWeightsTensor = 11;  // Optional

// Gates bias tensors of size {n_cell}
constexpr int kLstmInputGateBiasTensor = 12;  // Optional
constexpr int kLstmForgetGateBiasTensor = 13;
constexpr int kLstmCellGateBiasTensor = 14;
constexpr int kLstmOutputGateBiasTensor = 15;

// Projection weight tensor of size {n_output, n_cell}
constexpr int kLstmProjectionWeightsTensor = 16;  // Optional
// Projection bias tensor of size {n_output}
constexpr int kLstmProjectionBiasTensor = 17;  // Optional

// These state tensors are defined as variable tensors, and will be modified by
// this op.
constexpr int kLstmOutputStateTensor = 18;
constexpr int kLstmCellStateTensor = 19;

// Layer norm coefficient tensors of size {n_cell}, representing a diagonal
// matrix.
constexpr int kLstmInputLayerNormCoefficientsTensor = 20;   // Optional
constexpr int kLstmForgetLayerNormCoefficientsTensor = 21;  // Optional
constexpr int kLstmCellLayerNormCoefficientsTensor = 22;    // Optional
constexpr int kLstmOutputLayerNormCoefficientsTensor = 23;  // Optional

// Output tensors.
constexpr int kLstmOutputTensor = 0;

// Parameters for the two fully conncted computation inside each gate
struct GateParameters {
  FullyConnectedParams input_fc_params;
  FullyConnectedParams recurrent_fc_params;
};

// Paramaters for the element wise multiplications between gate outputs
struct InterGateParameters {
  ArithmeticParams forget_cell_mul_params;
  ArithmeticParams input_mul_params;
  ArithmeticParams output_mul_params;
};

// Size information about the LSTM kernel, which is deduced from tensors stored
// in the flat buffer file.
struct LstmSizeInfo {
  bool time_major;
  int batch_size;
  int time_steps;
  int input_dimension;
  int state_dimension;
};

// Contains information about the cell state tensor
struct CellStateInfo {
  float cell_clip;
  // clipping range for cell state only 16 bits cell is supported (could be
  // generalized through templatation)
  int16_t quantized_cell_clip;
  // 2^-cell_state_scale_power = cell state scale, required by integer tanh
  // computation
  int32_t cell_state_scale_power;
};

// Contains required computation information for LSTM kernel evaluation.
// Specifically, it includes shape and quantization settings for the LSTM
// internal operations. Formatted to support operations defined in the
// tensorflow/lite/kernels/internal/reference/integer_ops
// Should be constructed during the preparation phase
struct OpDataLSTM {
  LstmSizeInfo size_info;
  CellStateInfo cell_state_info;
  TfLiteFusedActivation cell_gate_nonlinear_type;
  GateParameters forget_gate_parameters;
  GateParameters input_gate_parameters;
  GateParameters cell_gate_parameters;
  GateParameters output_gate_parameters;
  InterGateParameters inter_gate_parameters;
  int buffer_indices[4];  // TFLM only
};

// Provide an interface to access the internal tensors and buffers used for LSTM
// invocation. Constructed during the invocation phase
struct LSTMKernelContents {
 public:
  // Internal tensors, fixed (const). see lstm_shared.h for tensor names
  const TfLiteEvalTensor* GetInternalTensor(const int tensor_index) const {
    return internal_tensors[tensor_index];
  }
  // Variable tensors (will be changed, can not be const)
  TfLiteEvalTensor* HiddenStateTensor() {
    return internal_tensors[kLstmOutputStateTensor];
  }
  TfLiteEvalTensor* CellStateTensor() {
    return internal_tensors[kLstmCellStateTensor];
  }
  // Node internal tensors with indexes defined at the beginning of the file
  TfLiteEvalTensor* internal_tensors[24];
  TfLiteEvalTensor* output_tensor;
};

template <typename CellType>
struct LSTMBuffers {
  // TFLM buffers requires buffer index from LstmOpData.
  CellType* buffer0;
  CellType* buffer1;
  CellType* buffer2;
  CellType* buffer3;
};

}  // namespace tflite
#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_SHARED_H_
