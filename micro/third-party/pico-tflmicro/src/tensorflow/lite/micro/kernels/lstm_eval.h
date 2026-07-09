/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

// Functions to perform integer evaulation for standard LSTM (e.g., defined in
// the keras lstm layer, no peephole etc.). Currently used by the 16 bits
// activation case only

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_GENERAL_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_GENERAL_H_
#include <algorithm>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/lstm_shared.h"
#include "tensorflow/lite/micro/micro_log.h"

namespace tflite {

// Interface to access all the TempTfLiteTensors of the LSTM kernel during the
// preparation phase. Can only be constructed through the constructor to avoid
// memory leakage. All TempTfLiteTensors will be deallocated through the
// destructor.
class LstmTensors {
 public:
  LstmTensors(const LstmTensors& other) = delete;
  LstmTensors& operator=(const LstmTensors& other) = delete;

  LstmTensors(TfLiteContext* context, TfLiteNode* node);
  ~LstmTensors();

  // Verify the LSTM internal tensor properties (e.g., type checks)
  // Input/output/states/fc weights tensors are required for kernel evaulation.
  // The state tensors should be variables. Variants of the standard LSTM
  // are not supported here, therefore their corresponding tensors should be
  // invalid
  TfLiteStatus ValidateTensorStatus(TfLiteContext* context) const;

  // Internal tensors. see lstm_shared.h for tensor names
  const TfLiteTensor* GetInternalTensor(const int tensor_index) const {
    return internal_tensors_[tensor_index];
  }

  const TfLiteTensor* HiddenStateTensor() const {
    return internal_tensors_[kLstmOutputStateTensor];
  }
  const TfLiteTensor* CellStateTensor() const {
    return internal_tensors_[kLstmCellStateTensor];
  }
  const TfLiteTensor* OutputTensor() const { return output_tensor_; }

 private:
  // see lstm_shared.h for tensor names
  MicroContext* micro_context_;
  TfLiteTensor* internal_tensors_[24];
  TfLiteTensor* output_tensor_;
};

// Deduce the size information (Batch (B), Time Steps (T), Input dimension (I),
// State dimension (S)) that defines the LSTM using the input and hidden state
// tensor
LstmSizeInfo CreateLstmSizeInfo(
    const bool time_major, const TfLiteIntArray* input_tensor_shape,
    const TfLiteIntArray* hidden_state_tensor_shape);

TfLiteStatus ValidateWeightTensorSize(TfLiteContext* context,
                                      const TfLiteTensor* tensor, int dim1_size,
                                      int dim2_size);

TfLiteStatus ValidateBiasTensorSize(TfLiteContext* context,
                                    const TfLiteTensor* tensor, int size);

// Go through every tensors and make sure their shape match the kernel
// configuration
TfLiteStatus ValidateTensorSize(TfLiteContext* context,
                                const LstmTensors& tensors,
                                const LstmSizeInfo& size_info);

// Wrapper function to create gate parameters for the four internal LSTM gates
TfLiteStatus CreateGateParams(
    TfLiteContext* context,
    /*Input tensors*/
    const TfLiteTensor* input, const TfLiteTensor* input_weight,
    const TfLiteTensor* input_bias,
    /*Hidden state tensors*/
    const TfLiteTensor* hidden_state, const TfLiteTensor* hidden_state_weight,
    const TfLiteTensor* hidden_state_bias,
    /*Scale of the fc output (input to non-linear activation)*/
    const float nonlinear_activation_input_scale, const TfLiteType cell_type,
    const tflite::GateParameters& gate_params);

// Create parameters for element wise multiplication that happens in a) cell
// state update ; b) hidden state update
// Note that all the output of gates are symmetrically quantized so only scales
// are required for input. However, during the hidden state update phase, the
// output is the updated hidden state, which is asymmetrically quantized. Thus
// output may require zero point
tflite::ArithmeticParams CreateInterGateMulParams(const float input1_scale,
                                                  const float input2_scale,
                                                  const float output_scale,
                                                  const TfLiteType output_type,
                                                  const int output_zp = 0);

// Create the additional information about the cell state, which include:
// cell_state_scale_power: used in integer nonlinear function (e.g., tanh)
// quantized_cell_clip: quantized cell clip range
CellStateInfo CreateLstmCellStateInfo(const float cell_state_scale,
                                      const float cell_clip);

CellStateInfo CreateLstmCellStateInfoFloat(const float cell_clip);
tflite::FullyConnectedParams CreateFCParamsFloat();

tflite::GateParameters CreateGateParamsFloat();

tflite::ArithmeticParams CreateInterGateMulParamsFloat();

TfLiteStatus PrepareGateParametersFloat(TfLiteContext* context,
                                        const LstmTensors& lstm_tensors,
                                        OpDataLSTM* op_data_lstm);

TfLiteStatus PrepareGateParametersInteger(TfLiteContext* context,
                                          const LstmTensors& lstm_tensors,
                                          OpDataLSTM* op_data_lstm);

LSTMKernelContents CreateLSTMKernelContent(TfLiteContext* context,
                                           TfLiteNode* node);

template <typename CellType>
LSTMBuffers<CellType> CreateLSTMBuffers(TfLiteContext* context,
                                        const int* buffer_indices) {
  LSTMBuffers<CellType> buffers;
  buffers.buffer0 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[0]));
  buffers.buffer1 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[1]));
  buffers.buffer2 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[2]));
  buffers.buffer3 = reinterpret_cast<CellType*>(
      context->GetScratchBuffer(context, buffer_indices[3]));
  return buffers;
}

// Since LSTM includes multiple intermediate stages, introducing the internal
// namespace to expose them for testing
namespace lstm_internal {

void Sigmoid(const RuntimeShape& data_shape, int16_t* data);

void Sigmoid(const RuntimeShape& data_shape, float* data);

void Tanh(int32_t cell_state_scale_power, const RuntimeShape& input_data_shape,
          int16_t* input_data, const RuntimeShape& output_data_shape,
          int16_t* output_data);

void Tanh(int32_t cell_state_scale_power, const RuntimeShape& input_data_shape,
          float* input_data, const RuntimeShape& output_data_shape,
          float* output_data);

void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const int16_t* input1_data, const int16_t* input2_data,
         int8_t* output_data);

void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const int16_t* input1_data, const int16_t* input2_data,
         int16_t* output_data);

void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const float* input1_data, const float* input2_data,
         float* output_data);

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const int8_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int32_t* bias_data,
                    const RuntimeShape& output_shape, int16_t* output_data);

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const int16_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int64_t* bias_data,
                    const RuntimeShape& output_shape, int16_t* output_data);

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& filter_shape, const float* filter_data,
                    const RuntimeShape& bias_shape, const float* bias_data,
                    const RuntimeShape& output_shape, float* output_data);

void AddElementWise(const int16_t* input_1, const int16_t* input_2, int n_batch,
                    int n_input, int16_t* output);

void AddElementWise(const float* input_1, const float* input_2, int n_batch,
                    int n_input, float* output);

void Clipping(const int v_size, const CellStateInfo& cell_state_info,
              int16_t* vector);

void Clipping(const int v_size, const CellStateInfo& cell_state_info,
              float* vector);

// Manages the slice position (offset), slice length (sliced tensor shape),
// and update rules for input/output/hidden state/cell state tensors at each
// time step.
class LstmStepManager {
 public:
  LstmStepManager() = delete;
  // Does not take any ownership, and all pointers must refer to valid objects
  // that outlive the one constructed.
  explicit LstmStepManager(const LstmSizeInfo* size_info)
      : size_info_(*size_info) {}

  void UpdateTime();
  void UpdateBatch();

  void ResetTime() { current_time_ = 0; }
  RuntimeShape InputShape() const;
  RuntimeShape StateShape() const;

  int InputOffset() const { return input_offset_; }
  int OutputOffset() const { return output_offset_; }
  int HiddenStateOffset() const { return hidden_state_offset_; }
  int CellStateOffset() const { return cell_state_offset_; }

 private:
  int current_time_ = 0;
  int current_batch_ = 0;
  int input_offset_ = 0;
  int output_offset_ = 0;
  int hidden_state_offset_ = 0;
  int cell_state_offset_ = 0;
  // Sizeinfo is from LstmOpData, which reside in the memory arena
  // (guarante to outlast LSTMStepManager, which reside in stack)
  const LstmSizeInfo& size_info_;
};

// Calculates a single LSTM gate.
// Implements the following formula:
//   gate = activate(FC(input) + FC(recurrent))
// Activation is sigmoid except for the "cell" gate (configurable, usually tanh)
template <typename ActivationType, typename WeightType, typename CellType,
          typename BiasType>
void CalculateLstmGate(
    const LstmStepManager& step_info, const GateParameters& gate_params,
    // Input FC
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* input_weight,
    const TfLiteEvalTensor* input_bias,
    // Recurrent FC
    const TfLiteEvalTensor* recurrent, const TfLiteEvalTensor* recurrent_weight,
    const TfLiteEvalTensor* recurrent_bias,
    // Output
    CellType* gate_output,
    // Scratch arrays
    CellType* fc_output_buffer, const TfLiteFusedActivation activation) {
  const auto gate_output_shape = step_info.StateShape();
  // Check offset validity to avoid memory overflow
  TFLITE_DCHECK_LE(step_info.InputOffset() + step_info.InputShape().FlatSize(),
                   tflite::micro::GetTensorShape(input).FlatSize());
  TFLITE_DCHECK_LE(
      step_info.HiddenStateOffset() + step_info.StateShape().FlatSize(),
      tflite::micro::GetTensorShape(recurrent).FlatSize());

  // Input FC
  FullyConnected(gate_params.input_fc_params, step_info.InputShape(),
                 tflite::micro::GetTensorData<ActivationType>(input) +
                     step_info.InputOffset(),
                 micro::GetTensorShape(input_weight),
                 tflite::micro::GetTensorData<WeightType>(input_weight),
                 tflite::micro::GetTensorShape(input_bias),
                 tflite::micro::GetOptionalTensorData<BiasType>(input_bias),
                 gate_output_shape, gate_output);

  // Recurrent FC
  FullyConnected(gate_params.recurrent_fc_params, step_info.StateShape(),
                 tflite::micro::GetTensorData<ActivationType>(recurrent) +
                     step_info.HiddenStateOffset(),
                 tflite::micro::GetTensorShape(recurrent_weight),
                 tflite::micro::GetTensorData<WeightType>(recurrent_weight),
                 tflite::micro::GetTensorShape(recurrent_bias),
                 tflite::micro::GetOptionalTensorData<BiasType>(recurrent_bias),
                 gate_output_shape, fc_output_buffer);

  AddElementWise(gate_output, fc_output_buffer,
                 /*n_batch=*/gate_output_shape.DimsData()[0],
                 /*n_state=*/gate_output_shape.DimsData()[1], gate_output);
  // Apply activation
  switch (activation) {
    case kTfLiteActSigmoid:
      Sigmoid(gate_output_shape, gate_output);
      break;
    case kTfLiteActTanh: {
      // Set the scale power to -12 to avoid shift
      Tanh(/*cell_state_scale_power=*/-12, gate_output_shape, gate_output,
           gate_output_shape, gate_output);
    } break;
    default:
      // Only Sigmoid or Tanh is used.
      TFLITE_ASSERT_FALSE;
  }
}

// Update the cell state using the output from the forget gate, input gate, and
// cell gate Formula: updated_cell_state = forget_gate_output*cell_state +
// input_gate_output * cell_gate_output, where * denotes element wise
// multiplication
template <typename CellType>
void UpdateLstmCell(const LstmStepManager& step_info,
                    TfLiteEvalTensor* cell_state,
                    // Gate outputs
                    CellType* forget_gate_output,
                    const CellType* input_gate_output,
                    const CellType* cell_gate_output,
                    // Mul parameters
                    const ArithmeticParams& forget_cell_mul_params,
                    const ArithmeticParams& input_mul_params,
                    const CellStateInfo& cell_state_info, CellType* buffer) {
  // Check offset validity to avoid memory overflow
  TFLITE_DCHECK_LE(
      step_info.CellStateOffset() + step_info.StateShape().FlatSize(),
      tflite::micro::GetTensorShape(cell_state).FlatSize());

  auto cell_state_shape = step_info.StateShape();
  // Forget Gate x Cell State
  Mul(cell_state_shape, forget_cell_mul_params, forget_gate_output,
      tflite::micro::GetTensorData<CellType>(cell_state) +
          step_info.CellStateOffset(),
      tflite::micro::GetTensorData<CellType>(cell_state) +
          step_info.CellStateOffset());
  // Input Gate x Cell Gate
  Mul(cell_state_shape, input_mul_params, input_gate_output, cell_gate_output,
      buffer);

  // Update the cell state
  AddElementWise(tflite::micro::GetTensorData<CellType>(cell_state) +
                     step_info.CellStateOffset(),
                 buffer,
                 /*n_batch=*/cell_state_shape.DimsData()[0],
                 /*n_state=*/cell_state_shape.DimsData()[1],
                 tflite::micro::GetTensorData<CellType>(cell_state) +
                     step_info.CellStateOffset());

  if (cell_state_info.cell_clip > 0) {
    Clipping(cell_state_shape.FlatSize(), cell_state_info,
             tflite::micro::GetTensorData<CellType>(cell_state) +
                 step_info.CellStateOffset());
  }
}

// Update the hidden state of the LSTM kernel using the following formula:
// updated_hidden_state = Tanh(updated_cell_state) * output_gate_output, * means
// element wise multiplication
template <typename CellType, typename ActivationType>
void UpdateLstmHidden(const LstmStepManager& step_info,
                      TfLiteEvalTensor* cell_state,
                      TfLiteEvalTensor* hidden_state,
                      const CellType* output_gate_output,
                      const ArithmeticParams& mul_params,
                      int32_t cell_state_scale_power, CellType* buffer) {
  // Check offset validity to avoid memory overflow
  TFLITE_DCHECK_LE(
      step_info.CellStateOffset() + step_info.StateShape().FlatSize(),
      tflite::micro::GetTensorShape(cell_state).FlatSize());
  TFLITE_DCHECK_LE(
      step_info.HiddenStateOffset() + step_info.StateShape().FlatSize(),
      tflite::micro::GetTensorShape(hidden_state).FlatSize());

  auto cell_state_shape = step_info.StateShape();
  CellType* cell_state_data =
      tflite::micro::GetTensorData<CellType>(cell_state) +
      step_info.CellStateOffset();
  // Tanh(cell_state)
  Tanh(cell_state_scale_power, cell_state_shape, cell_state_data,
       cell_state_shape, buffer);
  // Update the hidden state
  Mul(cell_state_shape, mul_params, buffer, output_gate_output,
      tflite::micro::GetTensorData<ActivationType>(hidden_state) +
          step_info.HiddenStateOffset());
}

template <typename ActivationType, typename WeightType, typename CellType,
          typename BiasType>
void LstmStep(const LstmStepManager& step_info, const OpDataLSTM& op_data,
              LSTMKernelContents& kernel_content,
              const LSTMBuffers<CellType>& buffers) {
  /*Step1: Calculate gate outputs to prepare cell state update*/
  CellType* gate_internal_buffer = buffers.buffer3;
  CellType* forget_gate_output = buffers.buffer0;
  CalculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
      step_info, op_data.forget_gate_parameters,
      // Input FC
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor),
      kernel_content.GetInternalTensor(tflite::kLstmInputToForgetWeightsTensor),
      kernel_content.GetInternalTensor(tflite::kLstmForgetGateBiasTensor),
      // Recurrent FC
      kernel_content.HiddenStateTensor(),
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToForgetWeightsTensor),
      /*recurrent_bias*/ nullptr,
      // Output
      forget_gate_output,
      // Scratch arrays
      gate_internal_buffer, kTfLiteActSigmoid);

  // Input Gate calculation;
  CellType* input_gate_output = buffers.buffer1;
  CalculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
      step_info, op_data.input_gate_parameters,
      // Input FC
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor),
      kernel_content.GetInternalTensor(tflite::kLstmInputToInputWeightsTensor),
      kernel_content.GetInternalTensor(tflite::kLstmInputGateBiasTensor),
      // Recurrent FC
      kernel_content.HiddenStateTensor(),
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToInputWeightsTensor),
      /*recurrent_bias*/ nullptr,
      // Output
      input_gate_output,
      // Scratch arrays
      gate_internal_buffer, kTfLiteActSigmoid);

  // Cell Gate calculation
  CellType* cell_gate_output = buffers.buffer2;
  CalculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
      step_info, op_data.cell_gate_parameters,
      // Input FC
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor),
      kernel_content.GetInternalTensor(tflite::kLstmInputToCellWeightsTensor),
      kernel_content.GetInternalTensor(tflite::kLstmCellGateBiasTensor),
      // Recurrent FC
      kernel_content.HiddenStateTensor(),
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToCellWeightsTensor),
      /*recurrent_bias*/ nullptr,
      // Output
      cell_gate_output,
      // Scratch arrays
      gate_internal_buffer, op_data.cell_gate_nonlinear_type);

  /*Step2: update the cell state */
  const InterGateParameters& inter_gate_params = op_data.inter_gate_parameters;
  CellType* updated_input_buffer = buffers.buffer1;  // reuse buffer

  UpdateLstmCell<CellType>(step_info, kernel_content.CellStateTensor(),
                           forget_gate_output, input_gate_output,
                           cell_gate_output,
                           inter_gate_params.forget_cell_mul_params,
                           inter_gate_params.input_mul_params,
                           op_data.cell_state_info, updated_input_buffer);

  /*Step3: update the hidden state */
  CellType* output_gate_output = buffers.buffer1;  // reuse buffer
  CalculateLstmGate<ActivationType, WeightType, CellType, BiasType>(
      step_info, op_data.output_gate_parameters,
      // Input FC
      kernel_content.GetInternalTensor(tflite::kLstmInputTensor),
      kernel_content.GetInternalTensor(tflite::kLstmInputToOutputWeightsTensor),
      kernel_content.GetInternalTensor(tflite::kLstmOutputGateBiasTensor),
      // Recurrent FC
      kernel_content.HiddenStateTensor(),
      kernel_content.GetInternalTensor(
          tflite::kLstmRecurrentToOutputWeightsTensor),
      /*recurrent_bias*/ nullptr,
      // Output
      output_gate_output,
      // Scratch arrays
      gate_internal_buffer, kTfLiteActSigmoid);

  CellType* tanh_activated_cell_buffer = buffers.buffer0;  // reuse buffer
  tflite::lstm_internal::UpdateLstmHidden<CellType, ActivationType>(
      step_info, kernel_content.CellStateTensor(),
      kernel_content.HiddenStateTensor(), output_gate_output,
      inter_gate_params.output_mul_params,
      op_data.cell_state_info.cell_state_scale_power,
      tanh_activated_cell_buffer);

  /*Step4: copy the update the hidden state to output*/
  // Check offset validity to avoid memory overflow
  TFLITE_DCHECK_LE(
      step_info.OutputOffset() + step_info.StateShape().FlatSize(),
      tflite::micro::GetTensorShape(kernel_content.output_tensor).FlatSize());
  // record the output (from the updated hidden state)
  ActivationType* output_ptr = tflite::micro::GetTensorData<ActivationType>(
      kernel_content.output_tensor);
  const auto* hidden_state = kernel_content.HiddenStateTensor();
  std::memcpy(output_ptr + step_info.OutputOffset(),
              tflite::micro::GetTensorData<ActivationType>(hidden_state) +
                  step_info.HiddenStateOffset(),
              step_info.StateShape().FlatSize() * sizeof(ActivationType));
}

}  // namespace lstm_internal

// Evaulate the LSTM kernel with (potential) multi-steps and multi-batch input
// Since
template <typename ActivationType, typename WeightType, typename CellType,
          typename BiasType>
TfLiteStatus EvalLstm(const OpDataLSTM& op_data,
                      LSTMKernelContents& kernel_content,
                      const LSTMBuffers<CellType>& buffers) {
  lstm_internal::LstmStepManager step_info(&op_data.size_info);
  const auto& size_info = op_data.size_info;
  // time is the first dimention, enable batch computation
  if (size_info.time_major) {
    for (int t = 0; t < size_info.time_steps; t++) {
      lstm_internal::LstmStep<ActivationType, WeightType, CellType, BiasType>(
          step_info, op_data, kernel_content, buffers);
      // prepare for the next time step
      step_info.UpdateTime();
    }
  } else {
    // batch first, unable to size the input data. single batch inference
    for (int b = 0; b < size_info.batch_size; b++) {
      for (int t = 0; t < size_info.time_steps; t++) {
        lstm_internal::LstmStep<ActivationType, WeightType, CellType, BiasType>(
            step_info, op_data, kernel_content, buffers);
        // prepare for the next time step
        step_info.UpdateTime();
      }
      // prepare for the next batch
      step_info.UpdateBatch();
      step_info.ResetTime();
    }
  }
  return kTfLiteOk;
}
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_16ACT_H_
