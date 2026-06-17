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

#ifndef TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/micro/kernels/lstm_eval.h"
#include "tensorflow/lite/micro/kernels/testdata/lstm_test_data.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {

/*Helper Functions (mainly about mimicking the kernel preparation)*/

// Create fully connected parameters using quantization settings of input and
// weight tensors.
// Since TfLiteContext is not available during the kernel test, here we mimic
// (put into stack memory) CalculateOpDataFullyConnected in
// tensorflow/lite/micro/kernels/fully_connected_common.cc
template <typename CellType>
tflite::FullyConnectedParams CreateFCParams(
    const TensorQuantizationParameters& input_quant_params,
    const TensorQuantizationParameters& weight_quant_params,
    const float nonlinear_activation_input_scale) {
  OpDataFullyConnected data;
  const double input_product_scale =
      input_quant_params.scale * weight_quant_params.scale;
  double effective_scale =
      input_product_scale /
      static_cast<double>(nonlinear_activation_input_scale);

  QuantizeMultiplier(effective_scale, &data.output_multiplier,
                     &data.output_shift);

  data.input_zero_point = input_quant_params.zero_point;

  data.filter_zero_point = 0;  // symmetrically quantized
  data.output_zero_point = 0;  // symmetrically quantized

  data.output_activation_min = std::numeric_limits<CellType>::min();
  data.output_activation_max = std::numeric_limits<CellType>::max();

  return tflite::FullyConnectedParamsQuantized(data);
}

inline tflite::FullyConnectedParams CreateFCParamsFloat() {
  FullyConnectedParams op_params;
  CalculateActivationRange(kTfLiteActNone, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

// Wrapper function to create gate parameters for the four internal LSTM gates
template <typename CellType>
tflite::GateParameters CreateGateParams(
    const TensorQuantizationParameters& input_quant_params,
    const TensorQuantizationParameters& hidden_state_quant_params,
    const GateQuantizationParameters& gate_quantization_settings,
    const float nonlinear_activation_input_scale) {
  tflite::GateParameters gate_params = {};
  gate_params.input_fc_params = CreateFCParams<CellType>(
      input_quant_params, gate_quantization_settings.activation_weight,
      nonlinear_activation_input_scale);
  gate_params.recurrent_fc_params = CreateFCParams<CellType>(
      hidden_state_quant_params, gate_quantization_settings.recurrent_weight,
      nonlinear_activation_input_scale);
  return gate_params;
}

inline tflite::GateParameters CreateGateParamsFloat() {
  tflite::GateParameters gate_params = {};
  gate_params.input_fc_params = CreateFCParamsFloat();
  gate_params.recurrent_fc_params = CreateFCParamsFloat();
  return gate_params;
}
// Create parameters for element wise multiplication that happens in a) cell
// state update ; b) hidden state update
// Note that all the output of gates are symmetrically quantized so only scales
// are required for input. However, during the hidden state update phase, the
// output is the updated hidden state, which is asymmetrically quantized. Thus
// output may require zero point
template <typename OutputType>
tflite::ArithmeticParams CreateInterGateMulParams(const float input1_scale,
                                                  const float input2_scale,
                                                  const float output_scale,
                                                  const int output_zp = 0) {
  tflite::ArithmeticParams op_params = {};
  op_params.quantized_activation_min = std::numeric_limits<OutputType>::min();
  op_params.quantized_activation_max = std::numeric_limits<OutputType>::max();
  op_params.input1_offset = 0;
  op_params.input2_offset = 0;
  op_params.output_offset = output_zp;

  const double input_product_scale =
      static_cast<double>(input1_scale) * static_cast<double>(input2_scale);
  double effective_scale =
      input_product_scale / static_cast<double>(output_scale);

  QuantizeMultiplier(effective_scale, &op_params.output_multiplier,
                     &op_params.output_shift);
  return op_params;
}

inline tflite::ArithmeticParams CreateInterGateMulParamsFloat() {
  tflite::ArithmeticParams op_params = {};
  CalculateActivationRange(kTfLiteActNone, &op_params.float_activation_min,
                           &op_params.float_activation_max);
  return op_params;
}

// Create the additional information about the cell state, which include:
// cell_state_scale_power: used in integer nonlinear function (e.g., tanh)
// quantized_cell_clip: quantized cell clip range
CellStateInfo CreateLstmCellStateInfo(const float cell_state_scale,
                                      const float cell_clip) {
  CellStateInfo cell_state_info;
  // cell_state_scale_power: 2^-cell_state_scale_power = cell state scale
  int buffer;
  tflite::CheckedLog2(cell_state_scale, &buffer);
  cell_state_info.cell_state_scale_power = buffer;
  // Cell state specifics
  cell_state_info.cell_clip = cell_clip;
  cell_state_info.quantized_cell_clip = static_cast<int16_t>(
      std::min(std::max(static_cast<double>(cell_clip) /
                            static_cast<double>(cell_state_scale),
                        -32768.0),
               32767.0));
  return cell_state_info;
}

// Create LSTMKernelContents from LstmNodeContent by copying TfLiteEvalTensor
// pointers
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
LSTMKernelContents CreateLSTMKernelContent(
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  LSTMKernelContents kernel_content;
  // Point to correct tensors
  kernel_content.internal_tensors[kLstmInputTensor] =
      node_contents.GetEvalTensor(kLstmInputTensor);
  kernel_content.internal_tensors[kLstmInputToInputWeightsTensor] =
      node_contents.GetEvalTensor(kLstmInputToInputWeightsTensor);
  kernel_content.internal_tensors[kLstmInputToForgetWeightsTensor] =
      node_contents.GetEvalTensor(kLstmInputToForgetWeightsTensor);
  kernel_content.internal_tensors[kLstmInputToCellWeightsTensor] =
      node_contents.GetEvalTensor(kLstmInputToCellWeightsTensor);
  kernel_content.internal_tensors[kLstmInputToOutputWeightsTensor] =
      node_contents.GetEvalTensor(kLstmInputToOutputWeightsTensor);
  kernel_content.internal_tensors[kLstmRecurrentToInputWeightsTensor] =
      node_contents.GetEvalTensor(kLstmRecurrentToInputWeightsTensor);
  kernel_content.internal_tensors[kLstmRecurrentToForgetWeightsTensor] =
      node_contents.GetEvalTensor(kLstmRecurrentToForgetWeightsTensor);
  kernel_content.internal_tensors[kLstmRecurrentToCellWeightsTensor] =
      node_contents.GetEvalTensor(kLstmRecurrentToCellWeightsTensor);
  kernel_content.internal_tensors[kLstmRecurrentToOutputWeightsTensor] =
      node_contents.GetEvalTensor(kLstmRecurrentToOutputWeightsTensor);
  kernel_content.internal_tensors[kLstmInputGateBiasTensor] =
      node_contents.GetEvalTensor(kLstmInputGateBiasTensor);
  kernel_content.internal_tensors[kLstmForgetGateBiasTensor] =
      node_contents.GetEvalTensor(kLstmForgetGateBiasTensor);
  kernel_content.internal_tensors[kLstmCellGateBiasTensor] =
      node_contents.GetEvalTensor(kLstmCellGateBiasTensor);
  kernel_content.internal_tensors[kLstmOutputGateBiasTensor] =
      node_contents.GetEvalTensor(kLstmOutputGateBiasTensor);
  kernel_content.internal_tensors[kLstmOutputStateTensor] =
      node_contents.GetEvalTensor(kLstmOutputStateTensor);
  kernel_content.internal_tensors[kLstmOutputGateBiasTensor] =
      node_contents.GetEvalTensor(kLstmOutputGateBiasTensor);
  kernel_content.internal_tensors[kLstmCellStateTensor] =
      node_contents.GetEvalTensor(kLstmCellStateTensor);
  // Not used internal tensors
  kernel_content.internal_tensors[kLstmCellToInputWeightsTensor] = nullptr;
  kernel_content.internal_tensors[kLstmCellToForgetWeightsTensor] = nullptr;
  kernel_content.internal_tensors[kLstmCellToOutputWeightsTensor] = nullptr;
  kernel_content.internal_tensors[kLstmProjectionWeightsTensor] = nullptr;
  kernel_content.internal_tensors[kLstmProjectionBiasTensor] = nullptr;
  kernel_content.internal_tensors[kLstmInputLayerNormCoefficientsTensor] =
      nullptr;
  kernel_content.internal_tensors[kLstmForgetLayerNormCoefficientsTensor] =
      nullptr;
  kernel_content.internal_tensors[kLstmInputLayerNormCoefficientsTensor] =
      nullptr;
  kernel_content.internal_tensors[kLstmCellLayerNormCoefficientsTensor] =
      nullptr;
  kernel_content.internal_tensors[kLstmOutputLayerNormCoefficientsTensor] =
      nullptr;
  // Output tensor
  kernel_content.output_tensor = node_contents.OutputEvalTensor();
  return kernel_content;
}

// Deduce the size information (Batch (B), Time Steps (T), Input dimension (I),
// State dimension (S)) that defines the LSTM using the input and hidden state
// tensor
LstmSizeInfo CreateLstmSizeInfo(
    const bool time_major, const TfLiteIntArray* input_tensor_shape,
    const TfLiteIntArray* hidden_state_tensor_shape) {
  LstmSizeInfo size_info;
  size_info.time_major = time_major;
  size_info.batch_size =
      time_major ? input_tensor_shape->data[1] : input_tensor_shape->data[0];
  size_info.time_steps =
      time_major ? input_tensor_shape->data[0] : input_tensor_shape->data[1];
  size_info.input_dimension = input_tensor_shape->data[2];
  size_info.state_dimension = hidden_state_tensor_shape->data[1];
  return size_info;
}

// Create the LstmOpData using the LstmNodeContent and
// NodeQuantizationParameters (defined in test_data/lstm_test_data) During the
// actual inference phase, OpDataLSTM is created using information from the
// flatbuffer file. The test divide the complete LSTM node information into
// LstmNodeContent and NodeQuantizationParameters for easy construction
// purposes
template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
OpDataLSTM CreateLstmOpData(
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  const auto& builtin_data = node_contents.BuiltinData();
  const auto& quantization_settings = node_contents.QuantizationSettings();
  OpDataLSTM op_data;

  op_data.cell_gate_nonlinear_type = builtin_data.activation;
  op_data.size_info =
      CreateLstmSizeInfo(builtin_data.time_major,
                         node_contents.GetEvalTensor(kLstmInputTensor)->dims,
                         node_contents.HiddenStateEvalTensor()->dims);

  op_data.cell_state_info = CreateLstmCellStateInfo(
      quantization_settings.cell_state.scale, builtin_data.cell_clip);

  // Gate Parameters
  op_data.forget_gate_parameters = CreateGateParams<CellType>(
      quantization_settings.input, quantization_settings.hidden_state,
      quantization_settings.forget_gate,
      quantization_settings.nonlinear_activation_input_scale);
  op_data.input_gate_parameters = CreateGateParams<CellType>(
      quantization_settings.input, quantization_settings.hidden_state,
      quantization_settings.input_gate,
      quantization_settings.nonlinear_activation_input_scale);
  op_data.cell_gate_parameters = CreateGateParams<CellType>(
      quantization_settings.input, quantization_settings.hidden_state,
      quantization_settings.cell_gate,
      quantization_settings.nonlinear_activation_input_scale);
  op_data.output_gate_parameters = CreateGateParams<CellType>(
      quantization_settings.input, quantization_settings.hidden_state,
      quantization_settings.output_gate,
      quantization_settings.nonlinear_activation_input_scale);
  // Inter gate multiplication parameters
  op_data.inter_gate_parameters.forget_cell_mul_params =
      CreateInterGateMulParams<CellType>(
          quantization_settings.nonlinear_activation_output_scale,
          quantization_settings.cell_state.scale,
          quantization_settings.cell_state.scale);
  op_data.inter_gate_parameters.input_mul_params =
      CreateInterGateMulParams<CellType>(
          quantization_settings.nonlinear_activation_output_scale,
          quantization_settings.nonlinear_activation_output_scale,
          quantization_settings.cell_state.scale);
  op_data.inter_gate_parameters.output_mul_params =
      CreateInterGateMulParams<ActivationType>(
          quantization_settings.nonlinear_activation_output_scale,
          quantization_settings.nonlinear_activation_output_scale,
          quantization_settings.hidden_state.scale,
          quantization_settings.hidden_state.zero_point);
  return op_data;
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
OpDataLSTM CreateLstmOpDataFloat(
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_contents) {
  const auto& builtin_data = node_contents.BuiltinData();
  OpDataLSTM op_data;

  op_data.cell_gate_nonlinear_type = builtin_data.activation;
  op_data.size_info =
      CreateLstmSizeInfo(builtin_data.time_major,
                         node_contents.GetEvalTensor(kLstmInputTensor)->dims,
                         node_contents.HiddenStateEvalTensor()->dims);
  op_data.cell_state_info.cell_clip = builtin_data.cell_clip;
  op_data.cell_state_info.quantized_cell_clip = 0;     // No quantization
  op_data.cell_state_info.cell_state_scale_power = 0;  // No quantization

  // Gate Parameters
  op_data.forget_gate_parameters = CreateGateParamsFloat();
  op_data.input_gate_parameters = CreateGateParamsFloat();
  op_data.cell_gate_parameters = CreateGateParamsFloat();
  op_data.output_gate_parameters = CreateGateParamsFloat();
  // Inter gate multiplication parameters
  op_data.inter_gate_parameters.forget_cell_mul_params =
      CreateInterGateMulParamsFloat();
  op_data.inter_gate_parameters.input_mul_params =
      CreateInterGateMulParamsFloat();
  op_data.inter_gate_parameters.output_mul_params =
      CreateInterGateMulParamsFloat();
  return op_data;
}

/*Test Functions Below Here*/
template <typename T>
void ValidateResultGoldens(const T* golden, const T* output_data,
                           const int output_len, const float tolerance) {
  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output_data[i], tolerance);
  }
}

template <int batch_size, int state_dimension>
void TestCalculateLstmGateFloat(const TfLiteEvalTensor* input,
                                const TfLiteEvalTensor* input_weight,
                                const TfLiteEvalTensor* input_bias,
                                // Recurrent FC
                                const TfLiteEvalTensor* recurrent,
                                const TfLiteEvalTensor* recurrent_weight,
                                const TfLiteEvalTensor* recurrent_bias,
                                // Result comparison
                                TfLiteFusedActivation nonlinear_type,
                                const float* expected_vals, float tolerance) {
  float gate_output[batch_size * state_dimension] = {};
  float fc_output_buffer[batch_size * state_dimension] = {};

  tflite::GateParameters gate_params = CreateGateParamsFloat();

  // Create step information: only one time step, no need to update
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ false, input->dims, recurrent->dims);
  // revise time_major = true to enable batch inference
  size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&size_info);

  tflite::lstm_internal::CalculateLstmGate<float, float, float, float>(
      step_info, gate_params,
      // Input FC
      input, input_weight, input_bias,
      // Recurrent FC
      recurrent, recurrent_weight, recurrent_bias,
      // Output
      gate_output,
      // Scratch arrays
      fc_output_buffer, nonlinear_type);

  ValidateResultGoldens(expected_vals, gate_output,
                        batch_size * state_dimension, tolerance);
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int state_dimension>
void TestCalculateLstmGateInteger(
    const TfLiteEvalTensor* input, const TfLiteEvalTensor* input_weight,
    const TfLiteEvalTensor* input_bias,
    // Recurrent FC
    const TfLiteEvalTensor* recurrent, const TfLiteEvalTensor* recurrent_weight,
    const TfLiteEvalTensor* recurrent_bias,
    // Quantization settings
    const NodeQuantizationParameters& node_quantization_settings,
    const GateQuantizationParameters& gate_quantization_settings,
    // Result comparison
    TfLiteFusedActivation nonlinear_type, const float* expected_vals,
    float tolerance) {
  CellType gate_output[batch_size * state_dimension] = {};
  CellType fc_output_buffer[batch_size * state_dimension] = {};

  tflite::GateParameters gate_params = CreateGateParams<CellType>(
      node_quantization_settings.input, node_quantization_settings.hidden_state,
      gate_quantization_settings,
      node_quantization_settings.nonlinear_activation_input_scale);

  // Create step information: only one time step, no need to update
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ false, input->dims, recurrent->dims);
  // revise time_major = true to enable batch inference
  size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&size_info);

  // only int8 weight is supported now
  tflite::lstm_internal::CalculateLstmGate<ActivationType, WeightType, CellType,
                                           BiasType>(
      step_info, gate_params,
      // Input FC
      input, input_weight, input_bias,
      // Recurrent FC
      recurrent, recurrent_weight, recurrent_bias,
      // Output
      gate_output,
      // Scratch arrays
      fc_output_buffer, nonlinear_type);

  float gate_output_float[batch_size * state_dimension] = {};
  Dequantize(gate_output, batch_size * state_dimension,
             node_quantization_settings.nonlinear_activation_output_scale, 0,
             gate_output_float);

  ValidateResultGoldens(expected_vals, gate_output_float,
                        batch_size * state_dimension, tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestUpdateLstmCellFloat(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_content,
    const float tolerance) {
  float buffer[batch_size * state_dimension] = {};

  auto forget_cell_mul_params = CreateInterGateMulParamsFloat();
  auto input_mul_params = CreateInterGateMulParamsFloat();

  auto cell_state = node_content.CellStateEvalTensor();
  // Create step information: only one time step, no need to update
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ false,
      node_content.GetEvalTensor(tflite::kLstmInputTensor)->dims,
      node_content.HiddenStateEvalTensor()->dims);
  // revise time_major = true to enable batch inference
  size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&size_info);

  // copy the data since it will be updated
  float forget_gate[batch_size * state_dimension] = {};
  std::memcpy(forget_gate, gate_output_data.expected_forget_gate_output,
              batch_size * state_dimension * sizeof(float));

  CellStateInfo cell_state_info;
  cell_state_info.cell_clip = node_content.BuiltinData().cell_clip;
  // Call the function to be tested
  tflite::lstm_internal::UpdateLstmCell<float>(
      step_info, cell_state, forget_gate,
      gate_output_data.expected_input_gate_output,
      gate_output_data.expected_cell_gate_output, forget_cell_mul_params,
      input_mul_params, cell_state_info, buffer);

  ValidateResultGoldens(gate_output_data.expected_updated_cell,
                        tflite::micro::GetTensorData<float>(cell_state),
                        batch_size * state_dimension, tolerance);
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUpdateLstmCellInteger(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>& node_content,
    const float tolerance) {
  const auto& quantization_settings = node_content.QuantizationSettings();
  CellType quantized_forget_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_forget_gate_output,
                   quantized_forget_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  CellType quantized_input_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_input_gate_output,
                   quantized_input_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  CellType quantized_cell_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_cell_gate_output,
                   quantized_cell_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  CellType buffer[batch_size * state_dimension] = {};

  auto forget_cell_mul_params = CreateInterGateMulParams<CellType>(
      quantization_settings.nonlinear_activation_output_scale,
      quantization_settings.cell_state.scale,
      quantization_settings.cell_state.scale);
  auto input_mul_params = CreateInterGateMulParams<CellType>(
      quantization_settings.nonlinear_activation_output_scale,
      quantization_settings.nonlinear_activation_output_scale,
      quantization_settings.cell_state.scale);

  auto cell_state_info =
      CreateLstmCellStateInfo(quantization_settings.cell_state.scale,
                              node_content.BuiltinData().cell_clip);

  auto cell_state = node_content.CellStateEvalTensor();
  // Create step information: only one time step, no need to update
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ false,
      node_content.GetEvalTensor(tflite::kLstmInputTensor)->dims,
      node_content.HiddenStateEvalTensor()->dims);
  // revise time_major = true to enable batch inference
  size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&size_info);

  // Call the function to be tested
  tflite::lstm_internal::UpdateLstmCell<CellType>(
      step_info, cell_state, quantized_forget_gate, quantized_input_gate,
      quantized_cell_gate, forget_cell_mul_params, input_mul_params,
      cell_state_info, buffer);

  float cell_state_float[batch_size * state_dimension] = {};
  Dequantize(tflite::micro::GetTensorData<CellType>(cell_state),
             batch_size * state_dimension,
             quantization_settings.cell_state.scale,
             quantization_settings.cell_state.zero_point, cell_state_float);

  ValidateResultGoldens(gate_output_data.expected_updated_cell,
                        cell_state_float, batch_size * state_dimension,
                        tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestUpdateLstmHiddenFloat(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_content,
    const float tolerance) {
  float buffer[batch_size * state_dimension] = {};

  auto mul_params = CreateInterGateMulParamsFloat();

  int32_t cell_state_scale_power = 0;

  // Create step information: only one time step, no need to update
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ false,
      node_content.GetEvalTensor(tflite::kLstmInputTensor)->dims,
      node_content.HiddenStateEvalTensor()->dims);
  // revise time_major = true to enable batch inference
  size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&size_info);

  auto cell_state = node_content.CellStateEvalTensor();
  auto hidden_state = node_content.HiddenStateEvalTensor();

  tflite::lstm_internal::UpdateLstmHidden<float, float>(
      step_info, cell_state, hidden_state,
      gate_output_data.expected_output_gate_output, mul_params,
      cell_state_scale_power, buffer);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden,
                        tflite::micro::GetTensorData<float>(hidden_state),
                        batch_size * state_dimension, tolerance);
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestUpdateLstmHiddenInteger(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>& node_content,
    const float tolerance) {
  const auto& quantization_settings = node_content.QuantizationSettings();
  CellType quantized_output_gate[batch_size * state_dimension] = {};
  tflite::Quantize(gate_output_data.expected_output_gate_output,
                   quantized_output_gate, batch_size * state_dimension,
                   quantization_settings.nonlinear_activation_output_scale, 0);

  CellType buffer[batch_size * state_dimension] = {};

  auto mul_params = CreateInterGateMulParams<ActivationType>(
      quantization_settings.nonlinear_activation_output_scale,
      quantization_settings.nonlinear_activation_output_scale,
      quantization_settings.hidden_state.scale,
      quantization_settings.hidden_state.zero_point);

  int cell_state_scale_power_buffer;
  tflite::CheckedLog2(quantization_settings.cell_state.scale,
                      &cell_state_scale_power_buffer);
  int32_t cell_state_scale_power = cell_state_scale_power_buffer;

  // Create step information: only one time step, no need to update
  auto size_info = tflite::testing::CreateLstmSizeInfo(
      /*time_major*/ false,
      node_content.GetEvalTensor(tflite::kLstmInputTensor)->dims,
      node_content.HiddenStateEvalTensor()->dims);
  // revise time_major = true to enable batch inference
  size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&size_info);

  auto cell_state = node_content.CellStateEvalTensor();
  auto hidden_state = node_content.HiddenStateEvalTensor();

  tflite::lstm_internal::UpdateLstmHidden<CellType, ActivationType>(
      step_info, cell_state, hidden_state, quantized_output_gate, mul_params,
      cell_state_scale_power, buffer);

  float hidden_state_float[batch_size * state_dimension] = {};
  Dequantize(tflite::micro::GetTensorData<ActivationType>(hidden_state),
             batch_size * state_dimension,
             quantization_settings.hidden_state.scale,
             quantization_settings.hidden_state.zero_point, hidden_state_float);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden,
                        hidden_state_float, batch_size * state_dimension,
                        tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestLstmStepFloat(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    /*can not be const, state will be updated*/
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_contents) {
  // Mimicking the kernel preparation phase, node_contents approximate the
  LSTMKernelContents kernel_content = CreateLSTMKernelContent(node_contents);
  LSTMBuffers<float> buffers;
  // Scratch buffers on the stack
  float buffer0[batch_size * state_dimension] = {};
  buffers.buffer0 = buffer0;
  float buffer1[batch_size * state_dimension] = {};
  buffers.buffer1 = buffer1;
  float buffer2[batch_size * state_dimension] = {};
  buffers.buffer2 = buffer2;
  float buffer3[batch_size * state_dimension] = {};
  buffers.buffer3 = buffer3;

  OpDataLSTM op_data = CreateLstmOpDataFloat(node_contents);
  // set time_major to true to test batch inference
  op_data.size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&op_data.size_info);
  tflite::lstm_internal::LstmStep<float, float, float, float>(
      step_info, op_data, kernel_content, buffers);

  ValidateResultGoldens(
      gate_output_data.expected_updated_hidden,
      tflite::micro::GetTensorData<float>(kernel_content.HiddenStateTensor()),
      batch_size * state_dimension, hidden_state_tolerance);
  ValidateResultGoldens(
      gate_output_data.expected_updated_cell,
      tflite::micro::GetTensorData<float>(kernel_content.CellStateTensor()),
      batch_size * state_dimension, cell_state_tolerance);
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestLstmStepInteger(
    const GateOutputCheckData<batch_size * input_dimension,
                              batch_size * state_dimension>& gate_output_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    /*can not be const, state will be updated*/
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  // Mimicking the kernel preparation phase, node_contents approximate the
  LSTMKernelContents kernel_content = CreateLSTMKernelContent(node_contents);
  LSTMBuffers<CellType> buffers;

  // Scratch buffers on the stack
  CellType buffer0[batch_size * state_dimension] = {};
  buffers.buffer0 = buffer0;
  CellType buffer1[batch_size * state_dimension] = {};
  buffers.buffer1 = buffer1;
  CellType buffer2[batch_size * state_dimension] = {};
  buffers.buffer2 = buffer2;
  CellType buffer3[batch_size * state_dimension] = {};
  buffers.buffer3 = buffer3;

  OpDataLSTM op_data = CreateLstmOpData(node_contents);
  // set time_major to true to test batch inference
  op_data.size_info.time_major = true;
  tflite::lstm_internal::LstmStepManager step_info(&op_data.size_info);
  tflite::lstm_internal::LstmStep<ActivationType, WeightType, CellType,
                                  BiasType>(step_info, op_data, kernel_content,
                                            buffers);

  const auto& quantization_settings = node_contents.QuantizationSettings();
  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(
      tflite::micro::GetTensorData<ActivationType>(
          kernel_content.HiddenStateTensor()),
      batch_size * state_dimension, quantization_settings.hidden_state.scale,
      quantization_settings.hidden_state.zero_point, dequantized_hidden_state);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(
      tflite::micro::GetTensorData<CellType>(kernel_content.CellStateTensor()),
      batch_size * state_dimension, quantization_settings.cell_state.scale,
      quantization_settings.cell_state.zero_point, dequantized_cell_state);

  ValidateResultGoldens(gate_output_data.expected_updated_hidden,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);
  ValidateResultGoldens(gate_output_data.expected_updated_cell,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);
}

template <int batch_size, int time_steps, int input_dimension,
          int state_dimension>
void TestEvalLstmFloat(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<float, float, float, float, batch_size, time_steps,
                    input_dimension, state_dimension>& node_contents) {
  // Mimicking the kernel preparation phase, node_contents approximate the node
  LSTMKernelContents kernel_content = CreateLSTMKernelContent(node_contents);
  // Scratch buffers on the stack
  LSTMBuffers<float> buffers;
  float buffer0[batch_size * state_dimension] = {};
  buffers.buffer0 = buffer0;
  float buffer1[batch_size * state_dimension] = {};
  buffers.buffer1 = buffer1;
  float buffer2[batch_size * state_dimension] = {};
  buffers.buffer2 = buffer2;
  float buffer3[batch_size * state_dimension] = {};
  buffers.buffer3 = buffer3;

  OpDataLSTM op_data = CreateLstmOpDataFloat(node_contents);

  tflite::EvalLstm<float, float, float, float>(op_data, kernel_content,
                                               buffers);

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        node_contents.GetHiddenStateData(),
                        batch_size * state_dimension, hidden_state_tolerance);

  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        node_contents.GetCellStateData(),
                        batch_size * state_dimension, cell_state_tolerance);

  ValidateResultGoldens(eval_check_data.expected_output,
                        node_contents.GetOutputData(),
                        batch_size * state_dimension, hidden_state_tolerance);
}

template <typename ActivationType, typename WeightType, typename BiasType,
          typename CellType, int batch_size, int time_steps,
          int input_dimension, int state_dimension>
void TestEvalLstmInteger(
    const LstmEvalCheckData<
        batch_size * time_steps * input_dimension, batch_size * state_dimension,
        batch_size * state_dimension * time_steps>& eval_check_data,
    const float hidden_state_tolerance, const float cell_state_tolerance,
    LstmNodeContent<ActivationType, WeightType, BiasType, CellType, batch_size,
                    time_steps, input_dimension, state_dimension>&
        node_contents) {
  // Mimicking the kernel preparation phase, node_contents approximate the node
  LSTMKernelContents kernel_content = CreateLSTMKernelContent(node_contents);
  // Scratch buffers on the stack
  LSTMBuffers<CellType> buffers;
  CellType buffer0[batch_size * state_dimension] = {};
  buffers.buffer0 = buffer0;
  CellType buffer1[batch_size * state_dimension] = {};
  buffers.buffer1 = buffer1;
  CellType buffer2[batch_size * state_dimension] = {};
  buffers.buffer2 = buffer2;
  CellType buffer3[batch_size * state_dimension] = {};
  buffers.buffer3 = buffer3;

  OpDataLSTM op_data = CreateLstmOpData(node_contents);

  tflite::EvalLstm<ActivationType, WeightType, CellType, BiasType>(
      op_data, kernel_content, buffers);

  const auto& quantization_settings = node_contents.QuantizationSettings();
  float dequantized_hidden_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetHiddenStateData(), batch_size * state_dimension,
             quantization_settings.hidden_state.scale,
             quantization_settings.hidden_state.zero_point,
             dequantized_hidden_state);

  ValidateResultGoldens(eval_check_data.expected_hidden_state,
                        dequantized_hidden_state, batch_size * state_dimension,
                        hidden_state_tolerance);

  float dequantized_cell_state[batch_size * state_dimension] = {};
  Dequantize(node_contents.GetCellStateData(), batch_size * state_dimension,
             quantization_settings.cell_state.scale,
             quantization_settings.cell_state.zero_point,
             dequantized_cell_state);
  ValidateResultGoldens(eval_check_data.expected_cell_state,
                        dequantized_cell_state, batch_size * state_dimension,
                        cell_state_tolerance);

  float dequantized_output[batch_size * state_dimension * time_steps] = {};
  Dequantize(node_contents.GetOutputData(),
             batch_size * state_dimension * time_steps,
             quantization_settings.output.scale,
             quantization_settings.output.zero_point, dequantized_output);
  ValidateResultGoldens(eval_check_data.expected_output, dequantized_output,
                        batch_size * state_dimension, hidden_state_tolerance);
}

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_LSTM_EVAL_TEST_H_
