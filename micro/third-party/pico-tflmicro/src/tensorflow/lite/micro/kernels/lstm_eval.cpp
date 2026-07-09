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
#include "tensorflow/lite/micro/kernels/lstm_eval.h"

#include <limits>

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/mul.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/mul.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

LstmTensors::LstmTensors(TfLiteContext* context, TfLiteNode* node) {
  micro_context_ = GetMicroContext(context);
  // 24 internal tensors. see lstm_shared.h for tensor names
  for (size_t i = 0; i < 24; i++) {
    internal_tensors_[i] = micro_context_->AllocateTempInputTensor(node, i);
  }
  output_tensor_ =
      micro_context_->AllocateTempOutputTensor(node, kLstmOutputTensor);
}

LstmTensors::~LstmTensors() {
  for (size_t i = 0; i < 24; i++) {
    if (internal_tensors_[i] != nullptr) {
      micro_context_->DeallocateTempTfLiteTensor(internal_tensors_[i]);
    }
  }
  micro_context_->DeallocateTempTfLiteTensor(output_tensor_);
}

// Verify the LSTM internal tensor properties (e.g., type checks)
// Input/output/states/fc weights tensors are required for kernel evaulation.
// The state tensors should be variables. Variants of the standard LSTM
// are not supported here, therefore their corresponding tensors should be
// invalid
TfLiteStatus LstmTensors::ValidateTensorStatus(TfLiteContext* context) const {
  // Verify certain tensor properties
  // input tensor
  TF_LITE_ENSURE(context, internal_tensors_[kLstmInputTensor] != nullptr);
  // hidden state
  TF_LITE_ENSURE(context, internal_tensors_[kLstmOutputStateTensor] != nullptr);
  TF_LITE_ENSURE(context,
                 internal_tensors_[kLstmOutputStateTensor]->is_variable);
  // hidden state becomes input so they must have the same type
  TF_LITE_ENSURE_EQ(context, internal_tensors_[kLstmOutputStateTensor]->type,
                    internal_tensors_[kLstmInputTensor]->type);
  // cell state
  TF_LITE_ENSURE(context, internal_tensors_[kLstmCellStateTensor] != nullptr);
  TF_LITE_ENSURE(context, internal_tensors_[kLstmCellStateTensor]->is_variable);
  // output
  TF_LITE_ENSURE(context, output_tensor_ != nullptr);
  // output type is the same as the input type (activations)
  TF_LITE_ENSURE_EQ(context, output_tensor_->type,
                    internal_tensors_[kLstmInputTensor]->type);

  // weight tensors (1-9, see lstm_shared for index definition)
  const auto weight_type =
      internal_tensors_[kLstmInputToForgetWeightsTensor]->type;
  for (size_t i = 1; i < 9; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    TF_LITE_ENSURE_EQ(context, internal_tensors_[i]->type, weight_type);
  }

  // bias tensors (12-15, see lstm_shared for index definition)
  const auto bias_type = internal_tensors_[kLstmForgetGateBiasTensor]->type;
  for (size_t i = 12; i < 16; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] != nullptr);
    TF_LITE_ENSURE_EQ(context, internal_tensors_[i]->type, bias_type);
  }
  // Tensors from LSTM variants are invalid
  // No peephole
  for (size_t i = 9; i < 12; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] == nullptr);
  }
  // No projection
  for (size_t i = 16; i < 18; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] == nullptr);
  }
  // No internal layer norm
  for (size_t i = 20; i < 24; i++) {
    TF_LITE_ENSURE(context, internal_tensors_[i] == nullptr);
  }
  return kTfLiteOk;
}

namespace lstm_internal {

const int32_t kInt16Max = std::numeric_limits<int16_t>::max();
const int32_t kInt16Min = std::numeric_limits<int16_t>::min();

void AddElementWise(const int16_t* input_1, const int16_t* input_2, int n_batch,
                    int n_input, int16_t* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      int32_t sum = input_1[index] + input_2[index];
      const int32_t sum_clamped = std::min(kInt16Max, std::max(kInt16Min, sum));
      output[index] = static_cast<int16_t>(sum_clamped);
    }
  }
}

void AddElementWise(const float* input_1, const float* input_2, int n_batch,
                    int n_input, float* output) {
  for (int batch = 0; batch < n_batch; ++batch) {
    for (int i = 0; i < n_input; ++i) {
      const int index = batch * n_input + i;
      output[index] = input_1[index] + input_2[index];
    }
  }
}

void Sigmoid(const RuntimeShape& data_shape, int16_t* data) {
  reference_integer_ops::Logistic(
      0 /*data->input_multiplier*/, 0 /*data->input_left_shift */,
      data_shape.FlatSize() /*NumElements(input->dims)*/,
      data /* tflite::micro::GetTensorData<int16_t>(input) */,
      data /*tflite::micro::GetTensorData<int16_t>(output) */);
}

void Sigmoid(const RuntimeShape& data_shape, float* data) {
  reference_ops::Logistic(data_shape, data, data_shape, data);
}

void Tanh(int32_t cell_state_scale_power, const RuntimeShape& input_data_shape,
          int16_t* input_data, const RuntimeShape& output_data_shape,
          int16_t* output_data) {
  int32_t tanh_input_left_shift = (15 + cell_state_scale_power) - 3;
  int32_t input_multiplier = 0;
  if (tanh_input_left_shift < 0) /* handling negative shift value */
  {
    tanh_input_left_shift = -tanh_input_left_shift;
    input_multiplier = 3;
  }
  reference_integer_ops::Tanh(input_multiplier, tanh_input_left_shift,
                              input_data_shape, input_data, output_data_shape,
                              output_data);
}

void Tanh(int32_t cell_state_scale_power, const RuntimeShape& input_data_shape,
          float* input_data, const RuntimeShape& output_data_shape,
          float* output_data) {
  reference_ops::Tanh(input_data_shape, input_data, output_data_shape,
                      output_data);
}

// Input and output have the same shape in LSTM
void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const int16_t* input1_data, const int16_t* input2_data,
         int8_t* output_data) {
  return reference_integer_ops::MulElementwise(
      shape.FlatSize(), params, input1_data, input2_data, output_data);
}

// Input and output have the same shape in LSTM
void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const int16_t* input1_data, const int16_t* input2_data,
         int16_t* output_data) {
  return reference_integer_ops::MulElementwise(
      shape.FlatSize(), params, input1_data, input2_data, output_data);
}

// Input and output have the same shape in LSTM
void Mul(const RuntimeShape& shape, const ArithmeticParams& params,
         const float* input1_data, const float* input2_data,
         float* output_data) {
  return reference_ops::Mul(params, shape, input1_data, shape, input2_data,
                            shape, output_data);
}

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const int8_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int32_t* bias_data,
                    const RuntimeShape& output_shape, int16_t* output_data) {
  return tflite::reference_integer_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const int16_t* input_data,
                    const RuntimeShape& filter_shape, const int8_t* filter_data,
                    const RuntimeShape& bias_shape, const int64_t* bias_data,
                    const RuntimeShape& output_shape, int16_t* output_data) {
  return tflite::reference_integer_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}

void FullyConnected(const FullyConnectedParams& params,
                    const RuntimeShape& input_shape, const float* input_data,
                    const RuntimeShape& filter_shape, const float* filter_data,
                    const RuntimeShape& bias_shape, const float* bias_data,
                    const RuntimeShape& output_shape, float* output_data) {
  return tflite::reference_ops::FullyConnected(
      params, input_shape, input_data, filter_shape, filter_data, bias_shape,
      bias_data, output_shape, output_data);
}

void Clipping(const int v_size, const CellStateInfo& cell_state_info,
              int16_t* vector) {
  for (int i = 0; i < v_size; i++) {
    vector[i] =
        std::max(std::min(cell_state_info.quantized_cell_clip, vector[i]),
                 static_cast<int16_t>(-cell_state_info.quantized_cell_clip));
  }
}

void Clipping(const int v_size, const CellStateInfo& cell_state_info,
              float* vector) {
  for (int i = 0; i < v_size; i++) {
    vector[i] = std::max(std::min(cell_state_info.cell_clip, vector[i]),
                         -cell_state_info.cell_clip);
  }
}

// Increment the data offset so the sigle time step invocation call can access
// the corresponding input/output tensor data at the time step
void LstmStepManager::UpdateTime() {
  current_time_ += 1;
  TFLITE_DCHECK_LE(current_time_, size_info_.time_steps);
  // default as one batch per inference
  int input_step = size_info_.input_dimension;
  int output_step = size_info_.state_dimension;
  // time major: batch inference
  if (size_info_.time_major) {
    input_step = input_step * size_info_.batch_size;
    output_step = output_step * size_info_.batch_size;
  }

  input_offset_ += input_step;
  output_offset_ += output_step;
}

// Increment the data offset so the sigle time step invocation call can access
// the corresponding hidden/cell state tensor data at the time step (for single
// batch inference only)
void LstmStepManager::UpdateBatch() {
  current_batch_ += 1;
  TFLITE_DCHECK_LE(current_batch_, size_info_.batch_size);
  // batch inference for time major: no action needed
  if (size_info_.time_major) {
    return;
  }
  // otherwise: singe batch inference, go to the next batch
  hidden_state_offset_ += size_info_.state_dimension;
  cell_state_offset_ += size_info_.state_dimension;
}

// Input shape for each single time LSTM invocation.
// Multi-batch for time_major input
RuntimeShape LstmStepManager::InputShape() const {
  int batch_size = 1;
  if (size_info_.time_major) {
    batch_size = size_info_.batch_size;
  }
  const int dims[2] = {batch_size, size_info_.input_dimension};
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims);
  return RuntimeShape(2, dims_data);
}

// State shape (both hidden and cell) for each single time LSTM invocation.
// Multi-batch for time_major input
RuntimeShape LstmStepManager::StateShape() const {
  int batch_size = 1;
  if (size_info_.time_major) {
    batch_size = size_info_.batch_size;
  }
  const int dims[2] = {batch_size, size_info_.state_dimension};
  const int32_t* dims_data = reinterpret_cast<const int32_t*>(dims);
  return RuntimeShape(2, dims_data);
}

}  // namespace lstm_internal
}  // namespace tflite
