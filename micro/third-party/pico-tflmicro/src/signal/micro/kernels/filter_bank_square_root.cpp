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

#include "signal/src/filter_bank_square_root.h"

#include <stdint.h>

#include "signal/micro/kernels/filter_bank_square_root.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/memory_helpers.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kScaleBitsTensor = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus FilterBankSquareRootEval(TfLiteContext* context,
                                      TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* scale_bits =
      tflite::micro::GetEvalInput(context, node, kScaleBitsTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  const uint64_t* input_data = tflite::micro::GetTensorData<uint64_t>(input);
  const int32_t* scale_bits_data =
      tflite::micro::GetTensorData<int32_t>(scale_bits);
  uint32_t* output_data = tflite::micro::GetTensorData<uint32_t>(output);
  int32_t num_channels = input->dims->data[0];
  tflm_signal::FilterbankSqrt(input_data, num_channels, *scale_bits_data,
                              output_data);
  return kTfLiteOk;
}

}  // namespace

namespace tflm_signal {

TFLMRegistration* Register_FILTER_BANK_SQUARE_ROOT() {
  static TFLMRegistration r = tflite::micro::RegisterOp(
      nullptr, FilterBankSquareRootPrepare, FilterBankSquareRootEval);
  return &r;
}

}  // namespace tflm_signal

}  // namespace tflite
