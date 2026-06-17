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

#include "signal/micro/kernels/fft_auto_scale_kernel.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "signal/src/fft_auto_scale.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_context.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;
constexpr int kScaleBitTensor = 1;

TfLiteStatus FftAutoScaleEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TfLiteEvalTensor* scale_bit =
      tflite::micro::GetEvalOutput(context, node, kScaleBitTensor);

  const int16_t* input_data = tflite::micro::GetTensorData<int16_t>(input);
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  int32_t* scale_bit_data = tflite::micro::GetTensorData<int32_t>(scale_bit);

  *scale_bit_data =
      tflm_signal::FftAutoScale(input_data, output->dims->data[0], output_data);
  return kTfLiteOk;
}

}  // namespace

// TODO(b/286250473): remove namespace once de-duped libraries
namespace tflm_signal {

TFLMRegistration* Register_FFT_AUTO_SCALE() {
  static TFLMRegistration r =
      tflite::micro::RegisterOp(nullptr, FftAutoScalePrepare, FftAutoScaleEval);
  return &r;
}

}  // namespace tflm_signal
}  // namespace tflite
