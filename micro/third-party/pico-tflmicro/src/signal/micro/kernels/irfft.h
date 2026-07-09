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
#ifndef SIGNAL_MICRO_KERNELS_IRFFT_H_
#define SIGNAL_MICRO_KERNELS_IRFFT_H_

#include "tensorflow/lite/micro/micro_common.h"

namespace tflite {
namespace tflm_signal {

TFLMRegistration* Register_IRFFT();
TFLMRegistration* Register_IRFFT_FLOAT();
TFLMRegistration* Register_IRFFT_INT16();
TFLMRegistration* Register_IRFFT_INT32();

}  // namespace tflm_signal
}  // namespace tflite

#endif  // SIGNAL_MICRO_KERNELS_IRFFT_H_
