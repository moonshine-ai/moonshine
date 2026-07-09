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
#ifndef THIRD_PARTY_TFLITE_MICRO_TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_
#define THIRD_PARTY_TFLITE_MICRO_TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_

#include "tensorflow/lite/c/common.h"

// TFLMRegistration defines the API that TFLM kernels need to implement.
// This will be replacing the current TfLiteRegistration_V1 struct with
// something more compatible Embedded enviroment TFLM is used in.
struct TFLMRegistration {
  void* (*init)(TfLiteContext* context, const char* buffer, size_t length);
  void (*free)(TfLiteContext* context, void* buffer);
  TfLiteStatus (*prepare)(TfLiteContext* context, TfLiteNode* node);
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
  void (*reset)(TfLiteContext* context, void* buffer);
  int32_t builtin_code;
  const char* custom_name;
};

struct TFLMInferenceRegistration {
  TfLiteStatus (*invoke)(TfLiteContext* context, TfLiteNode* node);
  void (*reset)(TfLiteContext* context, void* buffer);
};

#endif  // THIRD_PARTY_TFLITE_MICRO_TENSORFLOW_LITE_MICRO_MICRO_COMMON_H_
