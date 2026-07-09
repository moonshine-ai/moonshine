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

#include "tensorflow/lite/micro/micro_op_resolver.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {

TfLiteStatus GetRegistrationFromOpCode(const OperatorCode* opcode,
                                       const MicroOpResolver& op_resolver,
                                       const TFLMRegistration** registration) {
  TfLiteStatus status = kTfLiteOk;
  *registration = nullptr;
  auto builtin_code = GetBuiltinCode(opcode);

  if (builtin_code > BuiltinOperator_MAX) {
    MicroPrintf("Op builtin_code out of range: %d.", builtin_code);
    status = kTfLiteError;
  } else if (builtin_code != BuiltinOperator_CUSTOM) {
    *registration = op_resolver.FindOp(builtin_code);
    if (*registration == nullptr) {
      MicroPrintf("Didn't find op for builtin opcode '%s'",
                  EnumNameBuiltinOperator(builtin_code));
      status = kTfLiteError;
    }
  } else if (!opcode->custom_code()) {
    MicroPrintf("Operator with CUSTOM builtin_code has no custom_code.\n");
    status = kTfLiteError;
  } else {
    const char* name = opcode->custom_code()->c_str();
    *registration = op_resolver.FindOp(name);
    if (*registration == nullptr) {
      // Do not report error for unresolved custom op, we do the final check
      // while preparing ops.
      status = kTfLiteError;
    }
  }
  return status;
}
}  // namespace tflite
