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
#ifndef TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_
#define TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_

#include "signal/micro/kernels/irfft.h"
#include "signal/micro/kernels/rfft.h"
#include "tensorflow/lite/c/common.h"

// Forward declaration of all micro op kernel registration methods. These
// registrations are included with the standard `BuiltinOpResolver`.
//
// This header is particularly useful in cases where only a subset of ops are
// needed. In such cases, the client can selectively add only the registrations
// their model requires, using a custom `(Micro)MutableOpResolver`. Selective
// registration in turn allows the linker to strip unused kernels.

namespace tflite {

// TFLM is incrementally moving towards a flat tflite namespace
// (https://abseil.io/tips/130). Any new ops (or cleanup of existing ops should
// have their Register function declarations in the tflite namespace.

TFLMRegistration Register_ABS();
TFLMRegistration Register_ADD();
TFLMRegistration Register_ADD_N();
TFLMRegistration Register_ARG_MAX();
TFLMRegistration Register_ARG_MIN();
TFLMRegistration Register_ASSIGN_VARIABLE();
TFLMRegistration Register_AVERAGE_POOL_2D();
TFLMRegistration Register_BATCH_MATMUL();
TFLMRegistration Register_BATCH_TO_SPACE_ND();
TFLMRegistration Register_BROADCAST_ARGS();
TFLMRegistration Register_BROADCAST_TO();
TFLMRegistration Register_CALL_ONCE();
TFLMRegistration Register_CAST();
TFLMRegistration Register_CEIL();
// TODO(b/160234179): Change custom OPs to also return by value.
TFLMRegistration* Register_CIRCULAR_BUFFER();
TFLMRegistration Register_CONCATENATION();
TFLMRegistration Register_CONV_2D();
TFLMRegistration Register_COS();
TFLMRegistration Register_CUMSUM();
TFLMRegistration Register_DEPTH_TO_SPACE();
TFLMRegistration Register_DEPTHWISE_CONV_2D();
TFLMRegistration Register_DEQUANTIZE();
TFLMRegistration Register_DIV();
TFLMRegistration Register_ELU();
TFLMRegistration Register_EMBEDDING_LOOKUP();
TFLMRegistration Register_EQUAL();
TFLMRegistration* Register_ETHOSU();
TFLMRegistration Register_EXP();
TFLMRegistration Register_EXPAND_DIMS();
TFLMRegistration Register_FILL();
TFLMRegistration Register_FLOOR();
TFLMRegistration Register_FLOOR_DIV();
TFLMRegistration Register_FLOOR_MOD();
TFLMRegistration Register_FULLY_CONNECTED();
TFLMRegistration Register_GATHER();
TFLMRegistration Register_GATHER_ND();
TFLMRegistration Register_GREATER();
TFLMRegistration Register_GREATER_EQUAL();
TFLMRegistration Register_HARD_SWISH();
TFLMRegistration Register_IF();
TFLMRegistration Register_L2_NORMALIZATION();
TFLMRegistration Register_L2_POOL_2D();
TFLMRegistration Register_LEAKY_RELU();
TFLMRegistration Register_LESS();
TFLMRegistration Register_LESS_EQUAL();
TFLMRegistration Register_LOG();
TFLMRegistration Register_LOG_SOFTMAX();
TFLMRegistration Register_LOGICAL_AND();
TFLMRegistration Register_LOGICAL_NOT();
TFLMRegistration Register_LOGICAL_OR();
TFLMRegistration Register_LOGISTIC();
TFLMRegistration Register_MAX_POOL_2D();
TFLMRegistration Register_MAXIMUM();
TFLMRegistration Register_MEAN();
TFLMRegistration Register_MINIMUM();
TFLMRegistration Register_MIRROR_PAD();
TFLMRegistration Register_MUL();
TFLMRegistration Register_NEG();
TFLMRegistration Register_NOT_EQUAL();
TFLMRegistration Register_PACK();
TFLMRegistration Register_PAD();
TFLMRegistration Register_PADV2();
TFLMRegistration Register_PRELU();
TFLMRegistration Register_QUANTIZE();
TFLMRegistration Register_READ_VARIABLE();
TFLMRegistration Register_REDUCE_MAX();
TFLMRegistration Register_RELU();
TFLMRegistration Register_RELU6();
TFLMRegistration Register_RESHAPE();
TFLMRegistration Register_RESIZE_BILINEAR();
TFLMRegistration Register_RESIZE_NEAREST_NEIGHBOR();
TFLMRegistration Register_ROUND();
TFLMRegistration Register_RSQRT();
TFLMRegistration Register_SELECT_V2();
TFLMRegistration Register_SHAPE();
TFLMRegistration Register_SIN();
TFLMRegistration Register_SLICE();
TFLMRegistration Register_SOFTMAX();
TFLMRegistration Register_SPACE_TO_BATCH_ND();
TFLMRegistration Register_SPACE_TO_DEPTH();
TFLMRegistration Register_SPLIT();
TFLMRegistration Register_SPLIT_V();
TFLMRegistration Register_SQRT();
TFLMRegistration Register_SQUARE();
TFLMRegistration Register_SQUARED_DIFFERENCE();
TFLMRegistration Register_SQUEEZE();
TFLMRegistration Register_STRIDED_SLICE();
TFLMRegistration Register_SUB();
TFLMRegistration Register_SUM();
TFLMRegistration Register_SVDF();
TFLMRegistration Register_TANH();
TFLMRegistration Register_TRANSPOSE();
TFLMRegistration Register_TRANSPOSE_CONV();
// TODO(b/230666079): resolve conflict with xtensa implementation
TFLMRegistration Register_UNIDIRECTIONAL_SEQUENCE_LSTM();
TFLMRegistration Register_UNPACK();
TFLMRegistration Register_VAR_HANDLE();
TFLMRegistration Register_WHILE();
TFLMRegistration Register_ZEROS_LIKE();

// TODO(b/295174388): Add the rest of inference only registration functions.
TFLMInferenceRegistration RegisterInference_FULLY_CONNECTED();

// TODO(b/160234179): Change custom OPs to also return by value.
namespace tflm_signal {
TFLMRegistration* Register_DELAY();
TFLMRegistration* Register_FFT_AUTO_SCALE();
TFLMRegistration* Register_FILTER_BANK();
TFLMRegistration* Register_FILTER_BANK_LOG();
TFLMRegistration* Register_FILTER_BANK_SPECTRAL_SUBTRACTION();
TFLMRegistration* Register_FILTER_BANK_SQUARE_ROOT();
TFLMRegistration* Register_ENERGY();
TFLMRegistration* Register_FRAMER();
TFLMRegistration* Register_OVERLAP_ADD();
TFLMRegistration* Register_PCAN();
TFLMRegistration* Register_STACKER();
TFLMRegistration* Register_WINDOW();
}  // namespace tflm_signal

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_KERNELS_MICRO_OPS_H_
