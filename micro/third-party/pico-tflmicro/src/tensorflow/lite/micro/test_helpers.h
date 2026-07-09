/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_MICRO_TEST_HELPERS_H_
#define TENSORFLOW_LITE_MICRO_TEST_HELPERS_H_

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "third_party/flatbuffers/include/flatbuffers/flatbuffers.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/portable_type_to_tflitetype.h"
#include "tensorflow/lite/schema/schema_generated.h"

#ifdef USE_TFLM_COMPRESSION

#include "tensorflow/lite/micro/compression.h"
#include "tensorflow/lite/micro/micro_log.h"

#endif  // TENSORFLOW_LITE_MICRO_TEST_HELPERS_H_

namespace tflite {
namespace testing {

constexpr int kOfflinePlannerHeaderSize = 3;
using TestingOpResolver = tflite::MicroMutableOpResolver<10>;

struct NodeConnection_ {
  std::initializer_list<int32_t> input;
  std::initializer_list<int32_t> output;
};
typedef struct NodeConnection_ NodeConnection;

// A simple operator that returns the median of the input with the number of
// times the kernel was invoked. The implementation below is deliberately
// complicated, just to demonstrate how kernel memory planning works.
class SimpleStatefulOp {
  static constexpr int kBufferNotAllocated = 0;
  // Inputs:
  static constexpr int kInputTensor = 0;
  // Outputs:
  static constexpr int kMedianTensor = 0;
  static constexpr int kInvokeCount = 1;
  struct OpData {
    int* invoke_count = nullptr;
    int sorting_buffer = kBufferNotAllocated;
  };

 public:
  static const TFLMRegistration* getRegistration();
  static TFLMRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);
};

class MockCustom {
 public:
  static const TFLMRegistration* getRegistration();
  static TFLMRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static void Free(TfLiteContext* context, void* buffer);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

  static bool freed_;
};

// A simple operator with the purpose of testing multiple inputs. It returns
// the sum of the inputs.
class MultipleInputs {
 public:
  static const TFLMRegistration* getRegistration();
  static TFLMRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static void Free(TfLiteContext* context, void* buffer);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

  static bool freed_;
};

// A simple no-op operator.
class NoOp {
 public:
  static const TFLMRegistration* getRegistration();
  static TFLMRegistration* GetMutableRegistration();
  static void* Init(TfLiteContext* context, const char* buffer, size_t length);
  static void Free(TfLiteContext* context, void* buffer);
  static TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);
  static TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

  static bool freed_;
};

// Returns an Op Resolver that can be used in the testing code.
TfLiteStatus GetTestingOpResolver(TestingOpResolver& op_resolver);

// Returns a simple example flatbuffer TensorFlow Lite model. Contains 1 input,
// 1 layer of weights, 1 output Tensor, and 1 operator.
const Model* GetSimpleMockModel();

#ifdef USE_TFLM_COMPRESSION

// Returns a simple example flatbuffer TensorFlow Lite model. Contains 1 input,
// 1 layer of weights, 1 output Tensor, and 1 operator (BroadcastAddOp).  The
// weights tensor is compressed.
const Model* GetSimpleMockModelCompressed();

#endif  // USE_TFLM_COMPRESSION

// Returns a flatbuffer TensorFlow Lite model with more inputs, variable
// tensors, and operators.
const Model* GetComplexMockModel();

// Returns a simple example flatbuffer TensorFlow Lite model. Contains 1 input,
// 1 layer of weights, 1 output Tensor, and 1 operator.
// The size of all three tensors is 256 x 256, which is larger than what other
// models provide from this test helper.
const Model* GetModelWith256x256Tensor();

// Returns a simple flatbuffer model with two branches.
const Model* GetSimpleModelWithBranch();

// Returns a simple example flatbuffer TensorFlow Lite model. Contains 3 inputs,
// 1 output Tensor, and 1 operator.
const Model* GetSimpleMultipleInputsModel();

// Returns a simple flatbuffer model with offline planned tensors
// @param[in]       num_tensors           Number of tensors in the model.
// @param[in]       metadata_buffer       Metadata for offline planner.
// @param[in]       node_con              List of connections, i.e. operators
//                                        in the model.
// @param[in]       num_conns             Number of connections.
// @param[in]       num_subgraph_inputs   How many of the input tensors are in
//                                        the subgraph inputs. The default value
//                                        of 0 means all of the input tensors
//                                        are in the subgraph input list. There
//                                        must be at least 1 input tensor in the
//                                        subgraph input list.
const Model* GetModelWithOfflinePlanning(int num_tensors,
                                         const int32_t* metadata_buffer,
                                         NodeConnection* node_conn,
                                         int num_conns,
                                         int num_subgraph_inputs = 0);

// Returns a flatbuffer with a single operator, two inputs (one unused) and one
// output.
const Model* GetModelWithUnusedInputs();

// Returns a flatbuffer with a single operator, zero inputs and two outputs
// (one unused).
const Model* GetModelWithUnusedOperatorOutputs();

// Returns a flatbuffer model with `simple_stateful_op`
const Model* GetSimpleStatefulModel();

// Returns a flatbuffer model with "if" and two subgraphs.
const Model* GetSimpleModelWithSubgraphsAndIf();

// Returns a flatbuffer model with "if" and two subgraphs one of which is empty.
const Model* GetSimpleModelWithIfAndEmptySubgraph();

// Returns a flatbuffer model with "while" and three subgraphs.
const Model* GetSimpleModelWithSubgraphsAndWhile();

// Returns a flatbuffer model with "if" and two subgraphs and the input tensor 1
// of "if" subgraph overlaps with the input tensor 2 of subgraph 1.
const Model* GetModelWithIfAndSubgraphInputTensorOverlap();

// Returns a flatbuffer model with null subgraph/operator inputs and outputs.
const Model* GetSimpleModelWithNullInputsAndOutputs();

// Builds a one-dimensional flatbuffer tensor of the given size.
const Tensor* Create1dFlatbufferTensor(int size, bool is_variable = false);

// Builds a one-dimensional flatbuffer tensor of the given size with
// quantization metadata.
const Tensor* CreateQuantizedFlatbufferTensor(int size);

// Creates a one-dimensional tensor with no quantization metadata.
const Tensor* CreateMissingQuantizationFlatbufferTensor(int size);

// Creates a vector of flatbuffer buffers.
const flatbuffers::Vector<flatbuffers::Offset<Buffer>>*
CreateFlatbufferBuffers();

// Performs a simple string comparison without requiring standard C library.
int TestStrcmp(const char* a, const char* b);

void PopulateContext(TfLiteTensor* tensors, int tensors_size,
                     TfLiteContext* context);

// Create a TfLiteIntArray from an array of ints.  The first element in the
// supplied array must be the size of the array expressed as an int.
TfLiteIntArray* IntArrayFromInts(const int* int_array);

// Create a TfLiteFloatArray from an array of floats.  The first element in the
// supplied array must be the size of the array expressed as a float.
TfLiteFloatArray* FloatArrayFromFloats(const float* floats);

// Assumes that `src_tensor` is a buffer where each element is a 4-bit value
// stored in 8-bit.
// Returns a new buffer that is packed densely with 2 4-bit values in a byte.
// The packing format is low-bits-first, i.e. the lower nibble of a byte is
// filled first, followed by the upper nibble.
void PackInt4ValuesDenselyInPlace(uint8_t* src_buffer, int buffer_size);

template <typename T>
TfLiteTensor CreateTensor(const T* data, TfLiteIntArray* dims,
                          const bool is_variable = false,
                          TfLiteType type = kTfLiteNoType) {
  TfLiteTensor result;
  result.dims = dims;
  result.params = {};
  result.quantization = {kTfLiteNoQuantization, nullptr};
  result.is_variable = is_variable;
  result.allocation_type = kTfLiteMemNone;
  result.data.data = const_cast<T*>(data);

  if (type == kTfLiteInt4) {
    result.type = kTfLiteInt4;
    PackInt4ValuesDenselyInPlace(tflite::GetTensorData<uint8_t>(&result),
                                 ElementCount(*dims));
    result.bytes = ((ElementCount(*dims) + 1) / 2);
  } else {
    // Const cast is used to allow passing in const and non-const arrays within
    // a single CreateTensor method. A Const array should be used for immutable
    // input tensors and non-const array should be used for mutable and output
    // tensors.
    if (type == kTfLiteNoType) {
      result.type = typeToTfLiteType<T>();
    } else {
      result.type = type;
    }

    result.bytes = ElementCount(*dims) * TfLiteTypeGetSize(result.type);
  }
  return result;
}

template <typename T>
TfLiteTensor CreateQuantizedTensor(const T* data, TfLiteIntArray* dims,
                                   const float scale, const int zero_point = 0,
                                   const bool is_variable = false,
                                   TfLiteType type = kTfLiteNoType) {
  TfLiteTensor result = CreateTensor(data, dims, is_variable, type);
  result.params = {scale, zero_point};
  result.quantization = {kTfLiteAffineQuantization, nullptr};
  return result;
}

template <typename T>
TfLiteTensor CreateQuantizedTensor(const float* input, T* quantized,
                                   TfLiteIntArray* dims, float scale,
                                   int zero_point, bool is_variable = false,
                                   TfLiteType type = kTfLiteNoType) {
  int input_size = ElementCount(*dims);
  tflite::Quantize(input, quantized, input_size, scale, zero_point);
  return CreateQuantizedTensor(quantized, dims, scale, zero_point, is_variable,
                               type);
}

template <typename T>
TfLiteTensor CreateQuantizedBiasTensor(const float* data, T* quantized,
                                       TfLiteIntArray* dims, float input_scale,
                                       float weights_scale,
                                       bool is_variable = false) {
  float bias_scale = input_scale * weights_scale;
  tflite::SymmetricQuantize(data, quantized, ElementCount(*dims), bias_scale);

  // Quantized bias tensors always have a zero point of 0, since the range of
  // values is large, and because zero point costs extra cycles during
  // processing.
  TfLiteTensor result =
      CreateQuantizedTensor(quantized, dims, bias_scale, 0, is_variable);
  return result;
}

// Creates bias tensor with input data, and per-channel weights determined by
// input scale multiplied by weight scale for each channel.  Input data will not
// be quantized.
template <typename T>
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const T* input_data, TfLiteIntArray* dims, float input_scale,
    const TfLiteFloatArray* weight_scales, TfLiteFloatArray* scales,
    TfLiteIntArray* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, bool is_variable = false,
    TfLiteType type = kTfLiteNoType) {
  int num_channels = dims->data[quantized_dimension];
  zero_points->size = num_channels;
  scales->size = num_channels;
  for (int i = 0; i < num_channels; i++) {
    scales->data[i] = input_scale * weight_scales->data[i];
    zero_points->data[i] = 0;
  }

  affine_quant->scale = scales;
  affine_quant->zero_point = zero_points;
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(input_data, dims, is_variable, type);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  return result;
}

// Quantizes bias tensor with per-channel weights determined by input
// scale multiplied by weight scale for each channel.
template <typename T>
TfLiteTensor CreatePerChannelQuantizedBiasTensor(
    const float* input, T* quantized, TfLiteIntArray* dims, float input_scale,
    const float* weight_scales, float* scales, int* zero_points,
    TfLiteAffineQuantization* affine_quant, int quantized_dimension,
    bool is_variable = false) {
  int input_size = ElementCount(*dims);
  int num_channels = dims->data[quantized_dimension];
  // First element is reserved for array length
  zero_points[0] = num_channels;
  scales[0] = static_cast<float>(num_channels);
  float* scales_array = &scales[1];
  for (int i = 0; i < num_channels; i++) {
    scales_array[i] = input_scale * weight_scales[i];
    zero_points[i + 1] = 0;
  }

  SymmetricPerChannelQuantize<T>(input, quantized, input_size, num_channels,
                                 scales_array);

  affine_quant->scale = FloatArrayFromFloats(scales);
  affine_quant->zero_point = IntArrayFromInts(zero_points);
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(quantized, dims, is_variable);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};

  return result;
}

template <typename T>
TfLiteTensor CreatePerChannelQuantizedTensor(
    const T* quantized, TfLiteIntArray* dims, TfLiteFloatArray* scales,
    TfLiteIntArray* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, bool is_variable = false,
    TfLiteType type = kTfLiteNoType) {
  affine_quant->scale = scales;
  affine_quant->zero_point = zero_points;
  affine_quant->quantized_dimension = quantized_dimension;

  TfLiteTensor result = CreateTensor(quantized, dims, is_variable, type);
  result.quantization = {kTfLiteAffineQuantization, affine_quant};
  return result;
}

TfLiteTensor CreateSymmetricPerChannelQuantizedTensor(
    const float* input, int8_t* quantized, TfLiteIntArray* dims, float* scales,
    int* zero_points, TfLiteAffineQuantization* affine_quant,
    int quantized_dimension, bool is_variable = false,
    TfLiteType tensor_weight_type = kTfLiteNoType);

// Returns the number of tensors in the default subgraph for a tflite::Model.
size_t GetModelTensorCount(const Model* model);

// Derives the asymmetric quantization scaling factor from a min and max range.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
  return (max - min) /
         static_cast<float>((std::numeric_limits<T>::max() * 1.0) -
                            std::numeric_limits<T>::min());
}

// Derives the symmetric quantization scaling factor from a min and max range.
template <typename T>
inline float SymmetricScaleFromMinMax(const float min, const float max) {
  const int32_t kScale =
      std::numeric_limits<typename std::make_signed<T>::type>::max();
  const float range = std::max(std::abs(min), std::abs(max));
  if (range == 0) {
    return 1.0f;
  } else {
    return range / kScale;
  }
}

// Derives the quantization zero point from a min and max range.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(roundf(-min / ScaleFromMinMax<T>(min, max)));
}

#ifdef USE_TFLM_COMPRESSION

template <typename T>
struct TestCompressionInfo {
  T* value_table;
  size_t value_table_stride;
  int bit_width;
  CompressionScheme scheme;
};

template <typename T>
struct TestCompressionQuantizedInfo : TestCompressionInfo<T> {
  const uint8_t* compressed;
  const float* data;
  const int* dims_data;    // TfLiteIntArray
  const float* scales;     // TfLiteFloatArray (may be computed)
  const int* zero_points;  // TfLiteIntArray (may be computed)
};

template <size_t NTENSORS, size_t NINPUTS = 2>
class TestCompressedList {
 public:
  template <typename T>
  TfLiteStatus AddInput(const TestCompressionInfo<T>& tci,
                        const TfLiteTensor& tensor, const size_t tensor_index) {
    if (next_input_index_ >= NINPUTS) {
      MicroPrintf("TestCompressedList: too many inputs, max %u", NINPUTS);
      return kTfLiteError;
    }
    inputs_comp_data_[next_input_index_].data.lut_data =
        &inputs_ltd_[next_input_index_];
    inputs_comp_data_[next_input_index_].scheme = tci.scheme;
    inputs_comp_data_[next_input_index_].data.lut_data->compressed_bit_width =
        tci.bit_width;
    inputs_comp_data_[next_input_index_].data.lut_data->value_table =
        tci.value_table;
    inputs_comp_data_[next_input_index_]
        .data.lut_data->value_table_channel_stride = tci.value_table_stride;
    inputs_comp_data_[next_input_index_]
        .data.lut_data->is_per_channel_quantized =
        IsPerChannelQuantized(tensor);
    inputs_comp_data_[next_input_index_].data.lut_data->use_alternate_axis =
        UsesAltAxis(tensor);
    return SetCompressionData(tensor_index,
                              inputs_comp_data_[next_input_index_++]);
  }

  const CompressedTensorList* GetCompressedTensorList() {
    if (next_input_index_ == 0) {
      return nullptr;
    }

    return &ctl_;
  }

 private:
  size_t next_input_index_ = 0;
  LookupTableData inputs_ltd_[NINPUTS] = {};
  CompressionTensorData inputs_comp_data_[NINPUTS] = {};
  const CompressionTensorData* ctdp_[NTENSORS] = {};
  const CompressedTensorList ctl_ = {ctdp_};

  TfLiteStatus SetCompressionData(const size_t tensor_index,
                                  const CompressionTensorData& cd) {
    if (tensor_index >= NTENSORS) {
      MicroPrintf("TestCompressedList: bad tensor index %u", tensor_index);
      return kTfLiteError;
    }
    if (cd.data.lut_data->value_table == nullptr) {
      MicroPrintf("TestCompressedList: null value_table pointer");
      return kTfLiteError;
    }
    if (cd.data.lut_data->value_table_channel_stride == 0) {
      MicroPrintf("TestCompressedList: value_table_channel_stride not set");
      return kTfLiteError;
    }
    if (cd.scheme != CompressionScheme::kBinQuant) {
      MicroPrintf("TestCompressedList: unsupported compression scheme");
      return kTfLiteError;
    }
    if (ctdp_[tensor_index] != nullptr) {
      MicroPrintf("TestCompressedList: tensor index %d already in use",
                  tensor_index);
      return kTfLiteError;
    }

    ctdp_[tensor_index] = &cd;
    return kTfLiteOk;
  }

  bool IsPerChannelQuantized(const TfLiteTensor& tensor) {
    if (tensor.quantization.type == kTfLiteAffineQuantization &&
        tensor.quantization.params != nullptr) {
      const TfLiteAffineQuantization* qp =
          static_cast<const TfLiteAffineQuantization*>(
              tensor.quantization.params);
      if (qp->scale->size > 1) {
        return true;
      }
    }

    return false;
  }

  bool UsesAltAxis(const TfLiteTensor& tensor) {
    if (tensor.quantization.type == kTfLiteAffineQuantization &&
        tensor.quantization.params != nullptr) {
      const TfLiteAffineQuantization* qp =
          static_cast<const TfLiteAffineQuantization*>(
              tensor.quantization.params);
      if (qp->quantized_dimension != 0) {
        TFLITE_DCHECK_EQ(qp->quantized_dimension, tensor.dims->size - 1);
        return true;
      }
    }

    return false;
  }
};

#endif  // USE_TFLM_COMPRESSION

}  // namespace testing
}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_TEST_HELPERS_H_
