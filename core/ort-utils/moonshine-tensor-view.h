#ifndef MOONSHINE_TENSOR_VIEW_H
#define MOONSHINE_TENSOR_VIEW_H

#include <stdint.h>

#include <cassert>
#include <numeric>
#include <string>
#include <vector>

#include "moonshine-tensor.h"
#include "onnxruntime_c_api.h"

#define CHECK_SHAPE_RANK(tensor, rank)                                \
  if (tensor->shape_count != rank) {                                  \
    fprintf(stderr, #tensor " shape rank is not %d at %s:%d\n", rank, \
            __FILE__, __LINE__);                                      \
    return -1;                                                        \
  }

#define CHECK_DTYPE(tensor, expected_dtype)                                \
  if (tensor->dtype != expected_dtype) {                                   \
    fprintf(stderr, #tensor " dtype is not %d at %s:%d\n", expected_dtype, \
            __FILE__, __LINE__);                                           \
    return -1;                                                             \
  }

#define TENSOR_NAME(name)                                 \
  std::string(name) + "@" + std::string(__FILE__) + ":" + \
      std::to_string(__LINE__)

struct MoonshineTensorView;

size_t moonshine_dtype_to_bytes_per_element(uint32_t moonshine_dtype);

moonshine_dtype_t ort_dtype_to_moonshine_dtype(
    ONNXTensorElementDataType ort_dtype);

ONNXTensorElementDataType moonshine_dtype_to_ort_dtype(
    uint32_t moonshine_dtype);

size_t ort_dtype_to_bytes_per_element(ONNXTensorElementDataType ort_dtype);

MoonshineTensorView *moonshine_tensor_from_token_vector(
    std::vector<int32_t> &vector);

std::vector<int32_t> token_vector_from_moonshine_tensor(
    MoonshineTensorView *moonshine_tensor);

void float16_to_float32(const uint16_t *f16_array, float *f32_array,
                        size_t count);

void log_leaked_tensor_views();

struct MoonshineTensorView {
  moonshine_tensor_t *_tensor;
  std::vector<int64_t> _shape;
  std::string name;

  MoonshineTensorView();

  MoonshineTensorView(moonshine_tensor_t *tensor, std::string name = "");

  MoonshineTensorView(const std::vector<int64_t> &shape, uint32_t dtype,
                      void *data = nullptr, const std::string &name = "");

  MoonshineTensorView(const MoonshineTensorView &other);

  MoonshineTensorView(const OrtApi *ort_api, OrtValue *ort_tensor,
                      const std::string &name = "");

  ~MoonshineTensorView();

  MoonshineTensorView &operator=(const MoonshineTensorView &other);

  // Data pointer retrieval with type checking.
  template <typename T>
  T *data() {
    return static_cast<T *>(_tensor->data);
  }

  std::vector<int64_t> &shape();

  size_t element_count();

  size_t bytes_count();

  uint32_t dtype();

  void reshape(const std::vector<int64_t> &shape);

  MoonshineTensorView cast_f16_to_f32();

  int64_t argmax();

  // You need to call ort_api->ReleaseValue(output_ort_tensor) to release this
  // OrtValue.
  OrtValue *create_ort_value(const OrtApi *ort_api, OrtMemoryInfo *memory_info);

  std::string to_string();
};

template <>
inline void *MoonshineTensorView::data<void>() {
  return static_cast<void *>(_tensor->data);
}
template <>
inline float *MoonshineTensorView::data<float>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_FLOAT32) {
    fprintf(stderr, "Tensor data type is not float\n");
    return nullptr;
  }
  return static_cast<float *>(_tensor->data);
}
template <>
inline double *MoonshineTensorView::data<double>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_FLOAT64) {
    fprintf(stderr, "Tensor data type is not double\n");
    return nullptr;
  }
  return static_cast<double *>(_tensor->data);
}
template <>
inline int8_t *MoonshineTensorView::data<int8_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_INT8) {
    fprintf(stderr, "Tensor data type is not int8_t\n");
    return nullptr;
  }
  return static_cast<int8_t *>(_tensor->data);
}
template <>
inline int16_t *MoonshineTensorView::data<int16_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_INT16) {
    fprintf(stderr, "Tensor data type is not int16_t\n");
    return nullptr;
  }
  return static_cast<int16_t *>(_tensor->data);
}
template <>
inline int32_t *MoonshineTensorView::data<int32_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_INT32) {
    fprintf(stderr, "Tensor data type is not int32_t\n");
    return nullptr;
  }
  return static_cast<int32_t *>(_tensor->data);
}
template <>
inline int64_t *MoonshineTensorView::data<int64_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_INT64) {
    fprintf(stderr, "Tensor data type is not int64_t\n");
    return nullptr;
  }
  return static_cast<int64_t *>(_tensor->data);
}
template <>
inline uint8_t *MoonshineTensorView::data<uint8_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_UINT8) {
    fprintf(stderr, "Tensor data type is not uint8_t\n");
    return nullptr;
  }
  return static_cast<uint8_t *>(_tensor->data);
}
template <>
inline uint16_t *MoonshineTensorView::data<uint16_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_UINT16 &&
      _tensor->dtype != MOONSHINE_DTYPE_FLOAT16) {
    fprintf(stderr, "Tensor data type is not uint16_t or float16\n");
    return nullptr;
  }
  return static_cast<uint16_t *>(_tensor->data);
}
template <>
inline uint32_t *MoonshineTensorView::data<uint32_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_UINT32) {
    fprintf(stderr, "Tensor data type is not uint32_t\n");
    return nullptr;
  }
  return static_cast<uint32_t *>(_tensor->data);
}
template <>
inline uint64_t *MoonshineTensorView::data<uint64_t>() {
  if (_tensor->dtype != MOONSHINE_DTYPE_UINT64) {
    fprintf(stderr, "Tensor data type is not uint64_t\n");
    return nullptr;
  }
  return static_cast<uint64_t *>(_tensor->data);
}

#endif  // MOONSHINE_TENSOR_VIEW_H