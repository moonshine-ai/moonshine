#include "moonshine-tensor-view.h"

#include <cstdlib>
#include <cstring>
#include <list>

#include "debug-utils.h"
#include "ort-utils.h"

namespace {

moonshine_tensor_t *moonshine_tensor_from_shape_and_dtype(
    const std::vector<int64_t> &shape, uint32_t dtype,
    const void *data_to_copy) {
  if (shape.size() == 0) {
    fprintf(stderr, "Shape is empty\n");
    return nullptr;
  }
  moonshine_tensor_t *moonshine_tensor = static_cast<moonshine_tensor_t *>(
      DEBUG_CALLOC(1, sizeof(moonshine_tensor_t)));
  moonshine_tensor->dtype = dtype;
  moonshine_tensor->shape =
      static_cast<int64_t *>(DEBUG_CALLOC(shape.size(), sizeof(int64_t)));
  std::memcpy(moonshine_tensor->shape, shape.data(),
              shape.size() * sizeof(int64_t));
  moonshine_tensor->shape_count = shape.size();
  const size_t bytes_per_element = moonshine_dtype_to_bytes_per_element(dtype);
  const size_t element_count = std::accumulate(shape.begin(), shape.end(), 1,
                                               std::multiplies<int64_t>());
  const size_t data_size_in_bytes = element_count * bytes_per_element;
  moonshine_tensor->data =
      static_cast<uint8_t *>(DEBUG_CALLOC(data_size_in_bytes, 1));
  if (data_to_copy != nullptr) {
    std::memcpy(moonshine_tensor->data, data_to_copy, data_size_in_bytes);
  }
  return moonshine_tensor;
}

moonshine_tensor_t *moonshine_tensor_from_ort_tensor(const OrtApi *ort_api,
                                                     OrtValue *ort_tensor) {
  std::vector<int64_t> shape = ort_get_value_shape(ort_api, ort_tensor);
  if (shape.size() == 0) {
    fprintf(stderr, "Shape is empty\n");
    return nullptr;
  }
  ONNXTensorElementDataType ort_dtype = ort_get_value_type(ort_api, ort_tensor);
  moonshine_dtype_t moonshine_dtype = ort_dtype_to_moonshine_dtype(ort_dtype);
  void *ort_data = nullptr;
  LOG_ORT_ERROR(ort_api, ort_api->GetTensorMutableData(ort_tensor, &ort_data));
  moonshine_tensor_t *moonshine_tensor =
      moonshine_tensor_from_shape_and_dtype(shape, moonshine_dtype, ort_data);
  return moonshine_tensor;
}
}  // namespace

MoonshineTensorView::MoonshineTensorView()
    : _tensor(nullptr), name("anonymous") {}

MoonshineTensorView::MoonshineTensorView(moonshine_tensor_t *tensor,
                                         std::string name)
    : _tensor(tensor), name(name) {
  // register_allocated_view(name, this);
  if (tensor == nullptr) {
    fprintf(stderr, "Tensor is nullptr\n");
    assert(false);
  }
  _shape =
      std::vector<int64_t>(tensor->shape, tensor->shape + tensor->shape_count);
}

MoonshineTensorView::MoonshineTensorView(const std::vector<int64_t> &shape,
                                         uint32_t dtype, void *data,
                                         const std::string &name)
    : _tensor(nullptr), _shape(shape), name(name) {
  _tensor = moonshine_tensor_from_shape_and_dtype(shape, dtype, data);
}

MoonshineTensorView::MoonshineTensorView(const MoonshineTensorView &other)
    : _shape(other._shape), name("constructor copy of " + other.name) {
  _shape = other._shape;
  _tensor = moonshine_tensor_from_shape_and_dtype(_shape, other._tensor->dtype,
                                                  other._tensor->data);
}

MoonshineTensorView::MoonshineTensorView(const OrtApi *ort_api,
                                         OrtValue *ort_tensor,
                                         const std::string &name)
    : _tensor(moonshine_tensor_from_ort_tensor(ort_api, ort_tensor)),
      name(name) {
  if (_tensor == nullptr) {
    throw std::runtime_error("Failed to create moonshine tensor '" + name +
                             "' from ort tensor");
  }
  _shape = std::vector<int64_t>(_tensor->shape,
                                _tensor->shape + _tensor->shape_count);
}

MoonshineTensorView::~MoonshineTensorView() { moonshine_free_tensor(_tensor); }

MoonshineTensorView &MoonshineTensorView::operator=(
    const MoonshineTensorView &other) {
  moonshine_free_tensor(_tensor);

  _shape = other._shape;
  _tensor = moonshine_tensor_from_shape_and_dtype(_shape, other._tensor->dtype,
                                                  other._tensor->data);
  name = "assignment copy of " + other.name;
  return *this;
}

std::vector<int64_t> &MoonshineTensorView::shape() { return _shape; }

size_t MoonshineTensorView::element_count() {
  return std::accumulate(shape().begin(), shape().end(), 1,
                         std::multiplies<int64_t>());
}

size_t MoonshineTensorView::bytes_count() {
  return std::accumulate(shape().begin(), shape().end(), 1,
                         std::multiplies<int64_t>()) *
         moonshine_dtype_to_bytes_per_element(dtype());
}

uint32_t MoonshineTensorView::dtype() { return _tensor->dtype; }

void MoonshineTensorView::reshape(const std::vector<int64_t> &shape) {
  size_t new_element_count = std::accumulate(shape.begin(), shape.end(), 1,
                                             std::multiplies<int64_t>());
  if (new_element_count != element_count()) {
    throw std::runtime_error("New shape has a different number of elements");
  }
  _shape = shape;
}

MoonshineTensorView MoonshineTensorView::cast_f16_to_f32() {
  if (dtype() != MOONSHINE_DTYPE_FLOAT16) {
    throw std::runtime_error("Tensor is not float16");
  }
  MoonshineTensorView f32(_shape, MOONSHINE_DTYPE_FLOAT32);
  uint16_t *f16_data = data<uint16_t>();
  float *f32_data = f32.data<float>();
  const size_t count = element_count();
  float16_to_float32(f16_data, f32_data, count);
  return f32;
}

int64_t MoonshineTensorView::argmax() {
  if (dtype() != MOONSHINE_DTYPE_FLOAT32) {
    throw std::runtime_error("Tensor is not float32");
  }
  size_t max_index = 0;
  float max_value = data<float>()[0];
  const size_t count = element_count();
  for (size_t i = 1; i < count; i++) {
    if (data<float>()[i] > max_value) {
      max_value = data<float>()[i];
      max_index = i;
    }
  }
  return static_cast<int64_t>(max_index);
}

moonshine_dtype_t ort_dtype_to_moonshine_dtype(
    ONNXTensorElementDataType ort_dtype) {
  switch (ort_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return MOONSHINE_DTYPE_FLOAT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return MOONSHINE_DTYPE_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return MOONSHINE_DTYPE_FLOAT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return MOONSHINE_DTYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return MOONSHINE_DTYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return MOONSHINE_DTYPE_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return MOONSHINE_DTYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return MOONSHINE_DTYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return MOONSHINE_DTYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return MOONSHINE_DTYPE_UINT64;
    default:
      throw std::runtime_error("Unsupported tensor type");
  }
}

ONNXTensorElementDataType moonshine_dtype_to_ort_dtype(
    uint32_t moonshine_dtype) {
  switch (moonshine_dtype) {
    case MOONSHINE_DTYPE_FLOAT16:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case MOONSHINE_DTYPE_FLOAT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case MOONSHINE_DTYPE_FLOAT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case MOONSHINE_DTYPE_INT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case MOONSHINE_DTYPE_INT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case MOONSHINE_DTYPE_UINT32:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case MOONSHINE_DTYPE_UINT64:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    case MOONSHINE_DTYPE_BOOL:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      throw std::runtime_error("Unsupported tensor type");
  }
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
}

size_t ort_dtype_to_bytes_per_element(ONNXTensorElementDataType ort_dtype) {
  switch (ort_dtype) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return sizeof(float);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return sizeof(double);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return sizeof(int32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return sizeof(int64_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return sizeof(uint8_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return sizeof(uint16_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      return sizeof(uint32_t);
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      return sizeof(uint64_t);
    default:
      throw std::runtime_error("Unsupported tensor type");
  }
  return 0;
}

size_t moonshine_dtype_to_bytes_per_element(uint32_t moonshine_dtype) {
  ONNXTensorElementDataType ort_dtype =
      moonshine_dtype_to_ort_dtype(moonshine_dtype);
  return ort_dtype_to_bytes_per_element(ort_dtype);
}

moonshine_tensor_t *moonshine_tensor_from_shape_and_dtype(
    const std::vector<int64_t> &shape, uint32_t dtype,
    const void *data_to_copy) {
  moonshine_tensor_t *moonshine_tensor = static_cast<moonshine_tensor_t *>(
      DEBUG_CALLOC(1, sizeof(moonshine_tensor_t)));
  moonshine_tensor->dtype = dtype;
  moonshine_tensor->shape =
      static_cast<int64_t *>(DEBUG_CALLOC(shape.size(), sizeof(int64_t)));
  std::memcpy(moonshine_tensor->shape, shape.data(),
              shape.size() * sizeof(int64_t));
  moonshine_tensor->shape_count = shape.size();
  const size_t bytes_per_element = moonshine_dtype_to_bytes_per_element(dtype);
  const size_t element_count = std::accumulate(shape.begin(), shape.end(), 1,
                                               std::multiplies<int64_t>());
  const size_t data_size_in_bytes = element_count * bytes_per_element;
  moonshine_tensor->data =
      static_cast<uint8_t *>(DEBUG_CALLOC(data_size_in_bytes, 1));
  if (data_to_copy != nullptr) {
    std::memcpy(moonshine_tensor->data, data_to_copy, data_size_in_bytes);
  }
  return moonshine_tensor;
}

MoonshineTensorView *moonshine_tensor_from_token_vector(
    std::vector<int32_t> &vector) {
  MoonshineTensorView *moonshine_tensor = new MoonshineTensorView(
      {static_cast<int64_t>(vector.size())}, MOONSHINE_DTYPE_INT32,
      vector.data(), TENSOR_NAME("tokens"));
  return moonshine_tensor;
}

std::vector<int32_t> token_vector_from_moonshine_tensor(
    MoonshineTensorView *moonshine_tensor) {
  return std::vector<int32_t>(
      moonshine_tensor->data<int32_t>(),
      moonshine_tensor->data<int32_t>() + moonshine_tensor->shape()[0]);
}

OrtValue *MoonshineTensorView::create_ort_value(const OrtApi *ort_api,
                                                OrtMemoryInfo *memory_info) {
  OrtValue *output_ort_tensor = nullptr;
  LOG_ORT_ERROR(
      ort_api,
      ort_api->CreateTensorWithDataAsOrtValue(
          memory_info, this->data<void>(), this->bytes_count(),
          this->shape().data(), this->shape().size(),
          moonshine_dtype_to_ort_dtype(this->dtype()), &output_ort_tensor));
  return output_ort_tensor;
}

// Thank you Claude.
void float16_to_float32(const uint16_t *f16_array, float *f32_array,
                        size_t count) {
  for (size_t i = 0; i < count; i++) {
    uint16_t h = f16_array[i];

    // Extract components
    uint32_t sign = (h & 0x8000) >> 15;
    uint32_t exponent = (h & 0x7C00) >> 10;
    uint32_t mantissa = h & 0x03FF;

    uint32_t f32_bits;

    if (exponent == 0) {
      // Zero or subnormal
      if (mantissa == 0) {
        // Zero
        f32_bits = sign << 31;
      } else {
        // Subnormal - convert to normalized float32
        exponent = 1;
        while ((mantissa & 0x0400) == 0) {
          mantissa <<= 1;
          exponent--;
        }
        mantissa &= 0x03FF;
        f32_bits =
            (sign << 31) | ((127 - 15 + exponent) << 23) | (mantissa << 13);
      }
    } else if (exponent == 31) {
      // Infinity or NaN
      f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
      // Normal number
      f32_bits =
          (sign << 31) | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    // Type-pun the bits to float
    *((uint32_t *)&f32_array[i]) = f32_bits;
  }
}

std::string MoonshineTensorView::to_string() {
  std::string result = "MoonshineTensorView name='" + name + "', shape=(";
  for (size_t i = 0; i < shape().size(); i++) {
    result += std::to_string(shape()[i]);
    if (i < shape().size() - 1) {
      result += ", ";
    }
  }
  result += "), dtype=" + std::to_string(dtype());
  return result;
}