#ifndef MOONSHINE_TENSOR_H
#define MOONSHINE_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum moonshine_dtype_t {
  MOONSHINE_DTYPE_FLOAT16 = 0,
  MOONSHINE_DTYPE_FLOAT32 = 1,
  MOONSHINE_DTYPE_FLOAT64 = 2,
  MOONSHINE_DTYPE_INT8 = 3,
  MOONSHINE_DTYPE_INT16 = 4,
  MOONSHINE_DTYPE_INT32 = 5,
  MOONSHINE_DTYPE_INT64 = 6,
  MOONSHINE_DTYPE_UINT8 = 7,
  MOONSHINE_DTYPE_UINT16 = 8,
  MOONSHINE_DTYPE_UINT32 = 9,
  MOONSHINE_DTYPE_UINT64 = 10,
  MOONSHINE_DTYPE_BOOL = 11,
  MOONSHINE_DTYPE_MAX = 12,
} moonshine_dtype_t;

typedef struct {
  uint32_t dtype;  // Not an enum so we can be more sure of its memory layout.
  int64_t *shape;
  int64_t shape_count;
  void *data;
} moonshine_tensor_t;

typedef struct {
  moonshine_tensor_t **tensors;
  size_t tensors_count;
} moonshine_tensor_list_t;

void moonshine_free_tensor(moonshine_tensor_t *tensor);

void moonshine_free_tensor_list(moonshine_tensor_list_t *tensor_list);

#ifdef __cplusplus
}
#endif

#endif