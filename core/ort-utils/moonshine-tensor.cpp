#include "moonshine-tensor.h"

#include "debug-utils.h"
#include "ort-utils.h"

extern "C" void moonshine_free_tensor(moonshine_tensor_t *tensor) {
  if (tensor == NULL) {
    return;
  }
  DEBUG_FREE(tensor->shape);
  DEBUG_FREE(tensor->data);
  DEBUG_FREE(tensor);
}

extern "C" void moonshine_free_tensor_list(
    moonshine_tensor_list_t *tensor_list) {
  if (tensor_list == nullptr) {
    return;
  }
  for (size_t i = 0; i < tensor_list->tensors_count; i++) {
    DEBUG_FREE(tensor_list->tensors[i]->shape);
    DEBUG_FREE(tensor_list->tensors[i]->data);
    DEBUG_FREE(tensor_list->tensors[i]);
  }
  DEBUG_FREE(tensor_list->tensors);
  DEBUG_FREE(tensor_list);
}
