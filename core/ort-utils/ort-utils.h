#ifndef ORT_UTILS_H
#define ORT_UTILS_H

#include <filesystem>
#include <string>
#include <vector>

#if defined(ANDROID)
#include <android/asset_manager.h>
#endif

#include "debug-utils.h"
#include "onnxruntime_c_api.h"

#define RETURN_ON_ORT_ERROR(ort_api, expr)                     \
  do {                                                         \
    OrtStatus *onnx_status = (expr);                           \
    if (onnx_status != NULL) {                                 \
      const char *msg = ort_api->GetErrorMessage(onnx_status); \
      LOGF("ORT Error: %s", msg);                              \
      ort_api->ReleaseStatus(onnx_status);                     \
      return -1;                                               \
    }                                                          \
  } while (0);

#define LOG_ORT_ERROR(ort_api, expr)                           \
  do {                                                         \
    OrtStatus *onnx_status = (expr);                           \
    if (onnx_status != NULL) {                                 \
      const char *msg = ort_api->GetErrorMessage(onnx_status); \
      LOGF("ORT Error: %s", msg);                              \
      ort_api->ReleaseStatus(onnx_status);                     \
    }                                                          \
  } while (0);

int ort_session_from_path(const OrtApi *ort_api, OrtEnv *env,
                          OrtSessionOptions *session_options, const char *path,
                          OrtSession **session, const char **mmapped_data,
                          size_t *mmapped_data_size);

int ort_session_from_memory(const OrtApi *ort_api, OrtEnv *env,
                            OrtSessionOptions *session_options,
                            const uint8_t *data, size_t data_size,
                            OrtSession **session);

#if defined(ANDROID)
int ort_session_from_asset(const OrtApi *ort_api, OrtEnv *env,
                           OrtSessionOptions *session_options,
                           AAssetManager *assetManager, const char *path,
                           OrtSession **session, const char **mmapped_data,
                           size_t *mmapped_data_size);
#endif

std::vector<int64_t> ort_get_shape(const OrtApi *ort_api,
                                   OrtTypeInfo *type_info);

ONNXTensorElementDataType ort_get_type(const OrtApi *ort_api,
                                       OrtTypeInfo *type_info);

std::vector<int64_t> ort_get_input_shape(const OrtApi *ort_api,
                                         OrtSession *session, int index);

ONNXTensorElementDataType ort_get_input_type(const OrtApi *ort_api,
                                             OrtSession *session, int index);

std::vector<int64_t> ort_get_output_shape(const OrtApi *ort_api,
                                          OrtSession *session, int index);

ONNXTensorElementDataType ort_get_output_type(const OrtApi *ort_api,
                                              OrtSession *session, int index);

std::vector<int64_t> ort_get_value_shape(const OrtApi *ort_api,
                                         const OrtValue *value);

ONNXTensorElementDataType ort_get_value_type(const OrtApi *ort_api,
                                             const OrtValue *value);

#define ORT_RUN(ort_api, session, input_names, inputs, input_count,         \
                output_names, output_count, outputs)                        \
  ort_run(ort_api, session, input_names, inputs, input_count, output_names, \
          output_count, outputs, #session, this->log_ort_run)

OrtStatus *ort_run(const OrtApi *ort_api, OrtSession *session,
                   const char *const *input_names,
                   const OrtValue *const *inputs, size_t input_len,
                   const char *const *output_names, size_t output_names_len,
                   OrtValue **outputs, const char *session_name,
                   bool log_ort_run);

#endif