#include "ort-utils.h"

#include <filesystem>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include "onnxruntime_c_api.h"

#ifdef _WIN32
// No memory mapping on Windows and wchar for the file path.
int ort_session_from_path(const OrtApi *ort_api, OrtEnv *env,
                          OrtSessionOptions *session_options, const char *path,
                          OrtSession **session, const char **mmapped_data,
                          size_t *mmapped_data_size) {
  if (!std::filesystem::exists(path)) {
    fprintf(stderr, "Model directory '%s' does not exist at %s:%d\n", path,
            __FILE__, __LINE__);
    return -1;
  }
  std::filesystem::path fs_path(path);
  std::wstring wpath = fs_path.wstring();
  RETURN_ON_ORT_ERROR(
      ort_api,
      ort_api->CreateSession(env, wpath.c_str(), session_options, session));
  *mmapped_data = nullptr;
  *mmapped_data_size = 0;
  return 0;
}
#else
int ort_session_from_path(const OrtApi *ort_api, OrtEnv *env,
                          OrtSessionOptions *session_options, const char *path,
                          OrtSession **session, const char **mmapped_data,
                          size_t *mmapped_data_size) {
  std::string cpp_path(path);
  if (cpp_path.find(".ort") != std::string::npos) {
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
      fprintf(stderr, "Failed to open memory map file %s at %s:%d\n", path,
              __FILE__, __LINE__);
      return -1;
    }
    struct stat st;
    if (fstat(fd, &st) == -1) {
      fprintf(stderr, "Failed to get file size for %s at %s:%d\n", path,
              __FILE__, __LINE__);
      return -1;
    }
    *mmapped_data = static_cast<const char *>(
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (*mmapped_data == MAP_FAILED) {
      fprintf(stderr, "Failed to memory map file %s at %s:%d\n", path, __FILE__,
              __LINE__);
      return -1;
    }
    *mmapped_data_size = st.st_size;
    close(fd);
    RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateSessionFromArray(
                                     env, *mmapped_data, *mmapped_data_size,
                                     session_options, session));
  } else {
    if (!std::filesystem::exists(path)) {
      fprintf(stderr, "Model directory '%s' does not exist at %s:%d\n", path,
              __FILE__, __LINE__);
      return -1;
    }
    RETURN_ON_ORT_ERROR(
        ort_api, ort_api->CreateSession(env, path, session_options, session));
    *mmapped_data = nullptr;
    *mmapped_data_size = 0;
  }
  return 0;
}
#endif

int ort_session_from_memory(const OrtApi *ort_api, OrtEnv *env,
                            OrtSessionOptions *session_options,
                            const uint8_t *data, size_t data_size,
                            OrtSession **session) {
  RETURN_ON_NULL(ort_api);
  RETURN_ON_NULL(data);
  RETURN_ON_ORT_ERROR(
      ort_api, ort_api->CreateSessionFromArray(env, data, data_size,
                                               session_options, session));
  return 0;
}

#if defined(ANDROID)
int ort_session_from_asset(const OrtApi *ort_api, OrtEnv *env,
                           OrtSessionOptions *session_options,
                           AAssetManager *assetManager, const char *path,
                           OrtSession **session, const char **mmapped_data,
                           size_t *mmapped_data_size) {
  AAsset *asset = AAssetManager_open(assetManager, path, AASSET_MODE_STREAMING);
  if (asset == nullptr) {
    fprintf(stderr, "Failed to open asset %s at %s:%d\n", path, __FILE__,
            __LINE__);
    return -1;
  }
  off64_t start, length;
  int fd = AAsset_openFileDescriptor64(asset, &start, &length);
  if (fd == -1) {
    fprintf(stderr, "Failed to open file descriptor for asset %s at %s:%d\n",
            path, __FILE__, __LINE__);
    return -1;
  }
  const off64_t pageSize = sysconf(_SC_PAGESIZE);
  const off64_t alignOffset = start % pageSize;
  const off64_t alignedStart = start - alignOffset;
  const off64_t alignedLength = length + alignOffset;
  *mmapped_data = static_cast<const char *>(
      mmap(nullptr, alignedLength, PROT_READ, MAP_PRIVATE, fd, alignedStart));
  if (*mmapped_data == MAP_FAILED) {
    fprintf(stderr,
            "Failed to memory map asset '%s'. FD: %d, Length: %jd, Start: %jd. "
            "Error: %s (errno %d) at %s:%d\n",
            path, fd, (intmax_t)alignedLength, (intmax_t)alignedStart,
            strerror(errno), errno, __FILE__, __LINE__);
    // perror("mmap details"); // You can also use perror
    close(fd);
    AAsset_close(asset);  // Also close the asset itself
    return -1;
  }
  *mmapped_data_size = alignedLength;
  close(fd);
  const char *unaligned_mmapped_data = *mmapped_data + alignOffset;
  RETURN_ON_ORT_ERROR(ort_api, ort_api->CreateSessionFromArray(
                                   env, unaligned_mmapped_data, length,
                                   session_options, session));
  return 0;
}
#endif

std::vector<int64_t> ort_get_shape(const OrtApi *ort_api,
                                   OrtTypeInfo *type_info) {
  const OrtTensorTypeAndShapeInfo *tensor_info;
  LOG_ORT_ERROR(ort_api,
                ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info));
  ONNXTensorElementDataType type;
  LOG_ORT_ERROR(ort_api, ort_api->GetTensorElementType(tensor_info, &type));

  // Get input shapes/dims
  size_t num_dims;
  LOG_ORT_ERROR(ort_api, ort_api->GetDimensionsCount(tensor_info, &num_dims));
  std::vector<int64_t> shape(num_dims);
  LOG_ORT_ERROR(ort_api,
                ort_api->GetDimensions(tensor_info, shape.data(), num_dims));
  return shape;
}

ONNXTensorElementDataType ort_get_type(const OrtApi *ort_api,
                                       OrtTypeInfo *type_info) {
  const OrtTensorTypeAndShapeInfo *tensor_info;
  LOG_ORT_ERROR(ort_api,
                ort_api->CastTypeInfoToTensorInfo(type_info, &tensor_info));
  ONNXTensorElementDataType type;
  LOG_ORT_ERROR(ort_api, ort_api->GetTensorElementType(tensor_info, &type));
  return type;
}

std::vector<int64_t> ort_get_input_shape(const OrtApi *ort_api,
                                         OrtSession *session, int index) {
  OrtTypeInfo *type_info;
  LOG_ORT_ERROR(ort_api,
                ort_api->SessionGetInputTypeInfo(session, index, &type_info));
  return ort_get_shape(ort_api, type_info);
}

ONNXTensorElementDataType ort_get_input_type(const OrtApi *ort_api,
                                             OrtSession *session, int index) {
  OrtTypeInfo *type_info;
  LOG_ORT_ERROR(ort_api,
                ort_api->SessionGetInputTypeInfo(session, index, &type_info));
  return ort_get_type(ort_api, type_info);
}

std::vector<int64_t> ort_get_output_shape(const OrtApi *ort_api,
                                          OrtSession *session, int index) {
  OrtTypeInfo *type_info;
  LOG_ORT_ERROR(ort_api,
                ort_api->SessionGetOutputTypeInfo(session, index, &type_info));
  return ort_get_shape(ort_api, type_info);
}

ONNXTensorElementDataType ort_get_output_type(const OrtApi *ort_api,
                                              OrtSession *session, int index) {
  OrtTypeInfo *type_info;
  LOG_ORT_ERROR(ort_api,
                ort_api->SessionGetOutputTypeInfo(session, index, &type_info));
  return ort_get_type(ort_api, type_info);
}

std::vector<int64_t> ort_get_value_shape(const OrtApi *ort_api,
                                         const OrtValue *value) {
  OrtTypeInfo *type_info;
  LOG_ORT_ERROR(ort_api, ort_api->GetTypeInfo(value, &type_info));
  return ort_get_shape(ort_api, type_info);
}

ONNXTensorElementDataType ort_get_value_type(const OrtApi *ort_api,
                                             const OrtValue *value) {
  OrtTypeInfo *type_info;
  LOG_ORT_ERROR(ort_api, ort_api->GetTypeInfo(value, &type_info));
  return ort_get_type(ort_api, type_info);
}

OrtStatus *ort_run(const OrtApi *ort_api, OrtSession *session,
                   const char *const *input_names,
                   const OrtValue *const *inputs, size_t input_len,
                   const char *const *output_names, size_t output_names_len,
                   OrtValue **outputs, const char *session_name,
                   bool log_ort_run) {
  if (!log_ort_run) {
    return ort_api->Run(session, nullptr, input_names, inputs, input_len,
                        output_names, output_names_len, outputs);
  }
  std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();
  OrtStatus *status =
      ort_api->Run(session, nullptr, input_names, inputs, input_len,
                   output_names, output_names_len, outputs);
  std::chrono::steady_clock::time_point end_time =
      std::chrono::steady_clock::now();
  std::chrono::duration<double, std::milli> duration = end_time - start_time;
  LOGF("ORT Run %s took %.2f ms for inputs:", session_name, duration.count());
  for (size_t i = 0; i < input_len; i++) {
    std::vector<int64_t> shape = ort_get_value_shape(ort_api, inputs[i]);
    std::stringstream ss;
    ss << input_names[i] << " = [";
    for (size_t j = 0; j < shape.size(); j++) {
      ss << shape[j];
      if (j < shape.size() - 1) {
        ss << ", ";
      }
    }
    ss << "]";
    LOGF("%s", ss.str().c_str());
  }
  return status;
}