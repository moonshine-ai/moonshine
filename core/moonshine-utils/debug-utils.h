#ifndef UTILS_H
#define UTILS_H

#include <cinttypes>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#if defined(ANDROID)
#include <android/asset_manager.h>
#endif

static inline const char *_moonshine_filename_without_path(const char *path) {
  const char *filename = strrchr(path, '/');
  if (filename == NULL) {
    filename = strrchr(path, '\\');
  }
  return (filename == NULL) ? path : filename + 1;
}

#define FILENAME_ONLY (_moonshine_filename_without_path(__FILE__))

#if defined(ANDROID)
#include <android/log.h>
#define LOGF(format, ...)                                                  \
  do {                                                                     \
    __android_log_print(ANDROID_LOG_WARN, "Native", "%s:%d:%s(): " format, \
                        FILENAME_ONLY, __LINE__, __func__, __VA_ARGS__);   \
  } while (0)
#else
#define LOGF(format, ...)                                                     \
  do {                                                                        \
    fprintf(stderr, "%s:%d:%s(): " format, FILENAME_ONLY, __LINE__, __func__, \
            __VA_ARGS__);                                                     \
    fprintf(stderr, "\n");                                                    \
  } while (0)
#endif
#define LOG(x) LOGF("%s", (x))

#define RETURN_ON_ERROR(error)  \
  do {                          \
    if (error != 0) {           \
      LOGF("Error: %d", error); \
      return -1;                \
    }                           \
  } while (0)

#define RETURN_ON_FALSE(expr)           \
  do {                                  \
    if (!(expr)) {                      \
      LOG("Error: " #expr " is false"); \
      return -1;                        \
    }                                   \
  } while (0)

#define RETURN_ON_NULL(ptr)              \
  do {                                   \
    if (ptr == nullptr) {                \
      LOG("Error: " #ptr " is nullptr"); \
      return -1;                         \
    }                                    \
  } while (0)

#define RETURN_ON_NOT_EQUAL(expr1, expr2)      \
  do {                                         \
    if ((expr1) != (expr2)) {                  \
      LOGF("Error: %s != %s", #expr1, #expr2); \
      return -1;                               \
    }                                          \
  } while (0);

#define RETURN_ON_FILE_DOES_NOT_EXIST(path)          \
  do {                                               \
    if (!std::filesystem::exists(path)) {            \
      LOGF("Error: File '%s' does not exist", path); \
      return -1;                                     \
    }                                                \
  } while (0);

#define ENABLE_TIMER 1

#ifdef ENABLE_TIMER
#include <chrono>
#define TIMER_START(x) \
  auto x##_timer_start = std::chrono::high_resolution_clock::now();

#define TIMER_END(x)                                                          \
  auto x##_timer_end = std::chrono::high_resolution_clock::now();             \
  auto x##_timer_duration =                                                   \
      std::chrono::duration_cast<std::chrono::milliseconds>(x##_timer_end -   \
                                                            x##_timer_start); \
  LOGF(#x " took %lld milliseconds", (long long)x##_timer_duration.count());
#else
#define TIMER_START(x)
#define TIMER_END(x)
#endif

#ifdef DEBUG_ALLOC_ENABLED

// Needed in every file that uses DEBUG_CALLOC and DEBUG_FREE, declared static
// to avoid name conflicts.
namespace {
static size_t g_debug_alloc_size = 0;
}

#define DEBUG_ALLOC_MAGIC (0x71A3BEEF)

#define DEBUG_ALLOC_ALIGNMENT (32)

#define DEBUG_ALLOC_LOG_MIN_SIZE (1 * 1024 * 1024)  // 10MB

#define DEBUG_CALLOC(size, count) \
  debug_calloc(size, count, FILENAME_ONLY, __LINE__, __FUNCTION__)

namespace {
static inline void *debug_calloc(size_t size, size_t count, const char *, int,
                                 const char *) {
  size_t byteCount = DEBUG_ALLOC_ALIGNMENT + (size * count);
  g_debug_alloc_size += byteCount;
  uint8_t *ptr = (uint8_t *)calloc(byteCount, 1);
  size_t *sizePtr = (size_t *)(ptr + 0);
  uint32_t *magicPtr = (uint32_t *)(ptr + sizeof(size_t));
  *magicPtr = DEBUG_ALLOC_MAGIC;
  uint8_t *memPtr = (ptr + DEBUG_ALLOC_ALIGNMENT);
  *sizePtr = byteCount;
  return memPtr;
}
}  // namespace

#define DEBUG_FREE(ptr) debug_free(ptr, FILENAME_ONLY, __LINE__, __FUNCTION__)

static inline void debug_free(void *voidMemPtr, const char *file, int line,
                              const char *function) {
  if (voidMemPtr == nullptr) {
    return;
  }
  uint8_t *ptr = ((uint8_t *)voidMemPtr) - DEBUG_ALLOC_ALIGNMENT;
  size_t *sizePtr = (size_t *)(ptr + 0);
  uint32_t *magicPtr = (uint32_t *)(ptr + sizeof(size_t));
  if (*magicPtr != DEBUG_ALLOC_MAGIC) {
    LOGF(
        "Error: DEBUG_FREE: Invalid magic number: got 0x%08x, expected 0x%08x "
        "at %s:%d %s",
        *magicPtr, DEBUG_ALLOC_MAGIC, file, line, function);
    return;
  }
  size_t byteCount = *sizePtr;
  g_debug_alloc_size -= byteCount;
  free(sizePtr);
}

static inline size_t debug_alloc_get_size(void *voidMemPtr) {
  if (voidMemPtr == nullptr) {
    return 0;
  }
  uint8_t *ptr = ((uint8_t *)voidMemPtr) - DEBUG_ALLOC_ALIGNMENT;
  size_t *sizePtr = (size_t *)(ptr + 0);
  uint32_t *magicPtr = (uint32_t *)(ptr + sizeof(size_t));
  if (*magicPtr != DEBUG_ALLOC_MAGIC) {
    LOGF(
        "Error: debug_alloc_get_size: Invalid magic number: got 0x%08x, "
        "expected 0x%08x",
        *magicPtr, DEBUG_ALLOC_MAGIC);
    return 0;
  }
  return *sizePtr;
}

#else  // DEBUG_ALLOC_ENABLED

#define DEBUG_CALLOC(size, count) calloc(size, count)
#define DEBUG_FREE(ptr) free(ptr)

#endif  // DEBUG_ALLOC_ENABLED

#define TRACE()   \
  do {            \
    LOG("TRACE"); \
  } while (0)

#define THROW_WITH_LOG(message)                                             \
  do {                                                                      \
    LOG(message);                                                           \
    throw std::runtime_error(                                               \
        std::string(FILENAME_ONLY) + ":" + std::to_string(__LINE__) + ":" + \
        std::string(__func__) + " - " + std::string(message));              \
  } while (0)

#define LOG_INT(x) LOGF(#x " = %d", (x));
#define LOG_INT64(x) LOGF(#x " = %" PRId64, (x));
#define LOG_UINT64(x) LOGF(#x " = %" PRIu64, (x));
#define LOG_LONG(x) LOGF(#x " = %ld", (x));
#define LOG_SIZET(x) LOGF(#x " = %zu", (x));
#define LOG_PTR(x) LOGF(#x " = %p", (void *)(x));
#define LOG_VECTOR(x)                       \
  do {                                      \
    std::stringstream ss;                   \
    ss << #x << " = [";                     \
    for (size_t i = 0; i < x.size(); i++) { \
      ss << x[i];                           \
      if (i < x.size() - 1) {               \
        ss << ", ";                         \
      }                                     \
    }                                       \
    ss << "]";                              \
    LOGF("%s", ss.str().c_str());           \
  } while (0)
#define LOG_FLOAT(x) LOGF(#x " = %f", (x));
#define LOG_STRING(x) LOGF(#x " = %s", (x).c_str());
#define LOG_BOOL(x) LOGF(#x " = %s", (x) ? "true" : "false");
#define LOG_BYTES(x, size)                                     \
  do {                                                         \
    std::stringstream ss;                                      \
    ss << #x << " = [";                                        \
    for (size_t i = 0; i < (size); i++) {                      \
      if (i % 16 == 0) {                                       \
        ss << "\n  ";                                          \
      }                                                        \
      char buffer[4];                                          \
      snprintf(buffer, 4, "%02x ", (unsigned char)(((x))[i])); \
      ss << buffer;                                            \
    }                                                          \
    ss << "\n]";                                               \
    LOGF("%s", ss.str().c_str());                              \
  } while (0)

bool load_wav_data(const char *path, float **out_float_data,
                   size_t *out_num_samples, int32_t *out_sample_rate = nullptr);

bool save_wav_data(const char *path, const float *audio_data,
                   size_t num_samples, uint32_t sample_rate = 16000);

std::string float_vector_stats_to_string(const std::vector<float> &vector);

std::vector<uint8_t> load_file_into_memory(const std::string &path);
void save_memory_to_file(const std::string &path,
                         const std::vector<uint8_t> &data);

template <typename T>
T gate(T value, T min, T max) {
  return std::max<T>(min, std::min<T>(value, max));
}

#endif