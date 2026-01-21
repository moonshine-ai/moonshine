#include "moonshine-ort-allocator.h"

#define DEBUG_ALLOC_ENABLED 1
#include "debug-utils.h"

namespace {
void *MoonshineAlloc(struct OrtAllocator *this_, size_t size) {
  MoonshineOrtAllocator *moonshine_allocator = (MoonshineOrtAllocator *)this_;
  moonshine_allocator->total_allocated += size;
  void *result = DEBUG_CALLOC(size, 1);
  moonshine_allocator->allocated_blocks[result] = size;
  return result;
}

void MoonshineFree(struct OrtAllocator *this_, void *p) {
  MoonshineOrtAllocator *moonshine_allocator = (MoonshineOrtAllocator *)this_;
  moonshine_allocator->total_freed += p ? debug_alloc_get_size(p) : 0;
  DEBUG_FREE(p);
  moonshine_allocator->allocated_blocks.erase(p);
}

const struct OrtMemoryInfo *MoonshineInfo(const struct OrtAllocator *this_) {
  MoonshineOrtAllocator *moonshine_allocator = (MoonshineOrtAllocator *)this_;
  fprintf(stderr, "MoonshineInfo: %p\n",
          (void *)(moonshine_allocator->memory_info));
  return moonshine_allocator->memory_info;
}

void *MoonshineReserve(struct OrtAllocator *this_, size_t size) {
  MoonshineOrtAllocator *moonshine_allocator = (MoonshineOrtAllocator *)this_;
  moonshine_allocator->total_reserved += size;
  return DEBUG_CALLOC(size, 1);
}

void *MoonshineAllocOnStream(struct OrtAllocator *this_, size_t size,
                             OrtSyncStream *) {
  MoonshineOrtAllocator *moonshine_allocator = (MoonshineOrtAllocator *)this_;
  moonshine_allocator->total_alloc_on_stream += size;
  return DEBUG_CALLOC(size, 1);
}

OrtStatus *MoonshineGetStats(const struct OrtAllocator *this_,
                             OrtKeyValuePairs **outPairs) noexcept {
  MoonshineOrtAllocator *moonshine_allocator = (MoonshineOrtAllocator *)this_;
  moonshine_allocator->total_stats_requested += 1;
  *outPairs = nullptr;
  return nullptr;
}

void friendlySizeString(size_t byte_count, char *output, size_t output_size) {
  if (byte_count < 1024) {
    snprintf(output, output_size, "%zu bytes", byte_count);
  } else if (byte_count < 1024 * 1024) {
    const float kb = byte_count / 1024.0f;
    snprintf(output, output_size, "%.2f KB (%zu bytes)", kb, byte_count);
  } else if (byte_count < 1024 * 1024 * 1024) {
    const float mb = byte_count / (1024 * 1024.0f);
    snprintf(output, output_size, "%.2f MB (%zu bytes)", mb, byte_count);
  } else {
    const float gb = byte_count / (1024 * 1024 * 1024.0f);
    snprintf(output, output_size, "%.2f GB (%zu bytes)", gb, byte_count);
  }
}

void printFriendlySize(const char *prefix, size_t number) {
  char output[100];
  friendlySizeString(number, output, sizeof(output));
  fprintf(stderr, "%s %s\n", prefix, output);
}
}  // namespace

MoonshineOrtAllocator::MoonshineOrtAllocator(const OrtMemoryInfo *memory_info) {
  base.version = ORT_API_VERSION;
  base.Alloc = MoonshineAlloc;
  base.Free = MoonshineFree;
  base.Info = MoonshineInfo;
  base.Reserve = MoonshineReserve;
  base.GetStats = MoonshineGetStats;
  base.AllocOnStream = MoonshineAllocOnStream;
  this->memory_info = memory_info;
  this->total_allocated = 0;
  this->total_freed = 0;
  this->total_reserved = 0;
  this->total_stats_requested = 0;
  this->total_stats_released = 0;
  this->total_alloc_on_stream = 0;
}

MoonshineOrtAllocator::~MoonshineOrtAllocator() {
  //   print_stats();
}

void MoonshineOrtAllocator::print_stats() {
  printFriendlySize("Total allocated: ", total_allocated);
  printFriendlySize("Total freed: ", total_freed);
  printFriendlySize("Total reserved: ", total_reserved);
  printFriendlySize("Total stats requested: ", total_stats_requested);
  printFriendlySize("Total stats released: ", total_stats_released);
  printFriendlySize("Total alloc on stream: ", total_alloc_on_stream);
  fprintf(stderr, "Allocated blocks: %zu\n", allocated_blocks.size());
  for (auto it = allocated_blocks.begin(); it != allocated_blocks.end(); ++it) {
    fprintf(stderr, "  %p: %zu\n", it->first, it->second);
  }
  total_allocated = 0;
  total_freed = 0;
  total_reserved = 0;
  total_stats_requested = 0;
  total_stats_released = 0;
  total_alloc_on_stream = 0;
  allocated_blocks.clear();
}