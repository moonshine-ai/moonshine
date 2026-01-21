#ifndef MOONSHINE_ORT_ALLOCATOR_H
#define MOONSHINE_ORT_ALLOCATOR_H

#include <map>

#include "onnxruntime_c_api.h"

struct MoonshineOrtAllocator {
  OrtAllocator base;
  const OrtMemoryInfo *memory_info;
  size_t total_allocated;
  size_t total_freed;
  size_t total_reserved;
  size_t total_stats_requested;
  size_t total_stats_released;
  size_t total_alloc_on_stream;

  std::map<void *, size_t> allocated_blocks;

  MoonshineOrtAllocator(const OrtMemoryInfo *memory_info);

  ~MoonshineOrtAllocator();

  void print_stats();
};

#endif