// See op_profiler.h for the rationale (why not tflite::MicroProfiler).

#include "op_profiler.h"

#include <cstdio>

#include "pico/time.h"

namespace spelling {

uint32_t OpProfiler::BeginEvent(const char* tag) {
  if (num_events_ >= kMaxEvents) {
    overflowed_ = true;
    // Return the last valid slot; EndEvent will harmlessly overwrite it.
    return static_cast<uint32_t>(kMaxEvents - 1);
  }
  const int idx = num_events_++;
  tags_[idx] = tag;
  start_us_[idx] = time_us_32();
  end_us_[idx] = start_us_[idx];
  return static_cast<uint32_t>(idx);
}

void OpProfiler::EndEvent(uint32_t event_handle) {
  if (event_handle >= static_cast<uint32_t>(kMaxEvents)) return;
  end_us_[event_handle] = time_us_32();
}

void OpProfiler::Report(const char* label) const {
  // Per-instance breakdown: one line per op, in execution order. This
  // is what lets you separate the kernel>1 stem CONV_2D from the 1x1
  // pointwise CONV_2D instances (all tagged "CONV_2D" by TFLM).
  printf("[prof:%s] --- per-op (execution order) ---\n", label);
  uint32_t grand_total = 0;
  for (int i = 0; i < num_events_; ++i) {
    const uint32_t us = end_us_[i] - start_us_[i];
    grand_total += us;
    printf("[prof:%s] op %2d  %-20s %8lu us\n", label, i,
           tags_[i] ? tags_[i] : "(null)", static_cast<unsigned long>(us));
  }

  // Per-tag rollup: total + count per unique op type. Linear scan over
  // a tiny fixed table -- there are only a handful of distinct tags.
  struct TagAgg {
    const char* tag;
    uint32_t us;
    int count;
  };
  TagAgg agg[32] = {};
  int n_tags = 0;
  for (int i = 0; i < num_events_; ++i) {
    const uint32_t us = end_us_[i] - start_us_[i];
    int slot = -1;
    for (int t = 0; t < n_tags; ++t) {
      if (agg[t].tag == tags_[i]) {
        slot = t;
        break;
      }
    }
    if (slot < 0 && n_tags < 32) {
      slot = n_tags++;
      agg[slot].tag = tags_[i];
    }
    if (slot >= 0) {
      agg[slot].us += us;
      agg[slot].count += 1;
    }
  }

  printf("[prof:%s] --- per-tag totals ---\n", label);
  for (int t = 0; t < n_tags; ++t) {
    const double pct = grand_total ? (100.0 * static_cast<double>(agg[t].us) /
                                      static_cast<double>(grand_total))
                                   : 0.0;
    printf("[prof:%s] %-20s %8lu us  (%5.1f%%)  x%d\n", label,
           agg[t].tag ? agg[t].tag : "(null)",
           static_cast<unsigned long>(agg[t].us), pct, agg[t].count);
  }
  printf("[prof:%s] total profiled: %lu us across %d ops%s\n", label,
         static_cast<unsigned long>(grand_total), num_events_,
         overflowed_ ? "  [TRUNCATED: increase kMaxEvents]" : "");
}

}  // namespace spelling
