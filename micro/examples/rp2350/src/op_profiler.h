// Lightweight per-op profiler for the moonshine-micro SpellingCNN build.
//
// Why not tflite::MicroProfiler? The stock implementation statically
// allocates kMaxEvents=4096 entries across four parallel arrays
// (~80 KB of SRAM). The RP2350 build runs with only ~20 KB of static
// margin, so the stock profiler simply doesn't fit. Our model executes
// only a few dozen ops per Invoke(), so a tiny fixed-capacity recorder
// is plenty.
//
// This records, per Invoke():
//   * one (tag, microseconds) sample per op INSTANCE, in execution
//     order -- so individual conv layers are distinguishable, e.g. the
//     kernel>1 stem CONV_2D vs the many 1x1 pointwise CONV_2D instances
//     (TFLM gives them all the "CONV_2D" tag, so per-instance order is
//     the only way to tell them apart),
//   * a per-tag total, so you get the "where does the time go across op
//     TYPES" rollup (CONV_2D vs DEPTHWISE_CONV_2D vs ADD vs ...).
//
// Timing uses the RP2350's free-running 64-bit microsecond timer
// (time_us_64), which is driven from a fixed reference clock and so is
// unaffected by any sys-clock overclock. BeginEvent/EndEvent each cost
// a couple of timer reads (sub-microsecond), negligible against the
// millisecond-scale ops being measured.

#ifndef SPELLING_OP_PROFILER_H_
#define SPELLING_OP_PROFILER_H_

#include <cstdint>

#include "tensorflow/lite/micro/micro_profiler_interface.h"

namespace spelling {

class OpProfiler : public tflite::MicroProfilerInterface {
 public:
  OpProfiler() = default;

  // MicroProfilerInterface. BeginEvent returns the slot index used as
  // the handle; EndEvent stamps the end time into that slot. TFLM uses
  // these in strict LIFO (ScopedMicroProfiler) order, one pair per op.
  uint32_t BeginEvent(const char* tag) override;
  void EndEvent(uint32_t event_handle) override;

  // Drop all recorded events (call before each Invoke you want to
  // measure in isolation).
  void Reset() {
    num_events_ = 0;
    overflowed_ = false;
  }

  int num_events() const { return num_events_; }

  // Print the per-instance and per-tag breakdown over USB stdio. `label`
  // is prefixed to each line so multiple reports can be told apart.
  void Report(const char* label) const;

 private:
  // Plenty for this model (memory plan shows ~50 buffers => well under
  // 256 op instances). If a future model exceeds this we just stop
  // recording and flag it in the report rather than smashing memory.
  static constexpr int kMaxEvents = 256;

  const char* tags_[kMaxEvents] = {};
  uint32_t start_us_[kMaxEvents] = {};
  uint32_t end_us_[kMaxEvents] = {};
  int num_events_ = 0;
  bool overflowed_ = false;
};

}  // namespace spelling

#endif  // SPELLING_OP_PROFILER_H_
