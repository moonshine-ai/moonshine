// Rung-2 bring-up firmware: banner + TFLM interpreter init + ONE Invoke()
// of the s16x8 neural-TTS decoder model, with a blocking serial print
// before every op. The full TTS app hangs (no fault) inside the first
// Invoke(); this target exists to name the guilty kernel -- the last
// "op N ..." line on the wire is the one that never returned.
//
// Deliberately minimal: no WORLD synth, no USB audio, no PbDecoder
// wrapper, zero-filled input (kernel loop bounds depend on shapes, not
// values). Prints are printf+fflush+sleep so each line escapes the CDC
// FIFO before the next op runs.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "hardware/clocks.h"
#include "hardware/structs/qmi.h"
#include "hardware/structs/watchdog.h"
#include "hardware/vreg.h"
#include "hardware/watchdog.h"
#include "neural_tts_demo_data.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_profiler_interface.h"
#include "tensorflow/lite/schema/schema_generated.h"

// Same hard-fault trap as the TTS app: stash PC/LR/CFSR in watchdog
// scratch and reboot, so a crash is distinguishable from a hang on the
// next boot. Naked shim clears MSPLIM before any stack use so an
// MSPLIM violation can't double-fault into LOCKUP.
extern "C" void isr_hardfault_c(uint32_t* sp) {
  watchdog_hw->scratch[4] = 0xFA17FA17u;
  watchdog_hw->scratch[5] = sp[6];                             // PC
  watchdog_hw->scratch[6] = sp[5];                             // LR
  watchdog_hw->scratch[7] = *(volatile uint32_t*)0xE000ED28u;  // CFSR
  watchdog_reboot(0, 0, 10);
  while (true) {
  }
}

extern "C" __attribute__((naked)) void isr_hardfault(void) {
  __asm volatile(
      "movs r0, #0\n"
      "msr msplim, r0\n"
      "mrs r0, msp\n"
      "b isr_hardfault_c\n");
}

namespace {

#define TRACE(...)       \
  do {                   \
    printf(__VA_ARGS__); \
    fflush(stdout);      \
    sleep_ms(20);        \
  } while (0)

constexpr size_t kArenaBytes = 300 * 1024;
alignas(16) uint8_t g_arena[kArenaBytes];

// Accumulates per-op wall time silently (no serial I/O inside Invoke, so
// timings are undistorted); Report() prints the table afterwards.
class TimingProfiler : public tflite::MicroProfilerInterface {
 public:
  static constexpr int kMaxOps = 64;
  uint32_t BeginEvent(const char* tag) override {
    if (op_index_ < kMaxOps) tags_[op_index_] = tag;
    start_us_ = time_us_32();
    return op_index_++;
  }
  void EndEvent(uint32_t handle) override {
    if (handle < kMaxOps) us_[handle] += time_us_32() - start_us_;
  }
  void Reset() { op_index_ = 0; }
  void Report() {
    uint32_t total = 0;
    for (int i = 0; i < op_index_ && i < kMaxOps; ++i) total += us_[i];
    for (int i = 0; i < op_index_ && i < kMaxOps; ++i) {
      TRACE("op %2d %-16s %8lu us (%2lu%%)\n", i,
            tags_[i] ? tags_[i] : "(null)", (unsigned long)us_[i],
            (unsigned long)(us_[i] * 100ull / (total ? total : 1)));
      us_[i] = 0;
    }
    TRACE("total %lu us\n", (unsigned long)total);
  }

 private:
  int op_index_ = 0;
  uint32_t start_us_ = 0;
  const char* tags_[kMaxOps] = {};
  uint32_t us_[kMaxOps] = {};
};

// Feed the watchdog from a timer IRQ for the first ~2 minutes: single ops
// may legitimately run tens of seconds (reference int16 TRANSPOSE_CONV),
// but a genuine hang must still reboot the board so it stays reachable
// over USB without a physical replug.
bool FeedWatchdog(repeating_timer_t*) {
  static int beats = 0;
  if (++beats <= 120) watchdog_update();
  return true;
}

}  // namespace

int main() {
  // Same stack-guard relaxation as the TTS app (see its comment): give
  // the 4 KiB core-0 stack legal spill room down to the end of the heap.
  extern char __end__;
  __asm volatile("msr msplim, %0" ::"r"(&__end__ + PICO_HEAP_SIZE));

  // Same overclock as the TTS app: 300 MHz at 1.30 V, flash divider 3
  // (100 MHz QSPI, within the chip's 133 MHz rating).
#ifndef INVOKE_TEST_SYS_KHZ
#define INVOKE_TEST_SYS_KHZ 300000
#endif
#if INVOKE_TEST_SYS_KHZ > 150000
  vreg_set_voltage(VREG_VOLTAGE_1_30);
  sleep_ms(10);
  hw_write_masked(&qmi_hw->m[0].timing, 3 << QMI_M0_TIMING_CLKDIV_LSB,
                  QMI_M0_TIMING_CLKDIV_BITS);
  set_sys_clock_khz(INVOKE_TEST_SYS_KHZ, true);
#endif

  stdio_init_all();
  const bool was_watchdog = watchdog_caused_reboot();
  watchdog_enable(8000, true);

  for (int i = 0; i < 8; ++i) {
    watchdog_update();
    TRACE("tflm_invoke_test (boot %d)\n", i);
    if (watchdog_hw->scratch[4] == 0xFA17FA17u) {
      TRACE("!!! PREVIOUS BOOT HARD FAULT pc=%08lx lr=%08lx cfsr=%08lx\n",
            watchdog_hw->scratch[5], watchdog_hw->scratch[6],
            watchdog_hw->scratch[7]);
    } else if (was_watchdog) {
      TRACE("!!! previous boot ended in WATCHDOG reset (hang, not fault)\n");
    }
    sleep_ms(1000);
  }
  watchdog_hw->scratch[4] = 0;

  TRACE("model: %u bytes\n", static_cast<unsigned>(g_pb_decoder_model_len));
  const tflite::Model* model = tflite::GetModel(g_pb_decoder_model);
  TRACE("schema version %d\n", static_cast<int>(model->version()));

  static tflite::MicroMutableOpResolver<6> resolver;
  resolver.AddTranspose();
  resolver.AddReshape();
  resolver.AddTransposeConv();
  resolver.AddAdd();
  resolver.AddGelu();
  resolver.AddConv2D();

  static TimingProfiler profiler;
  static tflite::MicroInterpreter interp(model, resolver, g_arena,
                                         kArenaBytes,
                                         /*resource_variables=*/nullptr,
                                         &profiler);
  TRACE("AllocateTensors...\n");
  watchdog_update();
  if (interp.AllocateTensors() != kTfLiteOk) {
    while (true) {
      watchdog_update();
      TRACE("FATAL: AllocateTensors failed\n");
      sleep_ms(2000);
    }
  }
  TRACE("arena used %u bytes\n",
        static_cast<unsigned>(interp.arena_used_bytes()));

  TfLiteTensor* in = interp.input(0);
  TRACE("input type=%d bytes=%u\n", static_cast<int>(in->type),
        static_cast<unsigned>(in->bytes));

  // Real latents for utterance 0, tile 0 -- identical to what the full
  // TTS app feeds (PbDecoder::DecodeTileAt): sum of int8 codebook rows,
  // requantized to the graph's int16 input scale. Invoke with zeros runs
  // clean; this tests whether REAL activations trigger the corruption.
  {
    const PbDemoUtterance& utt = kPbUtterances[0];
    const int8_t* codebooks[3] = {g_pb_codebook0, g_pb_codebook1,
                                  g_pb_codebook2};
    const float* scales[3] = {g_pb_codebook0_scale, g_pb_codebook1_scale,
                              g_pb_codebook2_scale};
    int16_t* dst_base = in->data.i16;
    for (int j = 0; j < kPbTileLatents; ++j) {
      int16_t* dst = dst_base + j * kPbLatentDim;
      if (j >= utt.n_latents) {
        memset(dst, 0, sizeof(int16_t) * kPbLatentDim);
        continue;
      }
      float q[kPbLatentDim];
      for (int d = 0; d < kPbLatentDim; ++d) q[d] = 0.0f;
      for (int s = 0; s < kPbStages; ++s) {
        const int code = utt.codes[j * kPbStages + s];
        const int8_t* row = codebooks[s] + code * kPbLatentDim;
        for (int d = 0; d < kPbLatentDim; ++d) q[d] += row[d] * scales[s][d];
      }
      for (int d = 0; d < kPbLatentDim; ++d) {
        float v = q[d] / kPbInputScale;
        v = v < 0.0f ? v - 0.5f : v + 0.5f;
        int iv = static_cast<int>(v);
        if (iv > 32767) iv = 32767;
        if (iv < -32768) iv = -32768;
        dst[d] = static_cast<int16_t>(iv);
      }
    }
    TRACE("loaded real latents (utt 0, %d latents)\n", utt.n_latents);
  }

  TRACE("Invoke...\n");
  static repeating_timer_t wd_timer;
  add_repeating_timer_ms(1000, FeedWatchdog, nullptr, &wd_timer);
  for (unsigned round = 0;; ++round) {
    watchdog_update();
    profiler.Reset();
    const uint32_t t0 = time_us_32();
    const TfLiteStatus status = interp.Invoke();
    const uint32_t el = time_us_32() - t0;
    TRACE("round %u: Invoke returned %d in %lu us\n", round,
          static_cast<int>(status), static_cast<unsigned long>(el));
    profiler.Report();
    watchdog_update();
    sleep_ms(1000);
  }
}
