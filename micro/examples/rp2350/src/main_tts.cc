// Entry point for the standalone TTS service (`moonshine_micro_tts`).
//
// Boots the board and drops straight into the USB CDC speak loop
// (tts_service.cc): SPEAK/IPA in, "AUDIO <rate> <n>" + int16 PCM out.
// Drive it with examples/rp2350/scripts/tts_speak.py. Self-contained: no
// VAD/STT models, just the neural TTS engine + its flash pack.

#include <cstdio>

#include "hardware/clocks.h"
#include "hardware/vreg.h"
#include "hardware/watchdog.h"
#include "pico/stdlib.h"
#include "tts_service.h"

namespace {

// Working memory for the synthesizer (TFLM decoder arena + unit selection +
// control track). Same budget the recognition apps lend it.
constexpr std::size_t kArenaBytes = 340u * 1024u;
alignas(16) uint8_t g_arena[kArenaBytes];

}  // namespace

// Progress hooks called from the synthesis pipeline (pb_decoder /
// worldlite_synth, see neural-tts/src/hooks.cc for the no-op defaults).
// Same recovery scheme as the step7 bring-up ladder: every checkpoint
// feeds the hardware watchdog and stashes its argument in watchdog
// scratch registers, so a lockup mid-synthesis auto-reboots after 8 s
// and the boot banner reports the last checkpoint reached instead of
// wedging the board until a manual BOOTSEL power cycle. The service's
// idle loop (tts_service.cc ReadLine) also feeds the watchdog.
//
// NB: scratch[4..7] belong to watchdog_reboot()/the bootrom (reboot
// vector magic; the bootrom zeroes them on use), so the post-mortem data
// must live in scratch[0..3].
extern "C" void tts_checkpoint(uint32_t v) {
  watchdog_hw->scratch[0] = v;
  watchdog_update();
}
extern "C" void tts_checkpoint2(uint32_t v) { watchdog_hw->scratch[1] = v; }
extern "C" void tts_trace(uint32_t tag, uint32_t val) {
  watchdog_hw->scratch[2] = (tag << 24) | (val & 0xFFFFFFu);
}

// Hard fault post-mortem: stash the stacked (faulting) PC in scratch[3]
// and reboot immediately instead of waiting out the watchdog. STATUS
// reports it; map the address via moonshine_micro_tts.dis.
extern "C" void HardFaultC(uint32_t* frame) {
  watchdog_hw->scratch[3] = frame[6];  // stacked PC
  watchdog_reboot(0, 0, 1);
  for (;;) {
  }
}
extern "C" __attribute__((naked)) void isr_hardfault(void) {
  __asm volatile(
      "tst lr, #4\n"
      "ite eq\n"
      "mrseq r0, msp\n"
      "mrsne r0, psp\n"
      "b HardFaultC\n");
}

int main() {
  // 250 MHz overclock (core voltage first); synthesis is compute-bound.
  vreg_set_voltage(VREG_VOLTAGE_1_20);
  sleep_ms(10);
  set_sys_clock_khz(250000, true);

  stdio_init_all();
  for (int i = 0; i < 50 && !stdio_usb_connected(); ++i) sleep_ms(20);
  sleep_ms(200);

  printf("\n[boot] neural TTS service, clk_sys=%lu MHz\n",
         static_cast<unsigned long>(clock_get_hz(clk_sys) / 1000000u));
  spelling::BootReport report = {};
  report.watchdog_reboot = watchdog_caused_reboot();
  report.ckpt = watchdog_hw->scratch[0];
  report.ckpt2 = watchdog_hw->scratch[1];
  report.trace = watchdog_hw->scratch[2];
  report.fault_pc = watchdog_hw->scratch[3];
  spelling::SetBootReport(report);  // queryable later via STATUS
  if (report.watchdog_reboot) {
    printf("[boot] WATCHDOG REBOOT: ckpt=%lu ckpt2=%lu trace=%08lx "
           "fault_pc=%08lx\n",
           static_cast<unsigned long>(report.ckpt),
           static_cast<unsigned long>(report.ckpt2),
           static_cast<unsigned long>(report.trace),
           static_cast<unsigned long>(report.fault_pc));
  }
  fflush(stdout);
  watchdog_hw->scratch[0] = 0;
  watchdog_hw->scratch[1] = 0;
  watchdog_hw->scratch[2] = 0;
  watchdog_hw->scratch[3] = 0;
  watchdog_enable(8000, true);  // pause_on_debug

  spelling::RunTtsService(g_arena, kArenaBytes);
}
