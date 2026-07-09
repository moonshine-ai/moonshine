# Local patches to pico-tflmicro

This tree is vendored from
[raspberrypi/pico-tflmicro](https://github.com/raspberrypi/pico-tflmicro) (TFLM
under Apache 2.0, the Pico wrapper code MIT). It is **not pristine** — it carries
a small, platform-specific (RP2350) performance patch plus the usual
subdirectory-friendly trimming. Anything depending on TFLM being agnostic should
go through the `tflm` CMake INTERFACE target in the top-level `CMakeLists.txt`,
not reach in here.

## 1. Dual-core CMSIS-NN GEMM / depthwise split (RP2350-specific)

Gated behind the `SPELLING_TINY_MULTICORE` compile definition (set by the parent
build; ON by default). It splits the two hottest int8 kernels across both
Cortex-M33 cores for a measured ~1.6x inference speedup with bit-identical
output. Files touched:

- `src/third_party/cmsis_nn/Include/spelling_multicore.h` — **added.** The
  core-1 persistent-worker dispatch API (`spelling_mc_dispatch_task()` etc.).
- `src/third_party/cmsis_nn/Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c`
  — patched to partition the output channels across the two cores under
  `SPELLING_TINY_MULTICORE` (`spelling_mm_dsp_rows`, `__not_in_flash_func`).
- `src/third_party/cmsis_nn/Source/ConvolutionFunctions/arm_depthwise_conv_3x3_s8.c`
  — patched to partition the output rows across the two cores
  (`spelling_dw3x3_rows`).

When `SPELLING_TINY_MULTICORE` is undefined these fall back to the upstream
single-core scalar/SIMD paths, so the patch is inert in a single-core build.

Because this is platform-specific, it is the one piece of "platform code" that
could not be lifted out into `examples/rp2350/` — it is physically interleaved
with the CMSIS-NN kernels. A port to another platform would simply leave
`SPELLING_TINY_MULTICORE` undefined.

## 2. Subdirectory-friendly CMakeLists

The upstream top-level `project()` / `pico_sdk_init()` calls and the
example/test subdirectory blocks were removed so this builds as an
`add_subdirectory()` of the parent (which does both). See the comment at the top
of `CMakeLists.txt`.
