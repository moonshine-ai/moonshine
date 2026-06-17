/*
 * cpp-tiny RP2350 dual-core dispatch (shared across CMSIS-NN kernels).
 *
 * NOT part of upstream CMSIS-NN. Added for the moonshine-spelling
 * cpp-tiny port to let more than one kernel (the int8 SIMD GEMM and the
 * 3x3 depthwise conv) hand half their work to a persistent worker on
 * RP2350 core 1. The implementation lives in
 * Source/NNSupportFunctions/arm_nn_mat_mult_nt_t_s8.c (already compiled);
 * this header just exposes the generic dispatch so other kernels can
 * reuse the same single core-1 worker.
 *
 * Only one task runs at a time (TFLM executes ops sequentially on core
 * 0), so there is no queue: dispatch one task, compute the other half on
 * core 0, then wait. Pair every spelling_mc_dispatch_task() with exactly
 * one spelling_mc_wait().
 */

#ifndef SPELLING_MULTICORE_H
#define SPELLING_MULTICORE_H

#if defined(SPELLING_TINY_MULTICORE) && defined(ARM_MATH_DSP)

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*spelling_mc_task_fn)(void *arg);

// Launch core 1 (once, lazily) and hand it `fn(arg)`. Returns
// immediately so the caller can do the other half in parallel. `arg`
// must stay valid until spelling_mc_wait() returns.
void spelling_mc_dispatch_task(spelling_mc_task_fn fn, void *arg);

// Block until the dispatched task finishes (and observe its writes).
void spelling_mc_wait(void);

#ifdef __cplusplus
}
#endif

#endif  // SPELLING_TINY_MULTICORE && ARM_MATH_DSP

#endif  // SPELLING_MULTICORE_H
