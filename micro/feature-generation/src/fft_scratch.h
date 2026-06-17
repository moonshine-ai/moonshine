// Shared FFT scratch pool for the on-device log-mel front-ends.
//
// WHY THIS EXISTS: on the RP2350 each core's stack is a single 4 KB scratch
// bank (core 0 -> SCRATCH_Y at 0x20081000, core 1 -> SCRATCH_X at 0x20080000,
// immediately below). A per-frame FFT working set of frame_buf[512] +
// spectrum[257] + power_row[257] is ~5 KB, and the historical "upper bound"
// sizes ([1024]/[513]) were ~10 KB. Placing that on the stack overflows core
// 0's 4 KB bank straight down into core 1's stack. CMSIS-NN runs a persistent
// dual-core GEMM worker on core 1 during TFLM Invoke(), so the two stacks
// collide and silently corrupt data -- which showed up as NaN VAD features and
// a garbage-initialised STT LogMelSpectrogram. See
// https://petewarden.com/2024/01/16/understanding-the-raspberry-pi-picos-memory-layout/
//
// Keeping the scratch in .bss removes it from the stack entirely. Both the VAD
// mel streamer (MelStreamer::PushHop) and the STT front-end
// (LogMelSpectrogram::Compute) borrow this single pool. That is safe because
// they run ONLY on core 0, sequentially (VAD listening vs STT phase), and
// neither is ever re-entered.

#ifndef FEATURE_GENERATION_FFT_SCRATCH_H_
#define FEATURE_GENERATION_FFT_SCRATCH_H_

#include "kiss_fft.h"  // kiss_fft_cpx

namespace spelling {

// Both front-ends use n_fft == 512 (kVadNFft == kNFft). The pool is sized to
// exactly that; callers must assert n_fft <= kFftScratchNFft.
constexpr int kFftScratchNFft = 512;
constexpr int kFftScratchNFreq = kFftScratchNFft / 2 + 1;  // 257

extern float g_fft_scratch_frame[kFftScratchNFft];
extern kiss_fft_cpx g_fft_scratch_spec[kFftScratchNFreq];
extern float g_fft_scratch_pow[kFftScratchNFreq];

}  // namespace spelling

#endif  // FEATURE_GENERATION_FFT_SCRATCH_H_
