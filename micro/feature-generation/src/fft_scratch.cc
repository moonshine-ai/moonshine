#include "fft_scratch.h"

namespace spelling {

float g_fft_scratch_frame[kFftScratchNFft];
kiss_fft_cpx g_fft_scratch_spec[kFftScratchNFreq];
float g_fft_scratch_pow[kFftScratchNFreq];

}  // namespace spelling
