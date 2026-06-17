/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "signal/src/log.h"

#include "signal/src/msb.h"

namespace tflite {
namespace tflm_signal {
namespace {

const uint16_t kLogLut[] = {
    0,    224,  442,  654,  861,  1063, 1259, 1450, 1636, 1817, 1992, 2163,
    2329, 2490, 2646, 2797, 2944, 3087, 3224, 3358, 3487, 3611, 3732, 3848,
    3960, 4068, 4172, 4272, 4368, 4460, 4549, 4633, 4714, 4791, 4864, 4934,
    5001, 5063, 5123, 5178, 5231, 5280, 5326, 5368, 5408, 5444, 5477, 5507,
    5533, 5557, 5578, 5595, 5610, 5622, 5631, 5637, 5640, 5641, 5638, 5633,
    5626, 5615, 5602, 5586, 5568, 5547, 5524, 5498, 5470, 5439, 5406, 5370,
    5332, 5291, 5249, 5203, 5156, 5106, 5054, 5000, 4944, 4885, 4825, 4762,
    4697, 4630, 4561, 4490, 4416, 4341, 4264, 4184, 4103, 4020, 3935, 3848,
    3759, 3668, 3575, 3481, 3384, 3286, 3186, 3084, 2981, 2875, 2768, 2659,
    2549, 2437, 2323, 2207, 2090, 1971, 1851, 1729, 1605, 1480, 1353, 1224,
    1094, 963,  830,  695,  559,  421,  282,  142,  0,    0};

// Number of segments in the log lookup table. The table will be kLogSegments+1
// in length (with some padding).
// constexpr int kLogSegments = 128;
constexpr int kLogSegmentsLog2 = 7;

// Scale used by lookup table.
constexpr int kLogScale = 65536;
constexpr int kLogScaleLog2 = 16;
constexpr int kLogCoeff = 45426;

uint32_t Log2FractionPart32(uint32_t x, uint32_t log2x) {
  // Part 1
  int32_t frac = x - (1LL << log2x);
  if (log2x < kLogScaleLog2) {
    frac <<= kLogScaleLog2 - log2x;
  } else {
    frac >>= log2x - kLogScaleLog2;
  }
  // Part 2
  const uint32_t base_seg = frac >> (kLogScaleLog2 - kLogSegmentsLog2);
  const uint32_t seg_unit = (UINT32_C(1) << kLogScaleLog2) >> kLogSegmentsLog2;

  // ASSERT(base_seg < kLogSegments);
  const int32_t c0 = kLogLut[base_seg];
  const int32_t c1 = kLogLut[base_seg + 1];
  const int32_t seg_base = seg_unit * base_seg;
  const int32_t rel_pos = ((c1 - c0) * (frac - seg_base)) >> kLogScaleLog2;
  return frac + c0 + rel_pos;
}

}  // namespace

// Calculate integer logarithm, 32 Bit version
uint32_t Log32(uint32_t x, uint32_t out_scale) {
  // ASSERT(x != 0);
  const uint32_t integer = MostSignificantBit32(x) - 1;
  const uint32_t fraction = Log2FractionPart32(x, integer);
  const uint32_t log2 = (integer << kLogScaleLog2) + fraction;
  const uint32_t round = kLogScale / 2;
  const uint32_t loge =
      ((static_cast<uint64_t>(kLogCoeff)) * log2 + round) >> kLogScaleLog2;
  // Finally scale to our output scale
  const uint32_t loge_scaled = (out_scale * loge + round) >> kLogScaleLog2;
  return loge_scaled;
}

}  // namespace tflm_signal
}  // namespace tflite
